/* MPI C worker: pair-aware decomposition, derived datatypes, collectives,
 * nonblocking central state exchange, distributed CG and data-parallel SGD.
 * MPI calls remain on the main thread (MPI_THREAD_FUNNELED). */
#include "hft/api.h"
#include "hft/simulation.h"
#include <math.h>
#include <mpi.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef HFT_HAVE_OPENMP
#include <omp.h>
#endif
static MPI_Datatype record_type(int which) {
  MPI_Datatype t;
  int blocks[5] = {1, 1, 1, 1, 4};
  MPI_Aint offsets[5];
  MPI_Datatype types[5] = {MPI_UINT64_T, MPI_INT64_T, MPI_UINT32_T,
                           MPI_UINT32_T, MPI_INT};
  MPI_Aint size;
  if (which == 0) {
    blocks[4] = 8;
    types[4] = MPI_BYTE;
    offsets[0] = offsetof(HftEvent, ts);
    offsets[1] = offsetof(HftEvent, price);
    offsets[2] = offsetof(HftEvent, id);
    offsets[3] = offsetof(HftEvent, symbol);
    offsets[4] = offsetof(HftEvent, qty);
    size = sizeof(HftEvent);
  } else if (which == 1) {
    offsets[0] = offsetof(HftOrder, ts);
    offsets[1] = offsetof(HftOrder, price);
    offsets[2] = offsetof(HftOrder, id);
    offsets[3] = offsetof(HftOrder, symbol);
    offsets[4] = offsetof(HftOrder, qty);
    size = sizeof(HftOrder);
  } else {
    offsets[0] = offsetof(HftFill, ts);
    offsets[1] = offsetof(HftFill, price);
    offsets[2] = offsetof(HftFill, id);
    offsets[3] = offsetof(HftFill, symbol);
    offsets[4] = offsetof(HftFill, qty);
    size = sizeof(HftFill);
  }
  MPI_Datatype raw;
  MPI_Type_create_struct(5, blocks, offsets, types, &raw);
  MPI_Type_create_resized(raw, 0, size, &t);
  MPI_Type_commit(&t);
  MPI_Type_free(&raw);
  return t;
}
static void distributed_cg(int rank, int ranks) {
  int n = 128, local = (n + ranks - 1 - rank) / ranks;
  double x[128] = {0}, r[128], p[128], ap[128], mine[128];
  for (int i = 0; i < n; ++i)
    r[i] = p[i] = 1;
  double rr = n;
  int iter;
  double start = MPI_Wtime();
  for (iter = 0; iter < 256 && sqrt(rr) > 1e-10; ++iter) {
    memset(mine, 0, sizeof(mine));
    for (int q = 0; q < local; ++q) {
      int i = rank + q * ranks;
      if (i < n)
        mine[i] = 4 * p[i] - (i ? p[i - 1] : 0) - (i + 1 < n ? p[i + 1] : 0);
    }
    MPI_Allreduce(mine, ap, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double dot = 0, global;
    for (int i = rank; i < n; i += ranks)
      dot += p[i] * ap[i];
    MPI_Allreduce(&dot, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double alpha = rr / global, sum = 0, newrr;
    for (int i = 0; i < n; ++i) {
      x[i] += alpha * p[i];
      r[i] -= alpha * ap[i];
      if (i % ranks == rank)
        sum += r[i] * r[i];
    }
    MPI_Allreduce(&sum, &newrr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    for (int i = 0; i < n; ++i)
      p[i] = r[i] + newrr / rr * p[i];
    rr = newrr;
  }
  if (!rank)
    printf("{\"type\":\"cg\",\"seconds\":%.9g,\"iterations\":%d,\"residual\":%."
           "12g}\n",
           MPI_Wtime() - start, iter, sqrt(rr));
}
static int distributed_train(HftConfig *c, int rank, int ranks,
                             const char *prefix) {
  HftDataset d = {0};
  char error[HFT_ERROR] = {0};
  int ok = 1;
  if (!rank)
    ok = hft_dataset_make(c, c->train_samples, &d, error);
  MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (!ok)
    return 0;
  MPI_Bcast(&d.count, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&d.train_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank) {
    d.features = calloc((size_t)d.count * 32, sizeof(float));
    d.labels = calloc((size_t)d.count, sizeof(int));
  }
  if (!d.features || !d.labels)
    MPI_Abort(MPI_COMM_WORLD, 2);
  MPI_Bcast(d.features, d.count * 32, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(d.labels, d.count, MPI_INT, 0, MPI_COMM_WORLD);
  float *g = calloc(HFT_WEIGHTS, sizeof(float)),
        *total = calloc(HFT_WEIGHTS, sizeof(float));
  if (!g || !total)
    MPI_Abort(MPI_COMM_WORLD, 2);
  double start = MPI_Wtime();
  for (int ep = 0; ep < c->train_epochs; ++ep)
    for (int begin = 0; begin < d.train_count; begin += 32) {
      memset(g, 0, HFT_WEIGHTS * sizeof(float));
      int size = d.train_count - begin;
      if (size > 32)
        size = 32;
      for (int i = rank; i < size; i += ranks)
        hft_mlp_gradient(&c->model, d.features + (begin + i) * 32,
                         d.labels[begin + i], g);
      MPI_Allreduce(g, total, HFT_WEIGHTS, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
      for (int w = 0; w < HFT_WEIGHTS; ++w)
        c->model.weights[w] -=
            .005f * (total[w] / size + .0001f * c->model.weights[w]);
    }
  c->model.trained = 1;
  if (!rank) {
    char path[2048];
    snprintf(path, sizeof(path), "%s-model.bin", prefix);
    ok = hft_model_save(&c->model, path, error);
    printf("{\"type\":\"training\",\"seconds\":%.9g,\"saved\":%s}\n",
           MPI_Wtime() - start, ok ? "true" : "false");
  }
  free(g);
  free(total);
  hft_dataset_free(&d);
  return ok;
}
int main(int argc, char **argv) {
  int provided, rank, ranks;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  HftConfig c;
  hft_config_default(&c);
  c.engine = HFT_MPI;
  char error[HFT_ERROR] = {0};
  const char *prefix = "mpi-result";
  int cg = 0, train = 0, ok = 1;
  if (!rank) {
    for (int i = 1; i < argc; ++i) {
      if (!strcmp(argv[i], "--cg")) {
        cg = 1;
        continue;
      }
      if (!strcmp(argv[i], "--train")) {
        train = 1;
        continue;
      }
      if (i + 1 == argc) {
        ok = 0;
        break;
      }
      const char *k = argv[i], *v = argv[++i];
      if (!strcmp(k, "--config"))
        ok = hft_config_load(&c, v, error);
      else if (!strcmp(k, "--out"))
        prefix = v;
      else if (!strcmp(k, "--symbols"))
        c.symbols = atoi(v);
      else if (!strcmp(k, "--events"))
        c.events = strtoull(v, NULL, 10);
      else if (!strcmp(k, "--threads")) {
        c.threads = atoi(v);
        c.engine = HFT_HYBRID;
      } else if (!strcmp(k, "--topology"))
        c.topology = !strcmp(v, "central");
      else
        ok = 0;
    }
    ok = ok && hft_config_validate(&c, error);
    if (!ok)
      fprintf(stderr, "%s\n", error);
  }
  MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (!ok) {
    MPI_Finalize();
    return 2;
  }
  if (c.engine != HFT_HYBRID)
    c.engine = HFT_MPI;
  MPI_Bcast(&c, (int)sizeof(c), MPI_BYTE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cg, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&train, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (cg) {
    distributed_cg(rank, ranks);
    MPI_Finalize();
    return 0;
  }
  if (train) {
    ok = distributed_train(&c, rank, ranks, prefix);
    MPI_Finalize();
    return ok ? 0 : 1;
  }
  MPI_Datatype eventtype = record_type(0), ordertype = record_type(1),
               filltype = record_type(2);
  int groups = (c.symbols + 1) / 2;
  int *counts = calloc((size_t)ranks, sizeof(int)),
      *displs = calloc((size_t)ranks, sizeof(int)),
      *ids = calloc((size_t)groups, sizeof(int));
  if (!counts || !displs || !ids)
    MPI_Abort(MPI_COMM_WORLD, 2);
  int at = 0;
  for (int r = 0; r < ranks; ++r) {
    displs[r] = at;
    for (int g = c.partition ? groups * r / ranks : r;
         g < (c.partition ? groups * (r + 1) / ranks : groups);
         g += c.partition ? 1 : ranks) {
      ids[at++] = g * 2;
      counts[r]++;
    }
  }
  int own = counts[rank];
  int *mine = calloc((size_t)(own ? own : 1), sizeof(int));
  MPI_Scatterv(ids, counts, displs, MPI_INT, mine, own, MPI_INT, 0,
               MPI_COMM_WORLD);
  HftConfig local = c;
  local.symbols = own ? own * 2 : 1;
  if (own && mine[own - 1] + 1 == c.symbols)
    local.symbols--;
  if (!own)
    local.strategy = HFT_SMA;
  HftResult r = {0};
  if (!hft_initialize(&local, &r))
    MPI_Abort(MPI_COMM_WORLD, 2);
  for (int g = 0; g < own; ++g)
    for (int j = 0; j < 2 && 2 * g + j < local.symbols; ++j) {
      HftSymbol *s = r.states + 2 * g + j;
      s->symbol = mine[g] + j;
      for (int id = 1; id <= c.capacity; ++id)
        s->book.nodes[id].order.symbol = (hft_u32)s->symbol;
    }
  int local_symbols = own ? local.symbols : 0;
  int *bytes = calloc((size_t)ranks, sizeof(int)),
      *byteoffset = calloc((size_t)ranks, sizeof(int));
  int mybytes = local_symbols * (int)sizeof(HftSymbol);
  MPI_Gather(&mybytes, 1, MPI_INT, bytes, 1, MPI_INT, 0, MPI_COMM_WORLD);
  int total = 0;
  if (!rank)
    for (int i = 0; i < ranks; ++i) {
      byteoffset[i] = total;
      total += bytes[i];
    }
  unsigned char *gathered = rank ? NULL : malloc((size_t)total);
  if (!rank && !gathered)
    MPI_Abort(MPI_COMM_WORLD, 2);
  double begin = MPI_Wtime(), comm = 0;
  hft_u64 done = 0;
#ifdef HFT_HAVE_OPENMP
  omp_set_num_threads(c.engine == HFT_HYBRID ? c.threads : 1);
#endif
  while (done < c.events) {
    hft_u64 steps = c.events - done;
    if (steps > (hft_u64)c.batch)
      steps = c.batch;
    hft_u64 delay = c.mode ? (hft_u64)c.network_ns : 0;
#ifdef HFT_HAVE_OPENMP
#pragma omp parallel for schedule(static) if (c.engine == HFT_HYBRID)
#endif
    for (int g = 0; g < own; ++g)
      hft_group_inline(r.states, g * 2, local_symbols, &c, steps, 0, delay);
    done += steps;
    double a = MPI_Wtime();
    if (c.topology &&
        ranks > 1) { /* Deliberately centralized state exchange: root drains all
                      * rank books every batch. Nonblocking sends plus explicit
                      * waits preserve buffer lifetime. */
      MPI_Request request;
      if (rank) {
        MPI_Isend(r.states, mybytes, MPI_BYTE, 0, 18, MPI_COMM_WORLD, &request);
        MPI_Wait(&request, MPI_STATUS_IGNORE);
        int ack;
        MPI_Recv(&ack, 1, MPI_INT, 0, 19, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      } else
        for (int peer = 1; peer < ranks; ++peer) {
          MPI_Irecv(gathered + byteoffset[peer], bytes[peer], MPI_BYTE, peer,
                    18, MPI_COMM_WORLD, &request);
          MPI_Wait(&request, MPI_STATUS_IGNORE);
          MPI_Send(&peer, 1, MPI_INT, peer, 19, MPI_COMM_WORLD);
        }
    }
    hft_u64 completed = done * (hft_u64)local_symbols, global;
    MPI_Allreduce(&completed, &global, 1, MPI_UINT64_T, MPI_SUM,
                  MPI_COMM_WORLD);
    comm += MPI_Wtime() - a;
    if (!rank && (done == c.events || done % (hft_u64)(c.batch * 16) == 0)) {
      printf("{\"type\":\"progress\",\"fraction\":%.6f,\"events\":\"%llu\"}\n",
             (double)done / c.events, (unsigned long long)global);
      fflush(stdout);
    }
  }
  double elapsed = MPI_Wtime() - begin, maximum;
  MPI_Reduce(&elapsed, &maximum, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  double communication;
  MPI_Reduce(&comm, &communication, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Gatherv(r.states, mybytes, MPI_BYTE, gathered, bytes, byteoffset,
              MPI_BYTE, 0, MPI_COMM_WORLD);
  HftFill last = {0};
  if (own)
    last = r.states[0].last_fill;
  HftFill *fills = rank ? NULL : calloc((size_t)ranks, sizeof(HftFill));
  MPI_Gather(&last, 1, filltype, fills, 1, filltype, 0, MPI_COMM_WORLD);
  free(fills);
  if (!rank) {
    HftResult result = {0};
    if (!hft_initialize(&c, &result))
      MPI_Abort(MPI_COMM_WORLD, 2);
    for (int peer = 0; peer < ranks; ++peer) {
      HftSymbol *part = (HftSymbol *)(gathered + byteoffset[peer]);
      for (int s = 0; s < bytes[peer] / (int)sizeof(HftSymbol); ++s)
        result.states[part[s].symbol] = part[s];
    }
    result.seconds = maximum;
    result.communication_seconds = communication;
    result.workers = ranks * (c.engine == HFT_HYBRID ? c.threads : 1);
    result.rank_count = ranks;
    result.point_count = result.states[0].trace_count;
    for (int i = 0; i < result.point_count; ++i) {
      hft_i64 pnl = 0;
      for (int symbol = 0; symbol < c.symbols; ++symbol) {
        HftSymbol *st = result.states + symbol;
        int at = (st->trace_head - st->trace_count + i + HFT_TRACE) % HFT_TRACE;
        pnl += st->equity[at];
      }
      result.points[i].pnl = pnl;
      result.points[i].event =
          ((c.events - 1) / (hft_u64)c.sample_every * (hft_u64)c.sample_every +
           1 - (hft_u64)(result.point_count - 1 - i) * c.sample_every) *
          c.symbols;
    }
    hft_result_finalize(&result);
    HftConfig serial = c;
    serial.engine = HFT_SERIAL;
    serial.mode = 0;
    HftResult reference = {0};
    if (hft_run(&serial, &reference, NULL, NULL, NULL)) {
      result.verified = !c.mode && hft_result_equivalent(&result, &reference);
      result.speedup_serial = reference.seconds / result.seconds;
      result.efficiency = result.speedup_serial / result.workers;
      result.latency_cost = c.mode ? result.pnl - reference.pnl : 0;
    }
    hft_result_free(&reference);
    ok = hft_result_export(&result, prefix, error);
    printf("{\"type\":\"result\",\"ok\":%s,\"verified\":%s,\"seconds\":%.9g,"
           "\"ranks\":%d,\"hash\":\"%016llx\"}\n",
           ok ? "true" : "false", result.verified ? "true" : "false",
           result.seconds, ranks, (unsigned long long)result.hash);
    if (!result.verified && !c.mode)
      ok = 0;
    hft_result_free(&result);
  }
  MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
  free(gathered);
  free(bytes);
  free(byteoffset);
  free(mine);
  free(counts);
  free(displs);
  free(ids);
  hft_result_free(&r);
  MPI_Type_free(&eventtype);
  MPI_Type_free(&ordertype);
  MPI_Type_free(&filltype);
  MPI_Finalize();
  return ok ? 0 : 1;
}
