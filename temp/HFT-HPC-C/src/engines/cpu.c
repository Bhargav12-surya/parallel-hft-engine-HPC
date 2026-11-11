/* Purpose: serial, pthread/Win32 worker pipeline and OpenMP engines in C.
 * Units 1/2: queue ownership, affinity, symbol/pair decomposition, scheduling.
 * Each worker exclusively updates a pair. Host reduces after the batch barrier.
 * Complexity: O(events*symbols), bounded state. Parallelism never reorders a
 * pair. */
#include "hft/api.h"
#include "hft/simulation.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef HFT_HAVE_OPENMP
#include <omp.h>
#endif
void hft_step_group(HftSymbol *states, int first, const HftConfig *c,
                    hft_u64 steps, int naive, hft_u64 delay) {
  hft_group_inline(states, first, c->symbols, c, steps, naive, delay);
}
int hft_initialize(const HftConfig *c, HftResult *r) {
  int s, i, side;
  memset(r, 0, sizeof(*r));
  r->config = *c;
  if (!hft_config_validate(c, r->error))
    return 0;
  r->states = (HftSymbol *)calloc((size_t)c->symbols, sizeof(HftSymbol));
  r->points = (HftPoint *)calloc(2049, sizeof(HftPoint));
  if (!r->states || !r->points) {
    snprintf(r->error, HFT_ERROR, "Not enough memory for %d symbols",
             c->symbols);
    return 0;
  }
  hft_system_info(r->hardware, sizeof(r->hardware));
  {
    time_t now = time(NULL);
    struct tm value;
#ifdef _WIN32
    gmtime_s(&value, &now);
#else
    gmtime_r(&now, &value);
#endif
    strftime(r->timestamp, sizeof(r->timestamp), "%Y-%m-%dT%H:%M:%SZ", &value);
  }
  for (s = 0; s < c->symbols; ++s) {
    HftSymbol *st = &r->states[s];
    st->symbol = s;
    st->regime = c->regime;
    st->hash = st->market_hash = 14695981039346656037UL;
    st->fundamental = hft_log((double)c->initial_price);
    st->previous_mid = c->initial_price;
    hft_book_clear(&st->book, c);
    for (i = 1; i <= 16; ++i)
      for (side = 0; side < 2; ++side) {
        HftOrder o;
        o.id = (hft_u32)(2 * i - 1 + side);
        o.symbol = (hft_u32)s;
        o.price = c->initial_price + (side ? i : -i);
        o.ts = 0;
        o.qty = 50;
        o.side = side;
        o.agent = 1;
        o.ioc = 0;
        hft_submit(st, c, o, 0, 0);
      }
  }
  return 1;
}
typedef struct {
  HftThread thread;
  HftQueue jobs;
  HftAtomic completed, stop;
  HftResult *r;
  const HftConfig *cfg;
  hft_u64 steps, delay;
  int id;
  double seconds, overhead;
  HftHistogram latency;
} Worker;
static void *work(void *pointer) {
  Worker *w = (Worker *)pointer;
  hft_pin_thread(w->id);
  /* One producer fills this queue; this worker alone writes each dispatched
   * pair. Completed release publishes updates; host acquire waits before
   * snapshot/reduction. */
  while (!hft_load(&w->stop)) {
    int first;
    if (hft_queue_pop(&w->jobs, &first)) {
      double start = hft_now();
      for (hft_u64 e = 0; e < w->steps; ++e) {
        double a = hft_now();
        hft_step_group(w->r->states, first, w->cfg, 1, 0, w->delay);
        hft_hist_record(
            &w->latency,
            (hft_u64)hft_max(0, (hft_now() - a - w->overhead) * 1e9));
      }
      w->seconds += hft_now() - start;
      hft_add(&w->completed, 1);
    } else
      hft_sleep(0);
  }
  return NULL;
}
static void record_progress(HftResult *r, hft_u64 done, HftProgress progress,
                            void *user, double begin) {
  int s;
  hft_i64 pnl = 0;
  for (s = 0; s < r->config.symbols; ++s) {
    HftTop t = hft_top(&r->states[s].book, &r->config, 0);
    pnl += r->states[s].position.cash +
           (hft_i64)r->states[s].position.qty * hft_mid(t);
  }
  if (r->point_count < 2049) {
    HftPoint *p = &r->points[r->point_count++];
    p->event = done * (hft_u64)r->config.symbols;
    p->pnl = pnl;
    p->price = hft_mid(hft_top(&r->states[0].book, &r->config, 0));
    p->inventory = r->states[0].position.qty;
    p->regime = r->states[0].regime;
    p->selected = r->states[0].selected;
  }
  if (progress) {
    HftSnapshot *snap = (HftSnapshot *)malloc(sizeof(*snap));
    if (snap) {
      hft_make_snapshot(r, 0, snap);
      snap->progress = (double)done / (double)r->config.events;
      snap->events_second = (double)(done * (hft_u64)r->config.symbols) /
                            hft_max(1e-9, hft_now() - begin);
      progress(snap, user);
      free(snap);
    }
  }
}
int hft_run_cpu(const HftConfig *c, HftResult *r, HftAtomic *cancel,
                HftProgress progress, void *user) {
  int nworkers = 1, group, groups = (c->symbols + 1) / 2, i;
  hft_u64 done = 0, batches = 0, delay = 0;
  double begin, overhead = 1e9;
  Worker *workers = NULL;
  double *durations = NULL;
  HftHistogram *histograms = NULL;
  if (!hft_initialize(c, r))
    return 0;
  if (c->engine == HFT_THREADS || c->engine == HFT_OPENMP ||
      c->engine == HFT_HYBRID)
    nworkers = c->threads < groups ? c->threads : groups;
  r->workers = nworkers;
  r->worker_count = nworkers;
  r->worker_seconds = (double *)calloc((size_t)nworkers, sizeof(double));
  durations = (double *)calloc((size_t)nworkers, sizeof(double));
  histograms = (HftHistogram *)calloc((size_t)nworkers, sizeof(HftHistogram));
  if (!r->worker_seconds || !durations || !histograms) {
    snprintf(r->error, HFT_ERROR, "Worker allocation failed");
    goto fail;
  }
  for (i = 0; i < 1000; ++i) {
    double a = hft_now(), b = hft_now();
    if (b - a < overhead)
      overhead = b - a;
  }
  r->latency_overhead_ns = overhead * 1e9;
  r->latency_clock_resolution_ns = overhead * 1e9;
  if (c->engine == HFT_THREADS) {
    workers = (Worker *)calloc((size_t)nworkers, sizeof(*workers));
    if (!workers)
      goto fail;
    for (i = 0; i < nworkers; ++i) {
      workers[i].r = r;
      workers[i].cfg = c;
      workers[i].id = i;
      workers[i].overhead = overhead;
      if (!hft_thread_start(&workers[i].thread, work, &workers[i])) {
        int k;
        for (k = 0; k < i; ++k) {
          hft_store(&workers[k].stop, 1);
          hft_thread_join(workers[k].thread);
        }
        free(workers);
        workers = NULL;
        goto fail;
      }
    }
  }
#ifdef HFT_HAVE_OPENMP
  if (c->engine == HFT_OPENMP || c->engine == HFT_HYBRID) {
    omp_set_num_threads(nworkers);
    omp_set_schedule(c->schedule == 0 ? omp_sched_static
                                      : (c->schedule == 1 ? omp_sched_dynamic
                                                          : omp_sched_guided),
                     c->chunk);
  }
#endif
  begin = hft_now();
  while (done < c->events && (!cancel || !hft_load(cancel))) {
    hft_u64 steps = c->events - done;
    if (steps > (hft_u64)c->batch)
      steps = (hft_u64)c->batch;
    if (c->mode) {
      double estimate =
          batches ? (hft_now() - begin) / (double)(done * (hft_u64)c->symbols)
                  : 0;
      HftRandom rnd = hft_random(c->seed, 0, 501, batches);
      delay = (hft_u64)hft_max(
          0, estimate * 1e9 * c->latency_amplification + c->network_ns +
                 c->network_jitter_ns * hft_normal(rnd.x[0], rnd.x[1]));
    }
    if (workers) {
      for (i = 0; i < nworkers; ++i) {
        workers[i].steps = steps;
        workers[i].delay = delay;
        hft_store(&workers[i].completed, 0);
      }
      for (group = 0; group < groups; ++group)
        while (!hft_queue_push(&workers[group % nworkers].jobs, group * 2))
          hft_sleep(0);
      for (i = 0; i < nworkers; ++i) {
        hft_u64 target = (hft_u64)((groups + nworkers - 1 - i) / nworkers);
        while (hft_load(&workers[i].completed) < target)
          hft_sleep(0);
      }
    } else {
      /* cfg/model are shared read-only. group is private. Each iteration owns
       * two states, one duration and a thread-private histogram; implicit
       * barrier joins. */
#ifdef HFT_HAVE_OPENMP
#pragma omp parallel for schedule(runtime) if (c->engine == HFT_OPENMP ||      \
                                                   c->engine == HFT_HYBRID)
#endif
      for (group = 0; group < groups; ++group) {
        int tid = 0, first = group * 2;
        hft_u64 e;
        double start = hft_now();
#ifdef HFT_HAVE_OPENMP
        tid = omp_get_thread_num();
#endif
        for (e = 0; e < steps; ++e) {
          double a = hft_now(), b;
          hft_step_group(r->states, first, c, 1, c->engine == HFT_NAIVE, delay);
          b = hft_now();
          hft_hist_record(&histograms[tid],
                          (hft_u64)hft_max(0, (b - a - overhead) * 1e9));
        }
        durations[tid] += hft_now() - start;
      }
    }
    done += steps;
    ++batches;
    if (batches % (hft_u64)(1 + c->events / (hft_u64)c->batch / 1500) == 0 ||
        done == c->events)
      record_progress(r, done, progress, user, begin);
  }
  r->seconds = hft_now() - begin;
  r->cancelled = done < c->events;
  if (workers) {
    for (i = 0; i < nworkers; ++i) {
      hft_store(&workers[i].stop, 1);
      hft_thread_join(workers[i].thread);
      r->worker_seconds[i] = workers[i].seconds;
      hft_hist_merge(&r->latency, &workers[i].latency);
    }
    free(workers);
    workers = NULL;
    r->batch_latency_ns =
        r->seconds * 1e9 / (double)hft_max(1, (double)batches);
  } else {
    for (i = 0; i < nworkers; ++i)
      r->worker_seconds[i] = durations[i];
    for (i = 0; i < nworkers; ++i)
      hft_hist_merge(&r->latency, &histograms[i]);
  }
  free(durations);
  free(histograms);
  hft_result_finalize(r);
  return 1;
fail:
  free(durations);
  free(histograms);
  if (!r->error[0])
    snprintf(r->error, HFT_ERROR, "CPU engine initialization failed");
  return 0;
}
void hft_probe(int engine, HftCapability *cap) {
  memset(cap, 0, sizeof(*cap));
  cap->workers = hft_cores();
  snprintf(cap->name, sizeof(cap->name), "%s",
           engine >= 0 && engine < 8 ? hft_engine_names[engine] : "unknown");
  if (engine == HFT_NAIVE || engine == HFT_SERIAL || engine == HFT_THREADS)
    cap->available = 1;
  else if (engine == HFT_OPENMP || engine == HFT_HYBRID) {
#ifdef HFT_HAVE_OPENMP
    cap->available = 1;
#else
    snprintf(cap->reason, sizeof(cap->reason),
             "OpenMP C runtime was not found at build time");
#endif
  } else if (engine == HFT_MPI) {
#ifdef HFT_HAVE_MPI
    cap->available = 1;
#else
    snprintf(cap->reason, sizeof(cap->reason),
             "MPI C SDK/runtime was not found at build time");
#endif
  } else if (engine == HFT_OPENCL) {
    hft_opencl_probe(cap);
    return;
  } else if (engine == HFT_NVIDIA) {
    hft_nvidia_probe(cap);
    return;
  }
#ifndef HFT_HAVE_MPI
  if (engine == HFT_HYBRID) {
    cap->available = 0;
    snprintf(cap->reason, sizeof(cap->reason),
             "Hybrid requires both MPI C and OpenMP C");
  }
#endif
  if (cap->available)
    snprintf(cap->reason, sizeof(cap->reason), "Available");
}
int hft_run(const HftConfig *c, HftResult *r, HftAtomic *cancel,
            HftProgress progress, void *user) {
  HftCapability cap;
  memset(r, 0, sizeof(*r));
  if (!hft_config_validate(c, r->error))
    return 0;
  hft_probe(c->engine, &cap);
  if (!cap.available) {
    snprintf(r->error, HFT_ERROR, "%s", cap.reason);
    return 0;
  }
  if (c->engine == HFT_OPENCL)
    return hft_run_opencl(c, r, cancel, progress, user);
  if (c->engine == HFT_MPI || c->engine == HFT_HYBRID) {
    snprintf(r->error, HFT_ERROR,
             "Use hft_mpi_worker via mpiexec or the desktop MPI launcher");
    return 0;
  }
  if (c->engine == HFT_NVIDIA) {
    snprintf(r->error, HFT_ERROR,
             "NVIDIA libraries execute risk/matrix workloads; use scenario S4, "
             "not a tick-engine substitution");
    return 0;
  }
  return hft_run_cpu(c, r, cancel, progress, user);
}
