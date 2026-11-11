/* Batch CLI for actual S1-S8 measurements; CSV/JSON/SVG results. Unit 1.
 * Arguments are bounded and configuration is validated before execution. */
#include "cJSON.h"
#include "hft/api.h"
#include "hft/numeric.h"
#include <math.h>
#ifdef HFT_HAVE_OPENMP
#include <omp.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int indexof(const char *s, const char *const *names, int count) {
  for (int i = 0; i < count; ++i)
    if (!strcmp(s, names[i]))
      return i;
  return -1;
}
static int list(const char *s, int *values, int engines) {
  char temp[1024];
  if (strlen(s) >= sizeof(temp))
    return 0;
  strcpy(temp, s);
  int n = 0;
  char *token = strtok(temp, ",");
  while (token && n < 64) {
    int v = engines ? indexof(token, hft_engine_names, 8) : atoi(token);
    if (v < 0 || (!engines && v < 1))
      return 0;
    values[n++] = v;
    token = strtok(NULL, ",");
  }
  return n;
}
static int doublecmp(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}
static void usage(void) {
  puts("hft_bench --scenario S1..S8 --engines "
       "serial_naive,serial_opt,threaded,openmp,opencl\n  --symbols 1,64 "
       "--events 10000 --threads 1,2,4 --batch 1,64 --reps 5 --warmup 2\n  "
       "--strategy "
       "sma|ema|mean_reversion|breakout|rsi|macd|imbalance|market_making|pairs|"
       "execution|mlp|qlearning|selector\n  --seed 42 --mode verify|latency "
       "--network-ns 100000 --amplification 10\n  --config file.json --model "
       "weights.bin --out results --list --save-config file.json\nS4: "
       "covariance/CG/Cholesky/Monte Carlo; S5 inference; S6 MLP and Q "
       "training;\nS7 synchronization; S8 use mpiexec -n N hft_mpi_worker "
       "--config file --out prefix.");
}
static int analytics(HftConfig *c, int scenario, const int *engines, int ne,
                     const char *folder) {
  int n = 16, rows = 256;
  char path[2048], error[HFT_ERROR] = {0};
  snprintf(path, sizeof(path), "%s/S%d-analytics.csv", folder, scenario);
  FILE *f = fopen(path, "wb");
  if (!f)
    return 0;
  fprintf(f, "scenario,backend,kernel,seconds,value,detail\n");
  if (scenario == 4) {
    double *x = malloc((size_t)rows * n * sizeof(double)),
           *cov = calloc((size_t)n * n, sizeof(double)),
           *l = calloc((size_t)n * n, sizeof(double)),
           *sol = calloc((size_t)n, sizeof(double));
    float *xf = malloc((size_t)rows * n * sizeof(float)),
          *cf = malloc((size_t)n * n * sizeof(float));
    double weights[16], rhs[16];
    if (!x || !cov || !l || !sol || !xf || !cf) {
      fclose(f);
      free(x);
      free(cov);
      free(l);
      free(sol);
      free(xf);
      free(cf);
      return 0;
    }
    for (int i = 0; i < rows * n; ++i) {
      HftRandom r =
          hft_random(c->seed, (hft_u32)(i % n), 702, (hft_u64)(i / n));
      x[i] = hft_normal(r.x[0], r.x[1]);
      xf[i] = (float)x[i];
    }
    for (int e = 0; e < ne; ++e) {
      int engine = engines[e], ok = 1;
      if (engine == HFT_THREADS || engine == HFT_MPI || engine == HFT_HYBRID ||
          (engine == HFT_NVIDIA && scenario != 4)) {
        printf("SKIP %s: no dedicated S%d analytics implementation\n",
               hft_engine_names[engine], scenario);
        continue;
      }
      HftCapability cap;
      hft_probe(engine, &cap);
      if (!cap.available) {
        printf("SKIP %s: %s\n", hft_engine_names[engine], cap.reason);
        continue;
      }
#ifdef HFT_HAVE_OPENMP
      omp_set_num_threads(c->threads);
#endif

      double start = hft_now(), seconds = 0;
      float *library_normals = NULL;
      if (engine == HFT_OPENCL) {
        ok = hft_opencl_matrix(xf, rows, n, cf, 1, &seconds, error);
        for (int i = 0; i < n * n; ++i)
          cov[i] = cf[i];
      } else if (engine == HFT_NVIDIA) {
        library_normals = malloc(10000 * (size_t)n * sizeof(float));
        ok =
            library_normals && hft_nvidia_risk(xf, rows, n, cf, library_normals,
                                               10000, c->seed, &seconds, error);

        for (int i = 0; i < n * n; ++i)
          cov[i] = cf[i];
      } else {
        hft_covariance(x, rows, n, cov, engine != 0, engine == HFT_OPENMP);
        seconds = hft_now() - start;
      }
      if (!ok) {
        fprintf(stderr, "SKIP %s: %s\n", hft_engine_names[engine], error);
        free(library_normals);
        continue;
      }
      fprintf(f, "S4,%s,covariance,%.9g,%.9g,\n", hft_engine_names[engine],
              seconds, cov[0]);
      for (int i = 0; i < n; ++i) {
        cov[i * n + i] += .01;
        weights[i] = rhs[i] = 1.0 / n;
        sol[i] = 0;
      }
      int iterations = 0;
      double residual = 0, var = 0, cvar = 0;
      start = hft_now();
      ok = hft_cholesky(cov, n, l) &&
           hft_cg(cov, rhs, n, sol, 200, 1e-9, engine == HFT_OPENMP,
                  &iterations, &residual) &&
           (library_normals
                ? hft_risk_from_normals(cov, weights, library_normals, n, 10000,
                                        &var, &cvar)
                : hft_monte_carlo(cov, weights, n, 10000, c->seed, &var, &cvar,
                                  engine == HFT_OPENMP));
      free(library_normals);
      fprintf(f,
              "S4,%s,cpu_correlated_risk,%.9g,%.9g,VaR99=%.9g;CVaR99=%.9g;CG=%"
              "d;residual=%.9g;ok=%d\n",
              hft_engine_names[engine], hft_now() - start, cvar, var, cvar,
              iterations, residual, ok);
    }
    free(x);
    free(xf);
    free(cf);
    free(cov);
    free(l);
    free(sol);
  } else if (scenario == 5 || scenario == 6) {
    HftDataset d;
    if (!hft_dataset_make(c, c->train_samples, &d, error)) {
      fprintf(stderr, "%s\n", error);
      fclose(f);
      return 0;
    }
    for (int e = 0; e < ne; ++e) {
      HftModel m = c->model;
      double start = hft_now(), seconds = 0, loss = 0, accuracy = 0;
      int engine = engines[e], ok = 1;
      if (engine == HFT_THREADS || engine == HFT_MPI || engine == HFT_HYBRID ||
          (engine == HFT_NVIDIA && scenario != 4)) {
        printf("SKIP %s: no dedicated S%d analytics implementation\n",
               hft_engine_names[engine], scenario);
        continue;
      }
      HftCapability cap;
      hft_probe(engine, &cap);
      if (!cap.available) {
        printf("SKIP %s: %s\n", hft_engine_names[engine], cap.reason);
        continue;
      }
#ifdef HFT_HAVE_OPENMP
      omp_set_num_threads(c->threads);
#endif

      if (scenario == 5) {
        float *out = calloc((size_t)d.count * 3, sizeof(float));
        if (!out) {
          hft_dataset_free(&d);
          fclose(f);
          return 0;
        }
        if (engine == HFT_OPENCL)
          ok = hft_opencl_infer(&m, d.features, d.count, out, &seconds, error);
        else {
#ifdef HFT_HAVE_OPENMP
#pragma omp parallel for schedule(static) if (engine == HFT_OPENMP)
#endif
          for (int i = 0; i < d.count; ++i)
            hft_mlp_predict(&m, d.features + i * 32, out + i * 3);
          seconds = hft_now() - start;
        }
        if (out)
          accuracy = out[0];
        free(out);
      } else if (engine == HFT_OPENCL) {
        ok = hft_opencl_train(&m, d.features, d.labels, d.train_count,
                              c->train_epochs, &seconds, error);
      } else {
        ok = hft_mlp_train(&m, &d, c->train_epochs,
                           engine == HFT_OPENMP ? c->threads : 1, &loss,
                           &accuracy, NULL);
        seconds = hft_now() - start;
      }
      if (!ok) {
        fprintf(stderr, "SKIP %s: %s\n", hft_engine_names[engine], error);
        continue;
      }
      if (scenario == 6 && engine == HFT_OPENCL) {
        int correct = 0;
        loss = 0;
        for (int i = d.count * 3 / 4; i < d.count; ++i) {
          float prob[3];
          int best = 0;
          hft_mlp_predict(&m, d.features + i * 32, prob);
          for (int j = 1; j < 3; ++j)
            if (prob[j] > prob[best])
              best = j;
          correct += best == d.labels[i];
          loss -= log(fmax(1e-20, prob[d.labels[i]]));
        }
        accuracy = (double)correct / (d.count - d.count * 3 / 4);
        loss /= (d.count - d.count * 3 / 4);
      }
      fprintf(f, "S%d,%s,mlp,%.9g,%.9g,loss=%.9g\n", scenario,
              hft_engine_names[engine], seconds, accuracy, loss);
      if (scenario == 6) {
        snprintf(path, sizeof(path), "%s/model-%s.bin", folder,
                 hft_engine_names[engine]);
        hft_model_save(&m, path, error);
        start = hft_now();
        double reward = 0;
        if (engine == HFT_OPENCL)
          ok = hft_opencl_qtrain(&m, 8, 256, c->seed, &seconds, error);
        else {
          ok = hft_q_train(&m, c, 8, 256, engine == HFT_OPENMP ? c->threads : 1,
                           &reward, NULL);
          seconds = hft_now() - start;
        }
        fprintf(f, "S6,%s,qlearning,%.9g,%.9g,ok=%d\n",
                hft_engine_names[engine], seconds, reward, ok);
        snprintf(path, sizeof(path), "%s/model-%s.bin", folder,
                 hft_engine_names[engine]);
        hft_model_save(&m, path, error);
      }
    }
    hft_dataset_free(&d);
  } else if (scenario == 7) {
    double times[3];
    if (!hft_sync_benchmark(c->threads, 100000, times, error)) {
      fclose(f);
      return 0;
    }
    double omp_times[3];
    if (hft_openmp_sync(c->threads, 100000, omp_times)) {
      const char *methods[] = {"critical", "atomic", "reduction"};
      for (int i = 0; i < 3; ++i)
        fprintf(f, "S7,openmp,%s,%.9g,%d,100000 total iterations\n", methods[i],
                omp_times[i], c->threads);
    }
    const char *names[] = {"atomic_shared", "false_sharing", "padded_private"};
    for (int i = 0; i < 3; ++i)
      fprintf(f, "S7,threads,%s,%.9g,%d,100000 increments per thread\n",
              names[i], times[i], c->threads);
  }
  fclose(f);
  printf("Analytics: %s\n", folder);
  return 1;
}
int hft_bench_main(int argc, char **argv) {
  HftConfig c;
  char error[HFT_ERROR] = {0};
  hft_config_default(&c);
  int engines[64] = {HFT_NAIVE, HFT_SERIAL, HFT_THREADS, HFT_OPENMP},
      symbols[64] = {16}, threads[64] = {4}, batches[64] = {64}, ne = 4, ns = 1,
      nt = 1, nb = 1, scenario = 2, grid = 4;
  const char *folder = "results";
  const char *save = NULL;
  int status = 0;
  for (int a = 1; a < argc; ++a) {
    const char *k = argv[a];
    if (!strcmp(k, "--help")) {
      usage();
      return 0;
    }
    if (!strcmp(k, "--list")) {
      for (int i = 0; i < 8; ++i) {
        HftCapability cap;
        hft_probe(i, &cap);
        printf("%-14s %-11s %s\n", hft_engine_names[i],
               cap.available ? "available" : "unavailable", cap.reason);
      }
      return 0;
    }
    if (a + 1 >= argc) {
      usage();
      return 2;
    }
    const char *v = argv[++a];
    if (!strcmp(k, "--config")) {
      if (!hft_config_load(&c, v, error)) {
        fprintf(stderr, "%s\n", error);
        return 2;
      }
      symbols[0] = c.symbols;
      threads[0] = c.threads;
      batches[0] = c.batch;
      engines[0] = c.engine;
      ne = 1;
    } else if (!strcmp(k, "--engines"))
      ne = list(v, engines, 1);
    else if (!strcmp(k, "--symbols"))
      ns = list(v, symbols, 0);
    else if (!strcmp(k, "--threads"))
      nt = list(v, threads, 0);
    else if (!strcmp(k, "--batch"))
      nb = list(v, batches, 0);
    else if (!strcmp(k, "--events"))
      c.events = strtoull(v, NULL, 10);
    else if (!strcmp(k, "--seed"))
      c.seed = strtoull(v, NULL, 10);
    else if (!strcmp(k, "--reps"))
      c.repetitions = atoi(v);
    else if (!strcmp(k, "--warmup"))
      c.warmup = atoi(v);
    else if (!strcmp(k, "--strategy"))
      c.strategy = indexof(v, hft_strategy_names, 13);
    else if (!strcmp(k, "--scenario"))
      scenario = atoi(v + (*v == 'S' || *v == 's'));
    else if (!strcmp(k, "--mode")) {
      if (strcmp(v, "latency") && strcmp(v, "verify"))
        return 2;
      c.mode = !strcmp(v, "latency");
    } else if (!strcmp(k, "--network-ns"))
      c.network_ns = strtod(v, NULL);
    else if (!strcmp(k, "--amplification"))
      c.latency_amplification = strtod(v, NULL);
    else if (!strcmp(k, "--out"))
      folder = v;
    else if (!strcmp(k, "--save-config"))
      save = v;
    else if (!strcmp(k, "--epochs"))
      c.train_epochs = atoi(v);
    else if (!strcmp(k, "--grid"))
      grid = atoi(v);
    else if (!strcmp(k, "--samples"))
      c.train_samples = atoi(v);
    else if (!strcmp(k, "--model")) {
      if (!hft_model_load(&c.model, v, error)) {
        fprintf(stderr, "%s\n", error);
        return 2;
      }
    } else {
      fprintf(stderr, "Unknown argument %s\n", k);
      return 2;
    }
  }
  if (!ne || !ns || !nt || !nb || scenario < 1 || scenario > 8 || grid < 1 ||
      grid > 32) {
    usage();
    return 2;
  }
  c.symbols = symbols[0];
  c.threads = threads[0];
  c.batch = batches[0];
  if (!hft_config_validate(&c, error)) {
    fprintf(stderr, "%s\n", error);
    return 2;
  }
  if (save && !hft_config_save(&c, save, error)) {
    fprintf(stderr, "%s\n", error);
    return 2;
  }
  if (!hft_mkdir(folder)) {
    fprintf(stderr, "Cannot create %s\n", folder);
    return 2;
  }
  if (scenario >= 4 && scenario <= 7)
    return analytics(&c, scenario, engines, ne, folder) ? 0 : 1;
  if (scenario == 8) {
    fprintf(stderr, "S8 is distributed: run hft_mpi_worker with mpiexec; see "
                    "scripts/run_mpi.sh\n");
    return 2;
  }
  for (int si = 0; si < ns; ++si)
    for (int ti = 0; ti < nt; ++ti)
      for (int bi = 0; bi < nb; ++bi) {
        c.symbols = symbols[si];
        c.threads = threads[ti];
        c.batch = batches[bi];
        int sweeps = scenario == 3 ? grid * grid : 1;
        for (int sweep = 0; sweep < sweeps; ++sweep) {
          if (scenario == 3) {
            c.fast = 2 + (sweep % grid);
            c.slow = c.fast + 8 + (sweep / grid) * 2;
            c.strategy = HFT_SMA;
          }
          HftResult reference = {0};
          double naive_seconds = 0, serial_seconds = 0;
          HftConfig refcfg = c;
          refcfg.engine = HFT_SERIAL;
          refcfg.mode = 0;
          if (!hft_run(&refcfg, &reference, NULL, NULL, NULL)) {
            fprintf(stderr, "%s\n", reference.error);
            hft_result_free(&reference);
            return 1;
          }
          serial_seconds = reference.seconds;
          for (int ei = 0; ei < ne; ++ei) {
            c.engine = engines[ei];
            HftCapability cap;
            hft_probe(c.engine, &cap);
            if (!cap.available) {
              printf("SKIP %s: %s\n", hft_engine_names[c.engine], cap.reason);
              continue;
            }
            HftResult r = {0};
            double durations[100];
            int completed = 0;
            for (int rep = -c.warmup; rep < c.repetitions; ++rep) {
              hft_result_free(&r);
              if (!hft_run(&c, &r, NULL, NULL, NULL)) {
                fprintf(stderr, "%s: %s\n", hft_engine_names[c.engine],
                        r.error);
                status = 1;
                break;
              }
              if (rep >= 0)
                durations[completed++] = r.seconds;
            }
            if (completed) {
              qsort(durations, (size_t)completed, sizeof(double), doublecmp);
              double median = durations[completed / 2];
              if (c.engine == HFT_NAIVE)
                naive_seconds = median;
              r.verified = !c.mode && hft_result_equivalent(&reference, &r);
              if (!c.mode && !r.verified)
                status = 1;
              r.speedup_serial = serial_seconds / median;
              r.speedup_naive = naive_seconds > 0 ? naive_seconds / median : 0;
              r.efficiency = r.speedup_serial / (r.workers ? r.workers : 1);
              r.karp_flatt = r.workers > 1
                                 ? (1 / r.speedup_serial - 1.0 / r.workers) /
                                       (1 - 1.0 / r.workers)
                                 : 0;
              r.latency_cost = c.mode ? r.pnl - reference.pnl : 0;
              char prefix[2048];
              snprintf(prefix, sizeof(prefix), "%s/S%d-%s-n%d-t%d-b%d-sweep%d",
                       folder, scenario, hft_engine_names[c.engine], c.symbols,
                       c.threads, c.batch, sweep);
              if (!hft_result_export(&r, prefix, error)) {
                fprintf(stderr, "%s\n", error);
                status = 1;
              }
              char path[2100];
              snprintf(path, sizeof(path), "%s.svg", prefix);
              hft_svg_report(&r, path, error);
              snprintf(path, sizeof(path), "%s-timing.csv", prefix);
              FILE *f = fopen(path, "wb");
              if (f) {
                fprintf(f, "sorted_repetition,seconds\n");
                for (int i = 0; i < completed; ++i)
                  fprintf(f, "%d,%.17g\n", i, durations[i]);
                fclose(f);
              }
              printf("%-12s n=%d workers=%d median=%.6fs events/s=%.0f "
                     "p50=%llu p99=%llu ns pnl=%lld hash=%016llx %s\n",
                     hft_engine_names[c.engine], c.symbols, r.workers, median,
                     r.throughput,
                     (unsigned long long)hft_hist_percentile(&r.latency, .5),
                     (unsigned long long)hft_hist_percentile(&r.latency, .99),
                     (long long)r.pnl, (unsigned long long)r.hash,
                     c.mode ? "LATENCY MODE"
                            : (r.verified ? "MATCH" : "MISMATCH"));
            }
            hft_result_free(&r);
          }
          hft_result_free(&reference);
        }
      }
  return status;
}
#ifndef HFT_NO_MAIN
int main(int argc, char **argv) { return hft_bench_main(argc, argv); }

#endif
