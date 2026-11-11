/* Dense risk analytics in C. Unit 2: row ownership and reductions. Covariance
 * O(T*N^2), Cholesky O(N^3), CG O(k*N^2), Lloyd O(k*n*d*iterations). */
#include "hft/api.h"
#include "hft/numeric.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#ifdef HFT_HAVE_OPENMP
#include <omp.h>
#endif
int hft_covariance(const double *x, int rows, int cols, double *out, int tiled,
                   int parallel) {
  int i, j, k;
  double *mean;
  if (rows < 2 || cols < 1)
    return 0;
  mean = calloc((size_t)cols, sizeof(double));
  if (!mean)
    return 0;
  for (i = 0; i < cols; ++i)
    for (k = 0; k < rows; ++k)
      mean[i] += x[k * cols + i] / rows;
#ifdef HFT_HAVE_OPENMP
#pragma omp parallel for private(j, k) if (parallel) schedule(static)
#endif
  for (i = 0; i < cols; ++i)
    for (j = 0; j < cols; ++j) {
      double v = 0;
      if (tiled) {
        int b;
        for (b = 0; b < rows; b += 32)
          for (k = b; k < rows && k < b + 32; ++k)
            v += (x[k * cols + i] - mean[i]) * (x[k * cols + j] - mean[j]);
      } else
        for (k = 0; k < rows; ++k)
          v += (x[k * cols + i] - mean[i]) * (x[k * cols + j] - mean[j]);
      out[i * cols + j] = v / (rows - 1);
    }
  free(mean);
  return 1;
}
int hft_cholesky(const double *a, int n, double *l) {
  int i, j, k;
  memset(l, 0, (size_t)n * n * sizeof(double));
  for (i = 0; i < n; ++i)
    for (j = 0; j <= i; ++j) {
      double v = a[i * n + j];
      for (k = 0; k < j; ++k)
        v -= l[i * n + k] * l[j * n + k];
      if (i == j) {
        if (v <= 0 || !isfinite(v))
          return 0;
        l[i * n + j] = sqrt(v);
      } else
        l[i * n + j] = v / l[j * n + j];
    }
  return 1;
}
static double dot(const double *a, const double *b, int n) {
  double v = 0;
  int i;
  for (i = 0; i < n; ++i)
    v += a[i] * b[i];
  return v;
}
static void matvec(const double *a, const double *x, double *y, int n,
                   int parallel) {
  int i;
#ifdef HFT_HAVE_OPENMP
#pragma omp parallel for if (parallel) schedule(static)
#endif
  for (i = 0; i < n; ++i) {
    int j;
    double v = 0;
    for (j = 0; j < n; ++j)
      v += a[i * n + j] * x[j];
    y[i] = v;
  }
}
int hft_cg(const double *a, const double *b, int n, double *x, int maxiter,
           double tolerance, int parallel, int *iterations, double *residual) {
  double *r, *p, *ap, rr;
  int k, i, ok = 0;
  if (n < 1)
    return 0;
  r = calloc((size_t)n * 3, sizeof(double));
  if (!r)
    return 0;
  p = r + n;
  ap = p + n;
  matvec(a, x, ap, n, parallel);
  for (i = 0; i < n; ++i)
    p[i] = r[i] = b[i] - ap[i];
  rr = dot(r, r, n);
  for (k = 0; k < maxiter && sqrt(rr) > tolerance; ++k) {
    double pap, alpha, newrr;
    matvec(a, p, ap, n, parallel);
    pap = dot(p, ap, n);
    if (pap <= 0)
      break;
    alpha = rr / pap;
    for (i = 0; i < n; ++i) {
      x[i] += alpha * p[i];
      r[i] -= alpha * ap[i];
    }
    newrr = dot(r, r, n);
    for (i = 0; i < n; ++i)
      p[i] = r[i] + newrr / rr * p[i];
    rr = newrr;
  }
  *iterations = k;
  *residual = sqrt(rr);
  ok = *residual <= tolerance;
  free(r);
  return ok;
}
int hft_kmeans(const float *x, int rows, int dim, int clusters, int iterations,
               float *centers, int *labels, int parallel) {
  int k, i, iter;
  double *sum;
  int *count;
  if (rows < clusters || dim < 1 || clusters < 1)
    return 0;
  sum = calloc((size_t)clusters * dim, sizeof(double));
  count = calloc((size_t)clusters, sizeof(int));
  if (!sum || !count) {
    free(sum);
    free(count);
    return 0;
  }
  for (k = 0; k < clusters; ++k)
    memcpy(centers + k * dim, x + (k * rows / clusters) * dim,
           (size_t)dim * sizeof(float));
  for (iter = 0; iter < iterations; ++iter) {
#ifdef HFT_HAVE_OPENMP
#pragma omp parallel for if (parallel) schedule(static)
#endif
    for (i = 0; i < rows; ++i) {
      int c, d, best = 0;
      double closest = 1e300;
      for (c = 0; c < clusters; ++c) {
        double v = 0;
        for (d = 0; d < dim; ++d) {
          double diff = x[i * dim + d] - centers[c * dim + d];
          v += diff * diff;
        }
        if (v < closest) {
          closest = v;
          best = c;
        }
      }
      labels[i] = best;
    }
    memset(sum, 0, (size_t)clusters * dim * sizeof(double));
    memset(count, 0, (size_t)clusters * sizeof(int));
    for (i = 0; i < rows; ++i) {
      int d;
      ++count[labels[i]];
      for (d = 0; d < dim; ++d)
        sum[labels[i] * dim + d] += x[i * dim + d];
    }
    for (k = 0; k < clusters; ++k)
      if (count[k])
        for (i = 0; i < dim; ++i)
          centers[k * dim + i] = (float)(sum[k * dim + i] / count[k]);
  }
  free(sum);
  free(count);
  return 1;
}
static int compare_double(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}
int hft_monte_carlo(const double *cov, const double *weights, int n, int paths,
                    hft_u64 seed, double *var, double *cvar, int parallel) {
  double *l, *loss;
  int p;
  if (n < 1 || n > 256 || paths < 100)
    return 0;
  l = calloc((size_t)n * n, sizeof(double));
  loss = calloc((size_t)paths, sizeof(double));
  if (!l || !loss || !hft_cholesky(cov, n, l)) {
    free(l);
    free(loss);
    return 0;
  }
#ifdef HFT_HAVE_OPENMP
#pragma omp parallel for if (parallel) schedule(static)
#endif
  for (p = 0; p < paths; ++p) {
    double z[256], v = 0;
    int i, j;
    for (i = 0; i < n; ++i) {
      HftRandom rnd = hft_random(seed, (hft_u32)i, 901, (hft_u64)p);
      z[i] = hft_normal(rnd.x[0], rnd.x[1]);
    }
    for (i = 0; i < n; ++i) {
      double change = 0;
      for (j = 0; j <= i; ++j)
        change += l[i * n + j] * z[j];
      v -= weights[i] * change;
    }
    loss[p] = v;
  }
  qsort(loss, (size_t)paths, sizeof(double), compare_double);
  p = (int)(.99 * (paths - 1));
  *var = loss[p];
  *cvar = 0;
  for (int i = p; i < paths; ++i)
    *cvar += loss[i] / (paths - p);
  free(l);
  free(loss);
  return 1;
}
/* Correct synchronized counters versus padded private accumulation; false
 * sharing measures independent volatile slots, never undefined data races. */
typedef struct {
  HftAtomic *shared;
  volatile hft_u64 *slot;
  hft_u64 n;
  int mode;
} SyncJob;
static void *sync_work(void *arg) {
  SyncJob *j = arg;
  for (hft_u64 i = 0; i < j->n; ++i) {
    if (j->mode == 0)
      hft_add(j->shared, 1);
    else
      ++*j->slot;
  }
  return NULL;
}
int hft_sync_benchmark(int threads, hft_u64 n, double *times, char *error) {
  HftThread th[64];
  SyncJob jobs[64];
  HftAtomic shared = {0};
  volatile hft_u64 slots[64 * 16] = {0};
  if (threads < 1 || threads > 64) {
    strcpy(error, "threads must be 1..64");
    return 0;
  }
  for (int m = 0; m < 3; ++m) {
    double start = hft_now();
    for (int i = 0; i < threads; ++i) {
      jobs[i] = (SyncJob){&shared, &slots[i * (m == 2 ? 16 : 1)], n, m};
      if (!hft_thread_start(&th[i], sync_work, &jobs[i]))
        return 0;
    }
    for (int i = 0; i < threads; ++i)
      hft_thread_join(th[i]);
    times[m] = hft_now() - start;
  }
  return 1;
}
int hft_openmp_sync(int threads, hft_u64 iterations, double *times) {
#ifdef HFT_HAVE_OPENMP
  volatile hft_u64 value = 0;
  double start = hft_now();
#pragma omp parallel for num_threads(threads) schedule(static)
  for (hft_u64 i = 0; i < iterations; ++i) {
#pragma omp critical(hft_counter)
    value += 1;
  }
  times[0] = hft_now() - start;
  if (value != iterations)
    return 0;
  value = 0;
  start = hft_now();
#pragma omp parallel for num_threads(threads) schedule(static)
  for (hft_u64 i = 0; i < iterations; ++i) {
#pragma omp atomic update
    value += 1;
  }
  times[1] = hft_now() - start;
  if (value != iterations)
    return 0;
  hft_u64 sum = 0;
  start = hft_now();
#pragma omp parallel for num_threads(threads) reduction(+ : sum)               \
    schedule(static)
  for (hft_u64 i = 0; i < iterations; ++i)
    sum += (i % 2) + 1;
  times[2] = hft_now() - start;
  return sum == iterations + iterations / 2;
#else
  (void)threads;
  (void)iterations;
  (void)times;
  return 0;
#endif
}
/* Correlate externally generated normals (e.g. cuRAND) with CPU Cholesky;
 * the external generator is not discarded or relabeled as the CPU RNG. */
int hft_risk_from_normals(const double *covariance, const double *weights,
                          const float *normals, int assets, int paths,
                          double *var, double *cvar) {
  if (assets < 1 || assets > 256 || paths < 100)
    return 0;
  double *l = calloc((size_t)assets * assets, sizeof(double));
  double *losses = calloc((size_t)paths, sizeof(double));
  if (!l || !losses || !hft_cholesky(covariance, assets, l)) {
    free(l);
    free(losses);
    return 0;
  }
  for (int p = 0; p < paths; ++p) {
    double loss = 0;
    for (int i = 0; i < assets; ++i) {
      double move = 0;
      for (int j = 0; j <= i; ++j)
        move += l[i * assets + j] * normals[p * assets + j];
      loss -= weights[i] * move;
    }
    losses[p] = loss;
  }
  qsort(losses, (size_t)paths, sizeof(double), compare_double);
  int begin = (int)(.99 * (paths - 1));
  *var = losses[begin];
  *cvar = 0;
  for (int p = begin; p < paths; ++p)
    *cvar += losses[p] / (paths - begin);
  free(l);
  free(losses);
  return 1;
}
