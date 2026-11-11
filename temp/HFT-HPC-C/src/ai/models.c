/* Trainable 32-64-32-3 MLP, Q market maker and persisted C model.
 * Unit 4: temporal validation, minibatch SGD, private gradients + fixed
 * reduction. O(samples*4291*epochs); no training data from future enters
 * features. */
#include "hft/api.h"
#include "hft/backprop.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef HFT_HAVE_OPENMP
#include <omp.h>
#endif
void hft_model_init(HftModel *m, hft_u64 seed) {
  memset(m, 0, sizeof(*m));
  m->version = 1;
  for (int i = 0; i < HFT_WEIGHTS; ++i) {
    HftRandom r = hft_random(seed, 0, 701, (hft_u64)i);
    m->weights[i] = (float)(hft_normal(r.x[0], r.x[1]) * .12);
  }
  for (int i = 2048; i < 2112; ++i)
    m->weights[i] = 0;
  for (int i = 4160; i < 4192; ++i)
    m->weights[i] = 0;
  for (int i = 4288; i < 4291; ++i)
    m->weights[i] = 0;
  for (int i = 0; i < 16; ++i)
    m->centroids[i] = (float)((i / 4) - 1.5);
}
void hft_mlp_predict(const HftModel *m, const float *x, float *p) {
  hft_infer_shared(m, x, p);
}
double hft_mlp_gradient(const HftModel *m, const float *x, int label,
                        float *g) {
  return hft_gradient_shared(m, x, label, g);
}
int hft_dataset_make(const HftConfig *input, int count, HftDataset *d,
                     char *error) {
  HftConfig c = *input;
  HftResult r;
  hft_i64 *prices;
  int horizon = 16, warm = 128;
  memset(d, 0, sizeof(*d));
  if (count < 128 || count > 1000000) {
    strcpy(error, "dataset samples must be 128..1000000");
    return 0;
  }
  c.symbols = 2;
  c.strategy = HFT_MAKER;
  c.engine = HFT_SERIAL;
  c.mode = 0;
  prices = calloc((size_t)count + horizon, sizeof(*prices));
  d->features = calloc((size_t)count * 32, sizeof(float));
  d->labels = calloc((size_t)count, sizeof(int));
  if (!prices || !d->features || !d->labels || !hft_initialize(&c, &r)) {
    strcpy(error, "Dataset allocation failed");
    free(prices);
    hft_dataset_free(d);
    return 0;
  }
  for (int i = -warm; i < count + horizon; ++i) {
    hft_step_group(r.states, 0, &c, 1, 0, 0);
    if (i >= 0) {
      prices[i] = hft_mid(hft_top(&r.states[0].book, &c, 0));
      if (i < count)
        memcpy(d->features + i * 32, r.states[0].features.features,
               32 * sizeof(float));
    }
  }
  for (int i = 0; i < count; ++i) {
    hft_i64 diff = prices[i + horizon] - prices[i];
    d->labels[i] = diff > 1 ? 2 : (diff < -1 ? 0 : 1);
  }
  d->count = count;
  d->train_count = count * 3 / 4 - horizon;
  free(prices);
  hft_result_free(&r);
  return 1;
}
void hft_dataset_free(HftDataset *d) {
  free(d->features);
  free(d->labels);
  memset(d, 0, sizeof(*d));
}
int hft_mlp_train(HftModel *m, const HftDataset *d, int epochs, int threads,
                  double *loss, double *accuracy, HftAtomic *cancel) {
  float *grads;
  int nt = 1;
#ifdef HFT_HAVE_OPENMP
  nt = threads > 0 ? threads : 1;
  if (nt > 64)
    nt = 64;
#else
  (void)threads;
#endif
  grads = calloc((size_t)nt * HFT_WEIGHTS, sizeof(float));
  if (!grads)
    return 0;
  for (int ep = 0; ep < epochs && (!cancel || !hft_load(cancel)); ++ep) {
    *loss = 0;
    for (int start = 0; start < d->train_count; start += 32) {
      int size = d->train_count - start;
      if (size > 32)
        size = 32;
      double losses[64] = {0};
      memset(grads, 0, (size_t)nt * HFT_WEIGHTS * sizeof(float));
#ifdef HFT_HAVE_OPENMP
#pragma omp parallel for num_threads(nt) schedule(static)
#endif
      for (int i = 0; i < size; ++i) {
        int tid = 0;
#ifdef HFT_HAVE_OPENMP
        tid = omp_get_thread_num();
#endif
        losses[tid] +=
            hft_mlp_gradient(m, d->features + (start + i) * 32,
                             d->labels[start + i], grads + tid * HFT_WEIGHTS);
      }
      for (int t = 0; t < nt; ++t)
        *loss += losses[t];
      for (int w = 0; w < HFT_WEIGHTS; ++w) {
        float sum = 0;
        for (int t = 0; t < nt; ++t)
          sum += grads[t * HFT_WEIGHTS + w];
        m->weights[w] -= .005f * (sum / size + .0001f * m->weights[w]);
      }
    }
    *loss /= d->train_count;
  }
  int correct = 0, begin = d->count * 3 / 4;
  for (int i = begin; i < d->count; ++i) {
    float p[3];
    int best = 0;
    hft_mlp_predict(m, d->features + i * 32, p);
    for (int j = 1; j < 3; ++j)
      if (p[j] > p[best])
        best = j;
    correct += best == d->labels[i];
  }
  *accuracy = (double)correct / (d->count - begin);
  m->trained = 1;
  float *x = malloc((size_t)d->train_count * 4 * sizeof(float));
  int *labels = malloc((size_t)d->train_count * sizeof(int));
  if (x && labels) {
    for (int i = 0; i < d->train_count; ++i)
      for (int j = 0; j < 4; ++j)
        x[i * 4 + j] = d->features[i * 32 + 5 + j];
    hft_kmeans(x, d->train_count, 4, 4, 20, m->centroids, labels, nt > 1);
  }
  free(x);
  free(labels);
  free(grads);
  return 1;
}
int hft_q_train(HftModel *model, const HftConfig *cfg, int episodes, int steps,
                int threads, double *reward, HftAtomic *cancel) {
  float *tables = calloc((size_t)episodes * HFT_QSTATES * 9, sizeof(float));
  double *rewards = calloc((size_t)episodes, sizeof(double));
  if (!tables || !rewards) {
    free(tables);
    free(rewards);
    return 0;
  }
#ifdef HFT_HAVE_OPENMP
#pragma omp parallel for num_threads(threads) schedule(static)
#endif
  for (int ep = 0; ep < episodes; ++ep) {
    HftConfig c = *cfg;
    HftResult r;
    float *q = tables + ep * HFT_QSTATES * 9;
    c.symbols = 2;
    c.seed += (hft_u64)ep * 7919;
    c.engine = HFT_SERIAL;
    c.strategy = HFT_QLEARNING;
    c.mode = 0;
    memcpy(q, model->q, sizeof(model->q));
    if (!hft_initialize(&c, &r))
      continue;
    for (int i = 0; i < steps && (!cancel || !hft_load(cancel)); ++i) {
      HftSymbol *s = r.states;
      HftTop t = hft_top(&s->book, &c, 0);
      int state = hft_q_state(s, t, &c), act = 0;
      HftRandom rnd = hft_random(c.seed, 0, 811, (hft_u64)i);
      for (int a = 1; a < 9; ++a)
        if (q[state * 9 + a] > q[state * 9 + act])
          act = a;
      if (hft_uniform(rnd.x[0]) < .2)
        act = (int)(rnd.x[1] % 9);
      hft_i64 old = s->position.cash + (hft_i64)s->position.qty * hft_mid(t);
      memcpy(c.model.q, q, sizeof(c.model.q));
      for (int a = 0; a < 9; ++a)
        c.model.q[state * 9 + a] = a == act ? 1e20f : 0;
      hft_step_group(r.states, 0, &c, 1, 0, 0);
      t = hft_top(&s->book, &c, 0);
      int next = hft_q_state(s, t, &c);
      float best = q[next * 9];
      for (int a = 1; a < 9; ++a)
        if (q[next * 9 + a] > best)
          best = q[next * 9 + a];
      double rew = (double)(s->position.cash +
                            (hft_i64)s->position.qty * hft_mid(t) - old) -
                   .001 * s->position.qty * s->position.qty;
      q[state * 9 + act] +=
          .05f * ((float)rew + .95f * best - q[state * 9 + act]);
      rewards[ep] += rew;
    }
    hft_result_free(&r);
  }
  *reward = 0;
  for (int j = 0; j < HFT_QSTATES * 9; ++j) {
    double sum = 0;
    for (int e = 0; e < episodes; ++e)
      sum += tables[e * HFT_QSTATES * 9 + j];
    model->q[j] = (float)(sum / episodes);
  }
  for (int e = 0; e < episodes; ++e)
    *reward += rewards[e] / episodes;
  model->q_trained = 1;
  free(tables);
  free(rewards);
  return 1;
}
int hft_model_save(const HftModel *m, const char *path, char *error) {
  FILE *f = fopen(path, "wb");
  if (!f) {
    snprintf(error, HFT_ERROR, "Cannot write %s", path);
    return 0;
  }
  int ok = fwrite("HFTCML01", 1, 8, f) == 8 && fwrite(m, sizeof(*m), 1, f) == 1;
  ok = fclose(f) == 0 && ok;
  if (!ok)
    strcpy(error, "Model write failed");
  return ok;
}
int hft_model_load(HftModel *m, const char *path, char *error) {
  char magic[8];
  HftModel temp;
  FILE *f = fopen(path, "rb");
  if (!f) {
    snprintf(error, HFT_ERROR, "Cannot read %s", path);
    return 0;
  }
  int ok = fread(magic, 1, 8, f) == 8 && memcmp(magic, "HFTCML01", 8) == 0 &&
           fread(&temp, sizeof(temp), 1, f) == 1 && fgetc(f) == EOF &&
           temp.version == 1;
  fclose(f);
  for (int i = 0; ok && i < HFT_WEIGHTS; ++i)
    ok = isfinite(temp.weights[i]);
  for (int i = 0; ok && i < HFT_QSTATES * 9; ++i)
    ok = isfinite(temp.q[i]);
  if (ok)
    *m = temp;
  else
    strcpy(error, "Invalid model or incompatible binary version (little-endian "
                  "IEEE float required)");
  return ok;
}
