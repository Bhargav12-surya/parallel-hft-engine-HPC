#ifndef HFT_BACKPROP_H
#define HFT_BACKPROP_H
#include "simulation.h"
static inline double hft_gradient_shared(HFT_CONST HftModel *m,
                                         HFT_GLOBAL const float *x, int label,
                                         HFT_GLOBAL float *g) {
  float a[64], b[32], p[3], d1[64] = {0}, d2[32] = {0};
  int i, j;
  for (i = 0; i < 64; ++i) {
    float v = m->weights[2048 + i];
    for (j = 0; j < 32; ++j)
      v += x[j] * m->weights[i * 32 + j];
    a[i] = (v > 0 ? v : 0);
  }
  for (i = 0; i < 32; ++i) {
    float v = m->weights[4160 + i];
    for (j = 0; j < 64; ++j)
      v += a[j] * m->weights[2112 + i * 64 + j];
    b[i] = (v > 0 ? v : 0);
  }
  hft_infer_shared(m, x, p);
  double loss = -hft_log(hft_max(1e-20, p[label]));
  p[label] -= 1;
  for (i = 0; i < 3; ++i) {
    g[4288 + i] += p[i];
    for (j = 0; j < 32; ++j) {
      g[4192 + i * 32 + j] += p[i] * b[j];
      d2[j] += p[i] * m->weights[4192 + i * 32 + j];
    }
  }
  for (i = 0; i < 32; ++i) {
    d2[i] *= b[i] > 0;
    g[4160 + i] += d2[i];
    for (j = 0; j < 64; ++j) {
      g[2112 + i * 64 + j] += d2[i] * a[j];
      d1[j] += d2[i] * m->weights[2112 + i * 64 + j];
    }
  }
  for (i = 0; i < 64; ++i) {
    d1[i] *= a[i] > 0;
    g[2048 + i] += d1[i];
    for (j = 0; j < 32; ++j)
      g[i * 32 + j] += d1[i] * x[j];
  }
  return loss;
}

#endif
