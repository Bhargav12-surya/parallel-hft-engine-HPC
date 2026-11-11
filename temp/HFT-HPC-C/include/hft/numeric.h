/* Purpose: common Philox and fixed-order math. Units 1/3: deterministic
 * decisions. Complexity: ten RNG rounds; bounded-domain polynomial arithmetic.
 */
#ifndef HFT_NUMERIC_H
#define HFT_NUMERIC_H
#include "types.h"
#ifndef __OPENCL_VERSION__
#include <math.h>
#endif
#define HFT_PI 3.14159265358979323846264338327950288
typedef struct {
  hft_u32 x[4];
} HftRandom;
static inline hft_i64 hft_abs64(hft_i64 x) { return x < 0 ? -x : x; }
static inline double hft_abs(double x) { return x < 0 ? -x : x; }
static inline double hft_min(double a, double b) { return a < b ? a : b; }
static inline double hft_max(double a, double b) { return a > b ? a : b; }
static inline hft_i64 hft_clamp64(hft_i64 x, hft_i64 lo, hft_i64 hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}
static inline double hft_sqrt(double x) {
  double scale = 1, y;
  int i;
  if (x <= 0)
    return 0;
  while (x > 4) {
    x *= .25;
    scale *= 2;
  }
  while (x < 1) {
    x *= 4;
    scale *= .5;
  }
  y = (x + 1) * .5;
  for (i = 0; i < 8; ++i)
    y = .5 * (y + x / y);
  return y * scale;
}
static inline double hft_log(double x) {
  int e = 0, k;
  double z, z2, t, s;
  if (x <= 0)
    return -1.7976931348623157e308;
  while (x > 1.4142135623730951) {
    x *= .5;
    ++e;
  }
  while (x < .7071067811865475) {
    x *= 2;
    --e;
  }
  z = (x - 1) / (x + 1);
  z2 = z * z;
  t = s = z;
  for (k = 3; k <= 31; k += 2) {
    t *= z2;
    s += t / k;
  }
  return 2 * s + e * .69314718055994530941723212145817657;
}
static inline double hft_exp(double x) {
  int k, i;
  double r, t = 1, s = 1;
  if (x < -700)
    return 0;
  if (x > 700)
    x = 700;
  k = (int)(x / .6931471805599453094 + (x >= 0 ? .5 : -.5));
  r = (x - k * .6931471803691238) - k * 1.9082149292705877e-10;
  for (i = 1; i <= 18; ++i) {
    t *= r / i;
    s += t;
  }
  if (k >= 0)
    for (i = 0; i < k; ++i)
      s *= 2;
  else
    for (i = 0; i > k; --i)
      s *= .5;
  return s;
}
static inline double hft_cos(double x) {
  int q = (int)(x / (HFT_PI * .5) + (x >= 0 ? .5 : -.5)), i;
  double r = (x - q * 1.5707963267341256) - q * 6.077100506506192e-11,
         r2 = r * r, st = r, ct = 1, s = r, c = 1;
  for (i = 1; i <= 10; ++i) {
    st *= -r2 / ((2 * i) * (2 * i + 1));
    ct *= -r2 / ((2 * i - 1) * (2 * i));
    s += st;
    c += ct;
  }
  switch ((q % 4 + 4) % 4) {
  case 0:
    return c;
  case 1:
    return -s;
  case 2:
    return -c;
  default:
    return s;
  }
}
static inline HftRandom hft_philox(HftRandom r, hft_u32 k0, hft_u32 k1) {
  int i;
  for (i = 0; i < 10; ++i) {
    hft_u64 a = (hft_u64)0xD2511F53U * r.x[0],
            b = (hft_u64)0xCD9E8D57U * r.x[2];
    HftRandom next = {{(hft_u32)(b >> 32) ^ r.x[1] ^ k0, (hft_u32)b,
                       (hft_u32)(a >> 32) ^ r.x[3] ^ k1, (hft_u32)a}};
    r = next;
    k0 += 0x9E3779B9U;
    k1 += 0xBB67AE85U;
  }
  return r;
}
static inline HftRandom hft_random(hft_u64 seed, hft_u32 sym, hft_u32 stream,
                                   hft_u64 counter) {
  HftRandom r = {{(hft_u32)counter, (hft_u32)(counter >> 32), sym, stream}};
  return hft_philox(r, (hft_u32)seed, (hft_u32)(seed >> 32));
}
static inline double hft_uniform(hft_u32 x) {
  return ((double)x + .5) * 0x1p-32;
}
static inline double hft_normal(hft_u32 a, hft_u32 b) {
  return hft_sqrt(-2 * hft_log(hft_uniform(a))) *
         hft_cos(2 * HFT_PI * hft_uniform(b));
}
static inline int hft_poisson(hft_u32 a, hft_u32 b, double lambda) {
  int n = 0;
  double p, sum, u;
  if (lambda <= 0)
    return 0;
  if (lambda >= 30) {
    n = (int)(lambda + hft_sqrt(lambda) * hft_normal(a, b) + .5);
    return n < 0 ? 0 : n;
  }
  p = sum = hft_exp(-lambda);
  u = hft_uniform(a);
  while (u > sum && n < 512) {
    ++n;
    p *= lambda / n;
    sum += p;
  }
  return n;
}
static inline hft_u64 hft_hash_word(hft_u64 h, hft_u64 x, int bytes) {
  int i;
  for (i = 0; i < bytes; ++i) {
    h ^= x & 255;
    h *= 1099511628211UL;
    x >>= 8;
  }
  return h;
}
static inline hft_u64 hft_hash_order(hft_u64 h, HftOrder o) {
  h = hft_hash_word(h, 1, 1);
  h = hft_hash_word(h, o.id, 4);
  h = hft_hash_word(h, o.symbol, 4);
  h = hft_hash_word(h, (hft_u64)o.price, 8);
  h = hft_hash_word(h, o.ts, 8);
  h = hft_hash_word(h, (hft_u64)o.qty, 4);
  h = hft_hash_word(h, (hft_u64)o.side, 1);
  return hft_hash_word(h, (hft_u64)o.ioc, 1);
}
static inline hft_u64 hft_hash_fill(hft_u64 h, HftFill f) {
  h = hft_hash_word(h, 2, 1);
  h = hft_hash_word(h, f.id, 4);
  h = hft_hash_word(h, f.symbol, 4);
  h = hft_hash_word(h, (hft_u64)f.price, 8);
  h = hft_hash_word(h, f.ts, 8);
  h = hft_hash_word(h, (hft_u64)f.qty, 4);
  h = hft_hash_word(h, (hft_u64)f.side, 1);
  return hft_hash_word(h, (hft_u64)f.maker, 1);
}
#endif
