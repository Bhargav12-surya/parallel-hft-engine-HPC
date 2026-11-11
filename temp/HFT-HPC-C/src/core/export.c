/* Versioned exports outside timed paths. JSON preserves integers as decimal
 * strings, CSV is tidy metric/value, SVG is a self-contained report. */
#include "cJSON.h"
#include "hft/api.h"
#include "kernels.inc"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
extern cJSON *hft_config_json(const HftConfig *);
extern int hft_config_from_json(HftConfig *, const cJSON *, char *);
static void u64(cJSON *j, const char *name, hft_u64 v) {
  char s[32];
  snprintf(s, sizeof(s), "%llu", (unsigned long long)v);
  cJSON_AddStringToObject(j, name, s);
}
static void i64(cJSON *j, const char *name, hft_i64 v) {
  char s[32];
  snprintf(s, sizeof(s), "%lld", (long long)v);
  cJSON_AddStringToObject(j, name, s);
}
#define METRICS(X)                                                             \
  X(seconds)                                                                   \
  X(throughput)                                                                \
  X(speedup_naive) X(speedup_serial) X(efficiency) X(karp_flatt)               \
      X(h2d_seconds) X(kernel_seconds) X(d2h_seconds) X(communication_seconds) \
          X(wait_seconds) X(batch_latency_ns) X(latency_overhead_ns) X(sharpe) \
              X(sortino) X(max_drawdown) X(return_fraction) X(win_rate)        \
                  X(profit_factor) X(slippage)
int hft_result_export(const HftResult *r, const char *prefix, char *error) {
  char path[2048];
  cJSON *j = cJSON_CreateObject();
  cJSON_AddNumberToObject(j, "schema_version", 2);
  cJSON_AddItemToObject(j, "config", hft_config_json(&r->config));
  cJSON_AddStringToObject(j, "hardware", r->hardware);
  cJSON_AddStringToObject(j, "timestamp", r->timestamp);
  cJSON_AddStringToObject(j, "device", r->device);
  cJSON_AddBoolToObject(j, "cancelled", r->cancelled);
  cJSON_AddNumberToObject(j, "verified", r->verified);
  cJSON_AddNumberToObject(j, "workers", r->workers);
#define WRITE(name) cJSON_AddNumberToObject(j, #name, r->name);
  METRICS(WRITE)
#undef WRITE
  u64(j, "events", r->events);
  u64(j, "orders", r->orders);
  u64(j, "fills", r->fills);
  u64(j, "rejected", r->rejected);
  u64(j, "hash", r->hash);
  u64(j, "market_hash", r->market_hash);
  i64(j, "pnl", r->pnl);
  i64(j, "fees", r->fees);
  i64(j, "gross", r->gross);
  i64(j, "latency_cost", r->latency_cost);
  cJSON_AddStringToObject(
      j, "latency_scope",
      "CPU pair-step, including market/book/features/risk/execution; device "
      "batch reported separately");
  u64(j, "latency_count", r->latency.count);
  u64(j, "p50_ns", hft_hist_percentile(&r->latency, .5));
  u64(j, "p95_ns", hft_hist_percentile(&r->latency, .95));
  u64(j, "p99_ns", hft_hist_percentile(&r->latency, .99));
  u64(j, "max_ns", r->latency.maximum);
  cJSON *points = cJSON_AddArrayToObject(j, "points");
  for (int i = 0; i < r->point_count; ++i) {
    cJSON *p = cJSON_CreateObject();
    u64(p, "event", r->points[i].event);
    i64(p, "pnl", r->points[i].pnl);
    i64(p, "price", r->points[i].price);
    cJSON_AddNumberToObject(p, "inventory", r->points[i].inventory);
    cJSON_AddItemToArray(points, p);
  }
  snprintf(path, sizeof(path), "%s.json", prefix);
  FILE *f = fopen(path, "wb");
  char *s = cJSON_Print(j);
  int ok = f && s && fwrite(s, 1, strlen(s), f) == strlen(s);
  if (f)
    ok = fclose(f) == 0 && ok;
  free(s);
  cJSON_Delete(j);
  if (!ok) {
    snprintf(error, HFT_ERROR, "Cannot write JSON export");
    return 0;
  }
  snprintf(path, sizeof(path), "%s.csv", prefix);
  f = fopen(path, "wb");
  if (!f) {
    strcpy(error, "Cannot write CSV export");
    return 0;
  }
  fprintf(f, "engine,strategy,symbols,events,seed,metric,value\n");
#define CSV(name)                                                              \
  fprintf(f, "%s,%s,%d,%llu,%llu,%s,%.17g\n",                                  \
          hft_engine_names[r->config.engine],                                  \
          hft_strategy_names[r->config.strategy], r->config.symbols,           \
          (unsigned long long)r->events, (unsigned long long)r->config.seed,   \
          #name, r->name);
  METRICS(CSV)
#undef CSV
  fprintf(f, "%s,%s,%d,%llu,%llu,pnl_ticks,%lld\n",
          hft_engine_names[r->config.engine],
          hft_strategy_names[r->config.strategy], r->config.symbols,
          (unsigned long long)r->events, (unsigned long long)r->config.seed,
          (long long)r->pnl);
  return fclose(f) == 0;
}
static hft_u64 readu(const cJSON *j, const char *key) {
  const cJSON *v = cJSON_GetObjectItemCaseSensitive(j, key);
  return cJSON_IsString(v) ? (hft_u64)strtoull(v->valuestring, NULL, 10) : 0;
}
int hft_result_import(HftResult *r, const char *path, char *error) {
  FILE *f = fopen(path, "rb");
  if (!f) {
    strcpy(error, "Cannot read result");
    return 0;
  }
  fseek(f, 0, SEEK_END);
  long size = ftell(f);
  rewind(f);
  if (size < 1 || size > 10000000) {
    fclose(f);
    strcpy(error, "Invalid result size");
    return 0;
  }
  char *data = calloc((size_t)size + 1, 1);
  if (!data) {
    fclose(f);
    return 0;
  }
  size_t got = fread(data, 1, (size_t)size, f);
  fclose(f);
  cJSON *j = got == (size_t)size ? cJSON_Parse(data) : NULL;
  free(data);
  if (!j) {
    strcpy(error, "Invalid JSON result");
    return 0;
  }
  memset(r, 0, sizeof(*r));
  hft_config_default(&r->config);
  if (!hft_config_from_json(&r->config, cJSON_GetObjectItem(j, "config"),
                            error)) {
    cJSON_Delete(j);
    return 0;
  }
#define READ(name)                                                             \
  {                                                                            \
    const cJSON *v = cJSON_GetObjectItem(j, #name);                            \
    r->name = cJSON_IsNumber(v) ? v->valuedouble : 0;                          \
  }
  METRICS(READ)
#undef READ
  r->events = readu(j, "events");
  r->orders = readu(j, "orders");
  r->fills = readu(j, "fills");
  r->hash = readu(j, "hash");
  r->market_hash = readu(j, "market_hash");
  r->pnl = (hft_i64)readu(j, "pnl");
  r->fees = (hft_i64)readu(j, "fees");
  r->gross = (hft_i64)readu(j, "gross");
  r->rejected = readu(j, "rejected");
  const cJSON *points = cJSON_GetObjectItem(j, "points");
  r->point_count = cJSON_GetArraySize(points);
  if (r->point_count > 2049)
    r->point_count = 2049;
  r->points = calloc(2049, sizeof(HftPoint));
  if (!r->points) {
    cJSON_Delete(j);
    return 0;
  }
  for (int i = 0; i < r->point_count; ++i) {
    const cJSON *p = cJSON_GetArrayItem(points, i);
    r->points[i].event = readu(p, "event");
    r->points[i].pnl = (hft_i64)readu(p, "pnl");
    r->points[i].price = (hft_i64)readu(p, "price");
  }
  cJSON_Delete(j);
  return 1;
}
int hft_svg_report(const HftResult *r, const char *path, char *error) {
  FILE *f = fopen(path, "wb");
  if (!f) {
    strcpy(error, "Cannot create report");
    return 0;
  }
  fprintf(
      f,
      "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1200\" height=\"720\" "
      "viewBox=\"0 0 1200 720\"><rect width=\"1200\" height=\"720\" "
      "fill=\"#0b1220\"/><g fill=\"#e8edf4\" font-family=\"sans-serif\"><text "
      "x=\"48\" y=\"64\" font-size=\"30\">HFT / HPC STUDIO — C</text><text "
      "x=\"48\" y=\"105\" font-size=\"18\">%s · %s · %d symbols · seed "
      "%llu</text><text x=\"48\" y=\"160\" font-size=\"22\">%.0f events/s | "
      "%.3f seconds | P&amp;L %lld ticks</text><text x=\"48\" y=\"200\" "
      "font-size=\"16\">p50 %llu ns · p99 %llu ns (CPU pair-step) · "
      "verification %s</text><text x=\"48\" y=\"660\" "
      "font-size=\"14\">Synthetic market. Not a live trading system. Returns "
      "depend on the model and parameters.</text></g>",
      hft_engine_names[r->config.engine],
      hft_strategy_names[r->config.strategy], r->config.symbols,
      (unsigned long long)r->config.seed, r->throughput, r->seconds,
      (long long)r->pnl,
      (unsigned long long)hft_hist_percentile(&r->latency, .5),
      (unsigned long long)hft_hist_percentile(&r->latency, .99),
      r->verified ? "passed" : "not checked");
  hft_i64 low = 0, high = 1;
  for (int i = 0; i < r->point_count; ++i) {
    if (r->points[i].pnl < low)
      low = r->points[i].pnl;
    if (r->points[i].pnl > high)
      high = r->points[i].pnl;
  }
  fprintf(f, "<path fill=\"none\" stroke=\"#46d9cb\" stroke-width=\"2\" d=\"");
  for (int i = 0; i < r->point_count; ++i)
    fprintf(f, "%c %.2f %.2f ", i ? 'L' : 'M',
            48 + 1100.0 * i / (r->point_count > 1 ? r->point_count - 1 : 1),
            580 - 300.0 * (r->points[i].pnl - low) / (high - low));
  fprintf(f, "\"/></svg>\n");
  return fclose(f) == 0;
}
/* Minimal PNG encoder: unfiltered RGB scanlines and valid uncompressed DEFLATE.
 * O(width*height), intentionally favors zero dependencies over compression. */
static unsigned crc(const unsigned char *p, size_t n) {
  unsigned c = ~0u;
  for (size_t i = 0; i < n; ++i) {
    c ^= p[i];
    for (int k = 0; k < 8; ++k)
      c = (c >> 1) ^ (0xedb88320u & -(c & 1));
  }
  return ~c;
}
static void be(unsigned char *p, unsigned n) {
  p[0] = (unsigned char)(n >> 24);
  p[1] = (unsigned char)(n >> 16);
  p[2] = (unsigned char)(n >> 8);
  p[3] = (unsigned char)n;
}
static int chunk(FILE *f, const char *type, const unsigned char *p, size_t n) {
  unsigned char *b = malloc(n + 12);
  if (!b)
    return 0;
  be(b, (unsigned)n);
  memcpy(b + 4, type, 4);
  if (n)
    memcpy(b + 8, p, n);
  be(b + n + 8, crc(b + 4, n + 4));
  int ok = fwrite(b, 1, n + 12, f) == n + 12;
  free(b);
  return ok;
}
int hft_write_png(const char *path, const unsigned char *rgb, int w, int h,
                  char *error) {
  if (w < 1 || h < 1 || w > 8192 || h > 8192)
    return 0;
  size_t n = (size_t)(w * 3 + 1) * h;
  unsigned char *raw = malloc(n), *z = malloc(n + n / 65535 * 5 + 16);
  if (!raw || !z) {
    free(raw);
    free(z);
    return 0;
  }
  for (int y = 0; y < h; ++y) {
    raw[(size_t)y * (w * 3 + 1)] = 0;
    memcpy(raw + (size_t)y * (w * 3 + 1) + 1, rgb + (size_t)y * w * 3,
           (size_t)w * 3);
  }
  size_t at = 2, pos = 0;
  unsigned a = 1, b = 0;
  z[0] = 0x78;
  z[1] = 1;
  while (pos < n) {
    size_t len = n - pos;
    if (len > 65535)
      len = 65535;
    z[at++] = (unsigned char)(pos + len == n);
    z[at++] = (unsigned char)len;
    z[at++] = (unsigned char)(len >> 8);
    unsigned inv = (unsigned)(~len);
    z[at++] = (unsigned char)inv;
    z[at++] = (unsigned char)(inv >> 8);
    memcpy(z + at, raw + pos, len);
    at += len;
    pos += len;
  }
  for (size_t i = 0; i < n; ++i) {
    a = (a + raw[i]) % 65521;
    b = (b + a) % 65521;
  }
  be(z + at, (b << 16) | a);
  at += 4;
  FILE *f = fopen(path, "wb");
  unsigned char ihdr[13] = {0};
  be(ihdr, (unsigned)w);
  be(ihdr + 4, (unsigned)h);
  ihdr[8] = 8;
  ihdr[9] = 2;
  int ok = f && fwrite("\211PNG\r\n\032\n", 1, 8, f) == 8 &&
           chunk(f, "IHDR", ihdr, 13) && chunk(f, "IDAT", z, at) &&
           chunk(f, "IEND", NULL, 0);
  if (f)
    ok = fclose(f) == 0 && ok;
  free(raw);
  free(z);
  if (!ok)
    strcpy(error, "PNG write failed");
  return ok;
}

int hft_source_export(const char *path, char *error) {
  FILE *f = fopen(path, "wb");
  if (!f) {
    strcpy(error, "Cannot export source");
    return 0;
  }
  int ok = fwrite(hft_kernel_source, 1, strlen(hft_kernel_source), f) ==
           strlen(hft_kernel_source);
  return fclose(f) == 0 && ok;
}
