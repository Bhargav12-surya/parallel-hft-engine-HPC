/* Purpose: strict versioned C JSON configuration. Units 1/4: reproducibility.
 * cJSON is C, with 64-bit seed/event integers represented as decimal strings.
 * Parsing and allocation occur outside timed execution. O(configuration
 * fields). */
#include "cJSON.h"
#include "hft/api.h"
#include "hft/numeric.h"
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
const char *const hft_engine_names[8] = {
    "serial_naive", "serial_opt", "threaded", "openmp",
    "mpi",          "hybrid",     "opencl",   "nvidia_libs"};
const char *const hft_strategy_names[13] = {
    "sma",  "ema",       "mean_reversion", "breakout", "rsi",
    "macd", "imbalance", "market_making",  "pairs",    "execution",
    "mlp",  "qlearning", "selector"};
const char *const hft_regime_names[7] = {
    "calm",        "bull",      "bear", "sideways", "high_volatility",
    "flash_crash", "news_shock"};
void hft_config_default(HftConfig *c) {
  memset(c, 0, sizeof(*c));
  c->seed = 42;
  c->events = 10000;
  c->symbols = 16;
  c->depth = 512;
  c->capacity = 512;
  c->engine = HFT_SERIAL;
  c->threads = hft_cores() < 8 ? hft_cores() : 8;
  if (c->threads < 1)
    c->threads = 1;
  c->batch = 64;
  c->warmup = 2;
  c->repetitions = 5;
  c->fast = 8;
  c->slow = 32;
  c->quantity = 2;
  c->max_position = 100;
  c->max_order = 20;
  c->max_orders_second = 100;
  c->collar = 40;
  c->chunk = 1;
  c->block_size = 64;
  c->initial_price = 10000;
  c->max_gross = 1000000000;
  c->max_loss = 1000000;
  c->initial_capital = 10000000;
  c->tick_size = .01;
  c->volatility = .005;
  c->hawkes_mu = 400;
  c->hawkes_alpha = 120;
  c->hawkes_beta = 800;
  c->jump_rate = .05;
  c->jump_std = .01;
  c->cancel_probability = .2;
  c->z_enter = 2;
  c->z_exit = .5;
  c->imbalance = .35;
  c->latency_amplification = 1;
  c->parent_qty = 100;
  c->parent_horizon = 2000;
  c->sample_every = 40;
  c->train_epochs = 10;
  c->train_samples = 2048;
  c->cpu_hourly = .15;
  c->gpu_hourly = .75;
  c->weights[0] = 45;
  c->weights[1] = 30;
  c->weights[2] = 10;
  c->weights[3] = 10;
  c->weights[4] = 5;
  hft_model_init(&c->model, c->seed);
}
#define CHECK(condition, message)                                              \
  do {                                                                         \
    if (!(condition)) {                                                        \
      snprintf(error, HFT_ERROR, "%s", message);                               \
      return 0;                                                                \
    }                                                                          \
  } while (0)
int hft_config_validate(const HftConfig *c, char *error) {
  int i, total = 0;
  CHECK(c->symbols >= 1 && c->symbols <= 4096, "symbols must be 1..4096");
  CHECK(c->events >= 1 && c->events <= 10000000, "events must be 1..10000000");
  CHECK(c->depth >= 64 && c->depth <= HFT_LEVELS && c->depth % 64 == 0,
        "depth must be a multiple of 64, 64..2048");
  CHECK(c->capacity >= 64 && c->capacity <= HFT_ORDERS,
        "capacity must be 64..1024");
  CHECK(c->initial_price > c->depth && c->initial_price + c->depth < 1000000,
        "initial_price must exceed depth and remain below 1000000");
  CHECK(c->fast >= 1 && c->fast < c->slow && c->slow <= HFT_WINDOW,
        "Require 1 <= fast < slow <= 128");
  CHECK(c->strategy >= 0 && c->strategy < 13 && c->engine >= 0 && c->engine < 8,
        "Unknown strategy or engine");
  CHECK(c->strategy != HFT_PAIRS || c->symbols % 2 == 0,
        "Pairs requires an even symbol count");
  CHECK(c->threads >= 1 && c->threads <= 256 && c->batch >= 1 &&
            c->batch <= 65536,
        "Invalid threads or batch");
  CHECK(c->warmup >= 0 && c->warmup <= 20 && c->repetitions >= 1 &&
            c->repetitions <= 100,
        "Invalid repetition counts");
  CHECK(c->quantity > 0 && c->quantity <= c->max_order && c->max_order <= 1000,
        "Invalid order quantity");
  CHECK(c->max_position >= 1 && c->max_position <= 1000000 &&
            c->max_gross >= c->symbols && c->max_gross <= 1000000000000LL,
        "Invalid position or exposure limit");
  CHECK(c->max_loss >= c->symbols && c->max_loss <= 1000000000000LL &&
            c->initial_capital > c->max_loss &&
            c->initial_capital <= 1000000000000000LL,
        "Initial capital must exceed loss budget");
  CHECK(c->max_orders_second >= 1 && c->max_orders_second <= 1000000 &&
            c->collar >= 1 && c->collar <= c->depth,
        "Invalid rate/collar");
  CHECK(c->regime >= 0 && c->regime < 7 && c->sample_every >= 1,
        "Invalid regime or sample rate");
  CHECK(c->schedule >= 0 && c->schedule <= 2 && c->chunk >= 1 &&
            c->block_size >= 1 && c->block_size <= 1024,
        "Invalid scheduling/block parameters");
  CHECK(c->streams >= 0 && c->streams <= 2 && c->pinned >= 0 &&
            c->pinned <= 1 && c->mode >= 0 && c->mode <= 1,
        "Invalid mode/streams/pinned");
  CHECK(c->topology >= 0 && c->topology <= 1 && c->partition >= 0 &&
            c->partition <= 1,
        "Invalid MPI topology/partition");
  CHECK(c->parent_qty >= 1 && c->parent_qty <= c->max_position &&
            c->parent_horizon >= 1,
        "Invalid parent order");
  CHECK(c->train_epochs >= 1 && c->train_epochs <= 1000 &&
            c->train_samples >= 128 && c->train_samples <= 65536,
        "Invalid training size");
  CHECK(isfinite(c->tick_size) && c->tick_size > 0 && isfinite(c->volatility) &&
            c->volatility >= 0 && c->volatility <= .2,
        "Invalid volatility/tick size");
  CHECK(isfinite(c->drift) && fabs(c->drift) <= .2 && isfinite(c->hawkes_mu) &&
            c->hawkes_mu >= 1 && c->hawkes_mu <= 1e6,
        "Invalid drift/Hawkes base rate");
  CHECK(isfinite(c->hawkes_alpha) && isfinite(c->hawkes_beta) &&
            c->hawkes_alpha >= 0 && c->hawkes_alpha < c->hawkes_beta &&
            c->hawkes_beta <= 1e6,
        "Hawkes requires 0 <= alpha < beta <= 1e6");
  CHECK(isfinite(c->jump_rate) && c->jump_rate >= 0 && c->jump_rate <= 10 &&
            isfinite(c->jump_mean) && fabs(c->jump_mean) <= .2 &&
            isfinite(c->jump_std) && c->jump_std >= 0 && c->jump_std <= .2,
        "Invalid jump parameters");
  CHECK(isfinite(c->cancel_probability) && c->cancel_probability >= 0 &&
            c->cancel_probability <= 1,
        "Invalid cancellation probability");
  CHECK(isfinite(c->z_enter) && c->z_enter >= .1 && c->z_enter <= 4 &&
            isfinite(c->z_exit) && c->z_exit >= 0 && c->z_exit < c->z_enter,
        "Invalid z-score bounds");
  CHECK(isfinite(c->imbalance) && c->imbalance >= 0 && c->imbalance <= 1,
        "Invalid imbalance threshold");
  CHECK(isfinite(c->latency_amplification) && c->latency_amplification >= 1 &&
            c->latency_amplification <= 10000 && isfinite(c->network_ns) &&
            c->network_ns >= 0 && c->network_ns <= 1e10 &&
            isfinite(c->network_jitter_ns) && c->network_jitter_ns >= 0 &&
            c->network_jitter_ns <= 1e10,
        "Invalid latency parameters");
  CHECK(isfinite(c->cpu_hourly) && c->cpu_hourly >= 0 &&
            isfinite(c->gpu_hourly) && c->gpu_hourly >= 0,
        "Invalid illustrative cloud prices");
  CHECK(c->fee >= -100 && c->fee <= 100,
        "fee must be -100..100 tick units per share");
  for (i = 0; i < 5; ++i) {
    CHECK(c->weights[i] >= 0 && c->weights[i] <= 1000, "Invalid agent weights");
    total += c->weights[i];
  }
  CHECK(total > 0, "At least one agent class is required");
  for (i = 0; i < HFT_WEIGHTS; ++i)
    CHECK(isfinite(c->model.weights[i]), "Nonfinite model weight");
  error[0] = 0;
  return 1;
}
typedef struct {
  const char *name;
  size_t offset;
  int type;
} Field;
#define F(name, type) {#name, offsetof(HftConfig, name), type}
static const Field fields[] = {F(seed, 2),
                               F(events, 2),
                               F(symbols, 0),
                               F(depth, 0),
                               F(capacity, 0),
                               F(strategy, 0),
                               F(engine, 0),
                               F(threads, 0),
                               F(batch, 0),
                               F(warmup, 0),
                               F(repetitions, 0),
                               F(fast, 0),
                               F(slow, 0),
                               F(quantity, 0),
                               F(max_position, 0),
                               F(max_order, 0),
                               F(max_orders_second, 0),
                               F(collar, 0),
                               F(regime, 0),
                               F(auto_regime, 0),
                               F(schedule, 0),
                               F(chunk, 0),
                               F(block_size, 0),
                               F(streams, 0),
                               F(pinned, 0),
                               F(topology, 0),
                               F(partition, 0),
                               F(mode, 0),
                               F(parent_qty, 0),
                               F(parent_horizon, 0),
                               F(sample_every, 0),
                               F(train_epochs, 0),
                               F(train_samples, 0),
                               F(initial_price, 3),
                               F(max_gross, 3),
                               F(max_loss, 3),
                               F(initial_capital, 3),
                               F(fee, 3),
                               F(tick_size, 1),
                               F(volatility, 1),
                               F(drift, 1),
                               F(hawkes_mu, 1),
                               F(hawkes_alpha, 1),
                               F(hawkes_beta, 1),
                               F(jump_rate, 1),
                               F(jump_mean, 1),
                               F(jump_std, 1),
                               F(cancel_probability, 1),
                               F(z_enter, 1),
                               F(z_exit, 1),
                               F(imbalance, 1),
                               F(latency_amplification, 1),
                               F(network_ns, 1),
                               F(network_jitter_ns, 1),
                               F(cpu_hourly, 1),
                               F(gpu_hourly, 1)};
cJSON *hft_config_json(const HftConfig *c) {
  cJSON *j = cJSON_CreateObject();
  size_t i;
  char buf[64];
  cJSON_AddNumberToObject(j, "schema_version", 2);
  for (i = 0; i < sizeof(fields) / sizeof(fields[0]); ++i) {
    const char *p = (const char *)c + fields[i].offset;
    switch (fields[i].type) {
    case 0:
      cJSON_AddNumberToObject(j, fields[i].name, *(const int *)p);
      break;
    case 1:
      cJSON_AddNumberToObject(j, fields[i].name, *(const double *)p);
      break;
    case 2:
      snprintf(buf, sizeof(buf), "%llu",
               (unsigned long long)*(const hft_u64 *)p);
      cJSON_AddStringToObject(j, fields[i].name, buf);
      break;
    default:
      snprintf(buf, sizeof(buf), "%lld", (long long)*(const hft_i64 *)p);
      cJSON_AddStringToObject(j, fields[i].name, buf);
      break;
    }
  }
  {
    cJSON *a = cJSON_AddArrayToObject(j, "agent_weights");
    for (i = 0; i < 5; ++i)
      cJSON_AddItemToArray(a, cJSON_CreateNumber(c->weights[i]));
  }
  return j;
}
int hft_config_from_json(HftConfig *c, const cJSON *j, char *error) {
  const cJSON *item;
  size_t i;
  if (!cJSON_IsObject(j)) {
    snprintf(error, HFT_ERROR, "Config must be a JSON object");
    return 0;
  }
  cJSON_ArrayForEach(item, j) {
    int found = 0;
    if (!strcmp(item->string, "schema_version")) {
      if (!cJSON_IsNumber(item) || item->valuedouble != 2) {
        snprintf(error, HFT_ERROR, "Expected schema_version 2");
        return 0;
      }
      continue;
    }
    if (!strcmp(item->string, "agent_weights")) {
      int k;
      if (cJSON_GetArraySize(item) != 5) {
        snprintf(error, HFT_ERROR, "Need five agent weights");
        return 0;
      }
      for (k = 0; k < 5; ++k) {
        const cJSON *v = cJSON_GetArrayItem(item, k);
        if (!cJSON_IsNumber(v) || v->valuedouble < 0 || v->valuedouble > 1000 ||
            floor(v->valuedouble) != v->valuedouble)
          return 0;
        c->weights[k] = v->valueint;
      }
      continue;
    }
    for (i = 0; i < sizeof(fields) / sizeof(fields[0]); ++i)
      if (!strcmp(item->string, fields[i].name)) {
        char *p = (char *)c + fields[i].offset;
        found = 1;
        if (fields[i].type < 2) {
          if (!cJSON_IsNumber(item) || !isfinite(item->valuedouble)) {
            snprintf(error, HFT_ERROR, "Numeric field required: %s",
                     item->string);
            return 0;
          }
          if (fields[i].type == 0) {
            if (item->valuedouble < INT_MIN || item->valuedouble > INT_MAX ||
                floor(item->valuedouble) != item->valuedouble) {
              snprintf(error, HFT_ERROR, "Integer out of range: %s",
                       item->string);
              return 0;
            }
            *(int *)p = item->valueint;
          } else
            *(double *)p = item->valuedouble;
        } else {
          char *end = NULL;
          const char *value = item->valuestring;
          if (!cJSON_IsString(item) || !value || !*value) {
            snprintf(error, HFT_ERROR,
                     "64-bit field %s must be a decimal string", item->string);
            return 0;
          }
          errno = 0;
          if (fields[i].type == 2) {
            if (*value == '-') {
              snprintf(error, HFT_ERROR, "Unsigned integer required");
              return 0;
            }
            *(hft_u64 *)p = (hft_u64)strtoull(value, &end, 10);
          } else
            *(hft_i64 *)p = (hft_i64)strtoll(value, &end, 10);
          if (errno || !end || *end) {
            snprintf(error, HFT_ERROR, "Invalid 64-bit integer: %s",
                     item->string);
            return 0;
          }
        }
        break;
      }
    if (!found) {
      snprintf(error, HFT_ERROR, "Unknown configuration key: %s", item->string);
      return 0;
    }
  }
  return hft_config_validate(c, error);
}
int hft_config_load(HftConfig *c, const char *path, char *error) {
  FILE *f = fopen(path, "rb");
  long size;
  char *text;
  cJSON *j;
  int ok;
  if (!f) {
    snprintf(error, HFT_ERROR, "Cannot open %s", path);
    return 0;
  }
  fseek(f, 0, SEEK_END);
  size = ftell(f);
  rewind(f);
  if (size < 0 || size > 2000000) {
    fclose(f);
    snprintf(error, HFT_ERROR, "Config file too large");
    return 0;
  }
  text = (char *)malloc((size_t)size + 1);
  if (!text) {
    fclose(f);
    return 0;
  }
  if (fread(text, 1, (size_t)size, f) != (size_t)size) {
    free(text);
    fclose(f);
    return 0;
  }
  text[size] = 0;
  fclose(f);
  j = cJSON_Parse(text);
  free(text);
  if (!j) {
    snprintf(error, HFT_ERROR, "Invalid JSON");
    return 0;
  }
  ok = hft_config_from_json(c, j, error);
  cJSON_Delete(j);
  return ok;
}
int hft_config_save(const HftConfig *c, const char *path, char *error) {
  cJSON *j = hft_config_json(c);
  char *text = cJSON_Print(j);
  FILE *f = fopen(path, "wb");
  int ok = 0;
  if (f && text) {
    ok = fwrite(text, 1, strlen(text), f) == strlen(text);
    fclose(f);
  }
  if (!ok)
    snprintf(error, HFT_ERROR, "Cannot write %s", path);
  free(text);
  cJSON_Delete(j);
  return ok;
}
