/* Purpose: actual latency, accounting and statistical summaries. Unit 1.
 * Histogram recording O(1) (bounded bit scan); merge O(bins); metrics
 * O(samples). Tails describe timed pair steps. GPU batch latency is reported
 * separately. */
#include "hft/api.h"
#include "hft/simulation.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
void hft_hist_record(HftHistogram *h, hft_u64 ns) {
  int exponent = 0, index;
  hft_u64 x = ns;
  if (ns > 10000000000UL)
    ns = 10000000000UL;
  x = ns;
  if (ns < 128)
    index = (int)ns;
  else {
    while (x >> 1) {
      x >>= 1;
      ++exponent;
    }
    index = (exponent - 6) * 128 + (int)((ns >> (exponent - 7)) - 128);
  }
  ++h->counts[index];
  ++h->count;
  if (ns > h->maximum)
    h->maximum = ns;
  h->sum += (double)ns;
  h->squares += (double)ns * (double)ns;
}
hft_u64 hft_hist_percentile(const HftHistogram *h, double quantile) {
  hft_u64 target = (hft_u64)ceil(quantile * (double)h->count), n = 0;
  int i;
  if (!h->count)
    return 0;
  if (target < 1)
    target = 1;
  for (i = 0; i < HFT_LAT_BINS; ++i) {
    n += h->counts[i];
    if (n >= target) {
      hft_u64 upper;
      if (i < 128)
        upper = (hft_u64)i;
      else
        upper = ((128 + (hft_u64)(i % 128) + 1) << (i / 128 - 1)) - 1;
      return upper < h->maximum ? upper : h->maximum;
    }
  }
  return h->maximum;
}
void hft_hist_merge(HftHistogram *a, const HftHistogram *b) {
  int i;
  for (i = 0; i < HFT_LAT_BINS; ++i)
    a->counts[i] += b->counts[i];
  a->count += b->count;
  a->sum += b->sum;
  a->squares += b->squares;
  if (b->maximum > a->maximum)
    a->maximum = b->maximum;
}
void hft_result_finalize(HftResult *r) {
  int i;
  double sum = 0, sq = 0, down = 0, positive = 0, negative = 0,
         peak = (double)r->config.initial_capital, drawdown = 0;
  int wins = 0, n = 0;
  r->slippage = 0;
  r->events = r->orders = r->fills = r->rejected = 0;
  r->pnl = r->fees = r->gross = 0;
  r->hash = r->market_hash = 14695981039346656037UL;
  if (r->states)
    for (i = 0; i < r->config.symbols; ++i) {
      HftSymbol *s = &r->states[i];
      HftTop t = hft_top(&s->book, &r->config, 0);
      hft_i64 mark = hft_mid(t);
      r->events += s->event;
      r->orders += s->orders;
      r->fills += s->fills;
      r->rejected += s->rejected;
      r->pnl += s->position.cash + (hft_i64)s->position.qty * mark;
      r->fees += s->position.fees;
      r->gross += hft_abs64(s->position.qty) * mark;
      r->hash = hft_hash_word(r->hash, s->hash, 8);
      r->market_hash = hft_hash_word(r->market_hash, s->market_hash, 8);
      r->slippage += (double)s->slippage_sum;
    }
  for (i = 0; i < r->point_count; ++i) {
    double equity =
        (double)r->config.initial_capital + (double)r->points[i].pnl;
    if (equity > peak)
      peak = equity;
    if (peak > 0 && ((peak - equity) / peak) > drawdown)
      drawdown = (peak - equity) / peak;
    if (i) {
      double previous = (double)r->config.initial_capital +
                        (double)r->points[i - 1].pnl,
             delta = equity - previous,
             ret = previous != 0 ? delta / previous : 0;
      sum += ret;
      sq += ret * ret;
      if (ret < 0)
        down += ret * ret;
      if (delta > 0) {
        positive += delta;
        ++wins;
      } else
        negative -= delta;
      ++n;
    }
  }
  r->max_drawdown = drawdown;
  r->return_fraction = (double)r->pnl / (double)r->config.initial_capital;
  if (n > 1) {
    double mean = sum / n, var = fmax(0, sq / n - mean * mean);
    r->sharpe = var > 0 ? mean / sqrt(var) : 0;
    r->sortino = down > 0 ? mean / sqrt(down / n) : 0;
    r->win_rate = (double)wins / n;
  }
  r->profit_factor = negative > 0 ? positive / negative : NAN;
  r->throughput = r->seconds > 0 ? (double)r->events / r->seconds : 0;
  if (r->fills)
    r->slippage /= r->fills;
}
void hft_result_free(HftResult *r) {
  free(r->states);
  free(r->points);
  free(r->worker_seconds);
  r->states = NULL;
  r->points = NULL;
  r->worker_seconds = NULL;
}
int hft_result_equivalent(const HftResult *a, const HftResult *b) {
  int i;
  if (a->config.mode || b->config.mode || a->events != b->events ||
      a->hash != b->hash || a->market_hash != b->market_hash ||
      a->pnl != b->pnl)
    return 0;
  if (a->states && b->states) {
    if (a->config.symbols != b->config.symbols)
      return 0;
    for (i = 0; i < a->config.symbols; ++i)
      if (a->states[i].hash != b->states[i].hash ||
          a->states[i].position.qty != b->states[i].position.qty ||
          a->states[i].position.cash != b->states[i].position.cash)
        return 0;
  }
  return 1;
}
void hft_make_snapshot(const HftResult *r, int symbol, HftSnapshot *out) {
  int i, best;
  const HftSymbol *s;
  memset(out, 0, sizeof(*out));
  if (!r->states)
    return;
  if (symbol < 0 || symbol >= r->config.symbols)
    symbol = 0;
  s = &r->states[symbol];
  out->symbol = symbol;
  out->top = hft_top(&s->book, &r->config, 0);
  memcpy(out->candles, s->candles, sizeof(out->candles));
  memcpy(out->equity, s->equity, sizeof(out->equity));
  memcpy(out->inventory, s->inventory, sizeof(out->inventory));
  out->count = s->trace_count;
  out->head = s->trace_head;
  out->events = s->event;
  out->orders = s->orders;
  out->fills = s->fills;
  out->last_fill = s->last_fill;
  out->strategy = r->config.strategy;
  out->regime = s->regime;
  out->selected = s->selected;
  out->halted = s->halted;
  out->pnl = s->position.cash + (hft_i64)s->position.qty * hft_mid(out->top);
  best = out->top.bid ? (int)(out->top.bid - s->book.low) : -1;
  for (i = 0; i < 20 && best - i >= 0; ++i) {
    out->bids[i] = s->book.low + best - i;
    out->bid_qty[i] = s->book.levels[best - i].volume;
  }
  best = out->top.ask ? (int)(out->top.ask - s->book.low) : r->config.depth;
  for (i = 0; i < 20 && best + i < r->config.depth; ++i) {
    out->asks[i] = s->book.low + best + i;
    out->ask_qty[i] = s->book.levels[HFT_LEVELS + best + i].volume;
  }
  for (i = 0; i < r->config.symbols && i < 64; ++i)
    out->prices[i] =
        (double)hft_mid(hft_top(&r->states[i].book, &r->config, 0)) *
        r->config.tick_size;
  snprintf(out->status, sizeof(out->status), "%s | %s | %s",
           hft_engine_names[r->config.engine],
           hft_strategy_names[r->config.strategy], hft_regime_names[s->regime]);
}
void hft_snapshot_publish(HftTripleBuffer *b, const HftSnapshot *s) {
  b->slots[b->back] = *s;
  b->back = (int)(hft_exchange(&b->middle, (uint64_t)b->back | 4) & 3);
}
int hft_snapshot_consume(HftTripleBuffer *b, HftSnapshot *s) {
  if (!(hft_load(&b->middle) & 4))
    return 0;
  b->front = (int)(hft_exchange(&b->middle, (uint64_t)b->front) & 3);
  *s = b->slots[b->front];
  return 1;
}
