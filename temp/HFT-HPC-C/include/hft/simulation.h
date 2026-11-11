/* Purpose: one shared C/OpenCL C market, book and decision implementation.
 * Units 1-4: cache locality, symbol decomposition, deterministic GPU/AI
 * inference. One work item owns an even/odd symbol pair, including
 * pairs-strategy coupling. Complexity: optimized matching O(fills+depth/64);
 * reference scan O(capacity/fill). No allocation, locks, I/O, virtual dispatch
 * or platform math in the event path. */
#ifndef HFT_SIMULATION_H
#define HFT_SIMULATION_H
#include "numeric.h"
static inline void hft_book_clear(HFT_GLOBAL HftBook *b,
                                  HFT_CONST HftConfig *c) {
  int i;
  for (i = 0; i <= HFT_ORDERS; ++i)
    b->nodes[i].live = 0;
  for (i = 0; i < 2 * HFT_LEVELS; ++i) {
    b->levels[i].head = 0;
    b->levels[i].tail = 0;
    b->levels[i].volume = 0;
  }
  for (i = 0; i < 2 * (HFT_LEVELS / 64); ++i)
    b->occupied[i] = 0;
  b->sequence = 0;
  b->time = 0;
  b->low = c->initial_price - c->depth / 2;
  b->last = c->initial_price;
  b->live = b->rejections = 0;
}
static inline int hft_best(HFT_GLOBAL const HftBook *b, HFT_CONST HftConfig *c,
                           int side, int naive) {
  int i, best = 0;
  if (naive) {
    for (i = 1; i <= c->capacity; ++i)
      if (b->nodes[i].live && b->nodes[i].order.side == side) {
        if (!best ||
            (side == 0
                 ? b->nodes[i].order.price > b->nodes[best].order.price
                 : b->nodes[i].order.price < b->nodes[best].order.price) ||
            (b->nodes[i].order.price == b->nodes[best].order.price &&
             b->nodes[i].sequence < b->nodes[best].sequence))
          best = i;
      }
    return best;
  }
  if (side == 0) {
    for (i = (c->depth - 1) / 64; i >= 0; --i) {
      hft_u64 bits = b->occupied[i];
      if (bits) {
        int bit = 63;
        while (((bits >> bit) & 1) == 0)
          --bit;
        return b->levels[i * 64 + bit].head;
      }
    }
  } else {
    for (i = 0; i < (c->depth + 63) / 64; ++i) {
      hft_u64 bits = b->occupied[HFT_LEVELS / 64 + i];
      if (bits) {
        int bit = 0;
        while (((bits >> bit) & 1) == 0)
          ++bit;
        return b->levels[HFT_LEVELS + i * 64 + bit].head;
      }
    }
  }
  return 0;
}
static inline HftTop hft_top(HFT_GLOBAL const HftBook *b,
                             HFT_CONST HftConfig *c, int naive) {
  HftTop t;
  int bid = hft_best(b, c, 0, naive), ask = hft_best(b, c, 1, naive);
  t.bid = bid ? b->nodes[bid].order.price : 0;
  t.ask = ask ? b->nodes[ask].order.price : 0;
  t.bid_qty = bid ? b->levels[b->nodes[bid].level].volume : 0;
  t.ask_qty = ask ? b->levels[b->nodes[ask].level].volume : 0;
  t.last = b->last;
  t.ts = b->time;
  return t;
}
static inline hft_i64 hft_mid(HftTop t) {
  return t.bid && t.ask ? (t.bid + t.ask) / 2 : t.last;
}
static inline void hft_book_remove(HFT_GLOBAL HftBook *b, int id) {
  HftNode n = b->nodes[id];
  HFT_GLOBAL HftLevel *l = &b->levels[n.level];
  if (n.previous)
    b->nodes[n.previous].next = n.next;
  else
    l->head = n.next;
  if (n.next)
    b->nodes[n.next].previous = n.previous;
  else
    l->tail = n.previous;
  l->volume -= n.order.qty;
  if (!l->head)
    b->occupied[n.level / 64] &= ~((hft_u64)1 << (n.level % 64));
  b->nodes[id].live = 0;
  --b->live;
}
static inline int hft_book_cancel(HFT_GLOBAL HftBook *b, HFT_CONST HftConfig *c,
                                  int id) {
  if (id < 1 || id > c->capacity || !b->nodes[id].live)
    return 0;
  hft_book_remove(b, id);
  return 1;
}
static inline void hft_account(HFT_GLOBAL HftPosition *p, HftFill f,
                               hft_i64 fee) {
  int delta = f.side ? -f.qty : f.qty, old = p->qty;
  p->cash -= (hft_i64)delta * f.price + fee * f.qty;
  p->fees += fee * f.qty;
  if (!old || (old > 0) == (delta > 0))
    p->cost += (hft_i64)f.qty * f.price;
  else {
    hft_i64 closed = hft_abs64(old) < f.qty ? hft_abs64(old) : f.qty,
            cost = p->cost * closed / hft_abs64(old);
    p->realized += (old > 0 ? 1 : -1) * (closed * f.price - cost);
    p->cost -= cost;
    if (f.qty > hft_abs64(old))
      p->cost = (f.qty - hft_abs64(old)) * f.price;
  }
  p->qty += delta;
}
static inline void hft_on_fill(HFT_GLOBAL HftSymbol *s, HFT_CONST HftConfig *c,
                               HftOrder maker, HftOrder taker, hft_i64 price,
                               int qty) {
  HftFill f;
  int k;
  for (k = 0; k < 2; ++k) {
    HftOrder o = k ? taker : maker;
    if (o.agent == 5) {
      f.ts = taker.ts;
      f.price = price;
      f.id = o.id;
      f.symbol = o.symbol;
      f.qty = qty;
      f.side = o.side;
      f.maker = !k;
      f.pad = 0;
      hft_account(&s->position, f, c->fee);
      s->hash = hft_hash_fill(s->hash, f);
      ++s->fills;
      s->filled_qty += (hft_u64)qty;
      s->last_fill = f;
      s->turnover += (hft_i64)qty * price;
      s->slippage_sum +=
          (hft_i64)qty * (o.side ? -1 : 1) * (price - s->previous_mid);
    } else if (o.agent >= 0 && o.agent < 5)
      s->agent_inventory[o.agent] += (o.side ? -qty : qty);
  }
  s->book.last = price;
  s->traded_qty += qty;
  s->traded_value += price * qty;
  s->signed_volume += taker.side ? -qty : qty;
  if (!s->active_candle.volume) {
    s->active_candle.open = s->active_candle.high = s->active_candle.low =
        price;
  }
  if (price > s->active_candle.high)
    s->active_candle.high = price;
  if (price < s->active_candle.low)
    s->active_candle.low = price;
  s->active_candle.close = price;
  s->active_candle.volume += qty;
  s->active_candle.ts = taker.ts;
}
static inline int hft_submit(HFT_GLOBAL HftSymbol *s, HFT_CONST HftConfig *c,
                             HftOrder order, int market, int naive) {
  HFT_GLOBAL HftBook *b = &s->book;
  int id, level;
  HFT_GLOBAL HftLevel *l;
  b->time = order.ts;
  if (order.qty < 1 || order.qty > 1000 || order.symbol != (hft_u32)s->symbol)
    return 0;
  if (!market && (order.price < b->low || order.price >= b->low + c->depth)) {
    ++b->rejections;
    return 0;
  }
  if (!order.ioc && (order.id < 1 || order.id > (hft_u32)c->capacity ||
                     b->nodes[order.id].live)) {
    ++b->rejections;
    return 0;
  }
  while (order.qty > 0) {
    HftOrder maker;
    int q;
    id = hft_best(b, c, !order.side, naive);
    if (!id)
      break;
    maker = b->nodes[id].order;
    if (!market && (order.side == 0 ? maker.price > order.price
                                    : maker.price < order.price))
      break;
    if (maker.agent == 5 && order.agent == 5) {
      hft_book_remove(b, id);
      continue;
    }
    q = order.qty < maker.qty ? order.qty : maker.qty;
    hft_on_fill(s, c, maker, order, maker.price, q);
    order.qty -= q;
    b->nodes[id].order.qty -= q;
    b->levels[b->nodes[id].level].volume -= q;
    if (!b->nodes[id].order.qty)
      hft_book_remove(b, id);
  }
  if (order.ioc || market || !order.qty)
    return 1;
  id = (int)order.id;
  level = order.side * HFT_LEVELS + (int)(order.price - b->low);
  l = &b->levels[level];
  b->nodes[id].order = order;
  b->nodes[id].sequence = ++b->sequence;
  b->nodes[id].previous = l->tail;
  b->nodes[id].next = 0;
  b->nodes[id].level = level;
  b->nodes[id].live = 1;
  if (l->tail)
    b->nodes[l->tail].next = id;
  else
    l->head = id;
  l->tail = id;
  l->volume += order.qty;
  ++b->live;
  b->occupied[level / 64] |= (hft_u64)1 << (level % 64);
  return 1;
}
static inline void hft_apply_event(HFT_GLOBAL HftSymbol *s,
                                   HFT_CONST HftConfig *c, HftEvent e,
                                   int naive) {
  HftOrder o;
  s->book.time = e.ts;
  s->market_hash = hft_hash_word(s->market_hash, (hft_u64)e.price, 8);
  s->market_hash = hft_hash_word(s->market_hash, e.ts, 8);
  s->market_hash = hft_hash_word(s->market_hash, e.id, 4);
  s->market_hash = hft_hash_word(s->market_hash, e.type, 1);
  s->market_hash = hft_hash_word(s->market_hash, (hft_u64)e.qty, 4);
  if (e.type == HFT_CANCEL) {
    s->cancels += (hft_u64)hft_book_cancel(&s->book, c, (int)e.id);
    return;
  }
  o.id = e.id;
  o.symbol = e.symbol;
  o.price = e.price;
  o.ts = e.ts;
  o.qty = e.qty;
  o.side = e.side;
  o.agent = e.agent;
  o.ioc = e.type == HFT_MARKET;
  hft_submit(s, c, o, e.type == HFT_MARKET, naive);
}
static inline HftEvent hft_generate(HFT_GLOBAL HftSymbol *s,
                                    HFT_CONST HftConfig *c, HftTop t) {
  HftEvent e;
  HftRandom r = hft_random(c->seed, (hft_u32)s->symbol, 1, s->event),
            j = hft_random(c->seed, (hft_u32)s->symbol, 2, s->event);
  HftRandom common = hft_random(c->seed, 0, 77, s->event),
            sector = hft_random(c->seed, (hft_u32)s->symbol / 16, 78, s->event),
            own = hft_random(c->seed, (hft_u32)s->symbol, 79, s->event);
  double dt = 0, vol = c->volatility, drift = c->drift, anchor;
  int attempt = 0, agent = 0, total = 0, ticket, i, crash;
  hft_i64 mark = hft_mid(t), offset;
  if (c->auto_regime && s->event && s->event % 2048 == 0 && r.x[3] % 4 == 0)
    s->regime = (int)((r.x[3] / 4) % 7);
  for (;;) {
    double upper = c->hawkes_mu + s->excitation, wait;
    HftRandom rr = hft_random(c->seed, (hft_u32)s->symbol,
                              1000 + (hft_u32)attempt++, s->event);
    wait = -hft_log(hft_uniform(rr.x[0])) / upper;
    dt += wait;
    s->excitation *= hft_exp(-c->hawkes_beta * wait);
    if (hft_uniform(rr.x[1]) * upper <= c->hawkes_mu + s->excitation) {
      s->excitation += c->hawkes_alpha;
      break;
    }
  }
  s->time += (hft_u64)hft_max(1, dt * 1e9);
  if (s->regime == HFT_BULL)
    drift += .003;
  if (s->regime == HFT_BEAR)
    drift -= .003;
  if (s->regime == HFT_VOLATILE)
    vol *= 4;
  if (s->regime == HFT_SIDEWAYS)
    vol *= .6;
  crash = s->regime == HFT_CRASH && s->event % 4000 >= 1200 &&
          s->event % 4000 < 1600;
  if (s->regime == HFT_CRASH && s->event % 4000 >= 1600 &&
      s->event % 4000 < 2400)
    drift += .02;
  s->fundamental +=
      (drift - .5 * vol * vol -
       c->jump_rate *
           (hft_exp(c->jump_mean + .5 * c->jump_std * c->jump_std) - 1)) *
          dt +
      vol * hft_sqrt(dt) *
          (hft_sqrt(.2) * hft_normal(common.x[0], common.x[1]) +
           hft_sqrt(.4) * hft_normal(sector.x[0], sector.x[1]) +
           hft_sqrt(.4) * hft_normal(own.x[0], own.x[1])) +
      (c->jump_mean + c->jump_std * hft_normal(j.x[2], j.x[3])) *
          hft_poisson(j.x[0], j.x[1], c->jump_rate * dt);
  if (s->regime == HFT_SIDEWAYS)
    s->fundamental +=
        .8 * (hft_log((double)c->initial_price) - s->fundamental) * dt;
  if (s->regime == HFT_NEWS && s->event % 4000 == 1200)
    s->fundamental += .03;
  s->fundamental = hft_max(
      hft_log((double)(s->book.low + 8)),
      hft_min(hft_log((double)(s->book.low + c->depth - 9)), s->fundamental));
  anchor = hft_exp(s->fundamental);
  for (i = 0; i < 5; ++i)
    total += c->weights[i];
  ticket = (int)(r.x[0] % (hft_u32)total);
  while (agent < 4 && ticket >= c->weights[agent])
    ticket -= c->weights[agent++];
  e.ts = s->time;
  e.price = 0;
  e.id = 1 + (hft_u32)(s->event % (hft_u64)(c->capacity - 16));
  e.symbol = (hft_u32)s->symbol;
  e.qty = 1 + (int)(r.x[2] % 8);
  e.side = (unsigned char)(r.x[1] & 1);
  e.agent = (unsigned char)agent;
  e.type = HFT_ADD;
  e.flags = 0;
  if (hft_uniform(r.x[3]) < c->cancel_probability || (crash && agent == 1)) {
    e.type = HFT_CANCEL;
    e.id = 1 + r.x[2] % (hft_u32)(c->capacity - 16);
    return e;
  }
  offset = 1 + (hft_i64)(-hft_log(hft_uniform(r.x[2])) / .6931471805599453);
  if (agent == 0 && r.x[1] % 5 == 0)
    e.type = HFT_MARKET;
  if (agent == 1) {
    hft_i64 micro = t.bid && t.ask ? (t.ask * t.bid_qty + t.bid * t.ask_qty) /
                                         ((hft_i64)t.bid_qty + t.ask_qty)
                                   : mark;
    offset = s->regime == HFT_VOLATILE ? 5 : 2;
    e.price = micro - hft_clamp64(s->agent_inventory[1] / 50, -20, 20) +
              (e.side ? offset : -offset);
    e.qty = 10;
  }
  if (agent == 2) {
    e.side = (unsigned char)(mark < s->previous_mid);
    e.type = HFT_MARKET;
  }
  if (agent == 3) {
    e.side = (unsigned char)((double)mark >= anchor);
    e.type = hft_abs((double)mark - anchor) > 3 ? HFT_MARKET : HFT_ADD;
    e.price = (hft_i64)(anchor + .5) + (e.side ? 1 : -1);
  }
  if (agent == 4) {
    e.side = (unsigned char)((s->event / 64) % 2 == 0);
    e.type = HFT_MARKET;
    e.qty = 15;
  }
  if (crash && s->event % 3 == 0) {
    e.agent = 4;
    e.side = 1;
    e.type = HFT_MARKET;
    e.qty = 100;
  }
  if (!e.price)
    e.price = mark + (e.side ? offset : -offset);
  return e;
}
static inline void hft_features(HFT_GLOBAL HftSymbol *s, HFT_CONST HftConfig *c,
                                HftTop t, int naive) {
  HFT_GLOBAL HftFeatures *f = &s->features;
  hft_i64 price = hft_mid(t), change = price - f->last_mid, old;
  int i, n = (int)(f->count < (hft_u64)c->slow ? f->count : (hft_u64)c->slow);
  double mean, var, imbalance, spread = (double)(t.ask - t.bid),
                               raw[HFT_FEATURES];
  if (f->count >= (hft_u64)c->fast)
    f->sum_fast -= f->prices[(f->count - (hft_u64)c->fast) % HFT_WINDOW];
  if (f->count >= (hft_u64)c->slow) {
    old = f->prices[(f->count - (hft_u64)c->slow) % HFT_WINDOW];
    f->sum_slow -= old;
    f->square -= old * old;
  }
  f->sum_fast += price;
  f->sum_slow += price;
  f->square += price * price;
  f->high = 0;
  f->low = 1000000000;
  for (i = 0; i < n; ++i) {
    old = f->prices[(f->count - 1 - (hft_u64)i) % HFT_WINDOW];
    if (old > f->high)
      f->high = old;
    if (old < f->low)
      f->low = old;
  }
  f->prices[f->count % HFT_WINDOW] = price;
  f->volume[f->count % HFT_WINDOW] = (hft_i64)t.bid_qty + t.ask_qty;
  ++f->count;
  n = (int)(f->count < (hft_u64)c->slow ? f->count : (hft_u64)c->slow);
  if (naive) {
    f->sum_fast = f->sum_slow = f->square = 0;
    for (i = 0; i < n; ++i) {
      old = f->prices[(f->count - 1 - (hft_u64)i) % HFT_WINDOW];
      f->sum_slow += old;
      f->square += old * old;
      if (i < c->fast)
        f->sum_fast += old;
    }
  }
  if (f->count == 1) {
    f->ema_fast = f->ema_slow = price * 65536;
    f->gain = f->loss = 0;
  } else {
    f->ema_fast += (2 * (price * 65536 - f->ema_fast)) / (c->fast + 1);
    f->ema_slow += (2 * (price * 65536 - f->ema_slow)) / (c->slow + 1);
    f->gain = (13 * f->gain + (change > 0 ? change : 0) * 65536) / 14;
    f->loss = (13 * f->loss + (change < 0 ? -change : 0) * 65536) / 14;
  }
  f->macd_signal += (2 * ((f->ema_fast - f->ema_slow) - f->macd_signal)) / 10;
  f->vwap_num = s->traded_qty ? s->traded_value : price;
  f->vwap_den = s->traded_qty ? s->traded_qty : 1;
  mean = (double)f->sum_slow / n;
  var = hft_max(0, (double)f->square / n - mean * mean);
  imbalance = (double)(t.bid_qty - t.ask_qty) /
              hft_max(1, (double)t.bid_qty + t.ask_qty);
  for (i = 0; i < HFT_FEATURES; ++i)
    raw[i] = 0;
  {
    int lags[5] = {1, 5, 10, 50, 100};
    for (i = 0; i < 5; ++i)
      if (f->count > (hft_u64)lags[i]) {
        old = f->prices[(f->count - 1 - (hft_u64)lags[i]) % HFT_WINDOW];
        raw[i] = (double)(price - old) / hft_max(1, (double)old) * 1000;
      }
  }
  raw[5] = (double)(f->ema_fast - f->ema_slow) / 65536;
  raw[6] = hft_sqrt(var);
  raw[7] = spread;
  raw[8] = imbalance;
  raw[9] = (double)f->gain / hft_max(1, (double)f->gain + f->loss);
  raw[10] = (price - mean) / hft_max(1, hft_sqrt(var));
  raw[11] = (double)s->position.qty / c->max_position;
  raw[12] = (double)(f->ema_fast - f->ema_slow - f->macd_signal) / 65536;
  raw[13] = (double)(price - s->book.last);
  raw[14] = (double)(price - f->vwap_num / f->vwap_den);
  raw[15] = (double)t.bid_qty / 100;
  raw[16] = (double)t.ask_qty / 100;
  raw[17] = (double)s->orders / hft_max(1, (double)s->event);
  raw[18] = (double)s->cancels / hft_max(1, (double)s->event);
  raw[19] = (double)s->fills / hft_max(1, (double)s->orders);
  raw[20] = (double)(s->position.cash + (hft_i64)s->position.qty * price) /
            hft_max(1, (double)c->initial_capital);
  raw[21] = (double)(f->high - f->low);
  raw[22] = (double)(price - f->high);
  raw[23] = (double)(price - f->low);
  raw[24] = (double)s->excitation / c->hawkes_mu;
  raw[25] = (double)(s->time % 1000000000UL) / 1e9;
  raw[26] = (double)s->signed_volume / hft_max(1, (double)s->traded_qty);
  raw[27] = (double)s->book.live / c->capacity;
  raw[28] = (double)s->book.rejections / hft_max(1, (double)s->event);
  raw[29] =
      (double)(f->sum_fast /
               (f->count < (hft_u64)c->fast ? (hft_i64)f->count : c->fast)) -
      mean;
  raw[30] = change;
  raw[31] = 1;
  for (i = 0; i < HFT_FEATURES; ++i) {
    double delta = raw[i] - f->norm_mean[i];
    f->norm_mean[i] += delta / (double)f->count;
    f->norm_m2[i] += delta * (raw[i] - f->norm_mean[i]);
    f->features[i] = (float)hft_max(
        -8,
        hft_min(8, (raw[i] - f->norm_mean[i]) /
                       hft_max(1, hft_sqrt(f->norm_m2[i] / (double)f->count))));
  }
  f->last_mid = price;
}
static inline void hft_infer_shared(HFT_CONST HftModel *m,
                                    HFT_GLOBAL const float *input,
                                    float *probabilities) {
  float a[64], b[32], maxval, sum;
  int i, j;
  for (i = 0; i < 64; ++i) {
    float v = m->weights[2048 + i];
    for (j = 0; j < 32; ++j)
      v += input[j] * m->weights[i * 32 + j];
    a[i] = v > 0 ? v : 0;
  }
  for (i = 0; i < 32; ++i) {
    float v = m->weights[4160 + i];
    for (j = 0; j < 64; ++j)
      v += a[j] * m->weights[2112 + i * 64 + j];
    b[i] = v > 0 ? v : 0;
  }
  maxval = -1e30f;
  for (i = 0; i < 3; ++i) {
    float v = m->weights[4288 + i];
    for (j = 0; j < 32; ++j)
      v += b[j] * m->weights[4192 + i * 32 + j];
    probabilities[i] = v;
    if (v > maxval)
      maxval = v;
  }
  sum = 0;
  for (i = 0; i < 3; ++i) {
    probabilities[i] = (float)hft_exp((double)(probabilities[i] - maxval));
    sum += probabilities[i];
  }
  for (i = 0; i < 3; ++i)
    probabilities[i] /= sum;
}
static inline int hft_q_state(HFT_GLOBAL const HftSymbol *s, HftTop t,
                              HFT_CONST HftConfig *c) {
  int inv = s->position.qty < -c->max_position / 3
                ? 0
                : (s->position.qty > c->max_position / 3 ? 2 : 1);
  int im = t.bid_qty * 2 < t.ask_qty ? 0 : (t.ask_qty * 2 < t.bid_qty ? 2 : 1);
  int spread = t.ask - t.bid < 3 ? 0 : (t.ask - t.bid > 8 ? 2 : 1);
  int vol = hft_abs64(s->features.ema_fast - s->features.ema_slow) > 65536 * 3
                ? 2
                : 1;
  return ((inv * 3 + im) * 3 + spread) * 3 + vol;
}
static inline HftSignal hft_signal(HFT_GLOBAL HftSymbol *s,
                                   HFT_CONST HftConfig *c, HftTop t,
                                   hft_i64 pair_spread) {
  HftSignal out;
  HFT_GLOBAL HftFeatures *f = &s->features;
  hft_i64 p = hft_mid(t), v, dev, variance;
  int kind = c->strategy, dir = 0, i;
  out.direction = out.qty = out.is_quote = out.pad = 0;
  out.price = out.bid = out.ask = 0;
  out.confidence = 0;
  out.selected = kind;
  if (f->count < (hft_u64)c->slow || !t.bid || !t.ask)
    return out;
  if (kind == HFT_SELECTOR) {
    if (s->event % 64 == 0) {
      float feature[4];
      double best = 1e300;
      int regime = 0, arm = 0, total = 0;
      hft_i64 pnl = s->position.cash + (hft_i64)s->position.qty * p;
      int previous = s->detected * 3 + s->selected;
      if (s->event >= 64) {
        s->bandit_reward[previous] += (double)(pnl - s->bandit_last_pnl);
        ++s->bandit_count[previous];
      }
      s->bandit_last_pnl = pnl;
      feature[0] = f->features[5];
      feature[1] = f->features[6];
      feature[2] = f->features[7];
      feature[3] = f->features[8];
      for (i = 0; i < 4; ++i) {
        int k;
        double d = 0;
        for (k = 0; k < 4; ++k) {
          double x = feature[k] - c->model.centroids[i * 4 + k];
          d += x * x;
        }
        if (d < best) {
          best = d;
          regime = i;
        }
      }
      s->detected = regime;
      for (i = 0; i < 3; ++i)
        total += s->bandit_count[regime * 3 + i];
      best = -1e300;
      for (i = 0; i < 3; ++i) {
        int n = s->bandit_count[regime * 3 + i];
        double score = n ? s->bandit_reward[regime * 3 + i] / n +
                               100 * hft_sqrt(2 * hft_log(total + 1.0) / n)
                         : 1e20 - i;
        if (score > best) {
          best = score;
          arm = i;
        }
      }
      s->selected = arm;
    }
    kind = s->selected == 0 ? HFT_SMA
                            : (s->selected == 1 ? HFT_REVERSION : HFT_MAKER);
    out.selected = kind;
  }
  if (kind == HFT_SMA) {
    v = f->sum_fast * c->slow - f->sum_slow * c->fast;
    dir = (v > 0) - (v < 0);
  }
  if (kind == HFT_EMA) {
    v = f->ema_fast - f->ema_slow;
    dir = (v > 0) - (v < 0);
  }
  if (kind == HFT_REVERSION) {
    dev = p * c->slow - f->sum_slow;
    variance = c->slow * f->square - f->sum_slow * f->sum_slow;
    if ((double)(dev * dev) > c->z_enter * c->z_enter * (double)variance)
      dir = (dev < 0) - (dev > 0);
    else if ((double)(dev * dev) < c->z_exit * c->z_exit * (double)variance &&
             s->position.qty)
      dir = s->position.qty > 0 ? -1 : 1;
  }
  if (kind == HFT_BREAKOUT) {
    if (p > f->high && t.bid_qty > t.ask_qty)
      dir = 1;
    if (p < f->low && t.ask_qty > t.bid_qty)
      dir = -1;
  }
  if (kind == HFT_RSI) {
    if (100 * f->gain < 30 * (f->gain + f->loss))
      dir = 1;
    if (100 * f->gain > 70 * (f->gain + f->loss))
      dir = -1;
  }
  if (kind == HFT_MACD) {
    v = f->ema_fast - f->ema_slow - f->macd_signal;
    dir = (v > 0) - (v < 0);
  }
  if (kind == HFT_IMBALANCE) {
    double imbalance = (double)(t.bid_qty - t.ask_qty) /
                       hft_max(1, (double)t.bid_qty + t.ask_qty);
    if (imbalance > c->imbalance)
      dir = 1;
    if (imbalance < -c->imbalance)
      dir = -1;
  }
  if (kind == HFT_PAIRS) {
    if (pair_spread > 200)
      dir = (s->symbol & 1) ? 1 : -1;
    if (pair_spread < -200)
      dir = (s->symbol & 1) ? -1 : 1;
    if (hft_abs64(pair_spread) < 50 && s->position.qty)
      dir = s->position.qty > 0 ? -1 : 1;
  }
  if (kind == HFT_EXECUTION) {
    int target =
        (int)hft_min(c->parent_qty, (double)c->parent_qty * (double)s->event /
                                        c->parent_horizon);
    if (s->execution_done < target)
      dir = 1;
  }
  if (kind == HFT_MLP) {
    float probs[3];
    int best = 0;
    hft_infer_shared(&c->model, f->features, probs);
    for (i = 1; i < 3; ++i)
      if (probs[i] > probs[best])
        best = i;
    dir = best - 1;
    out.confidence = probs[best];
    if (out.confidence < .4f)
      dir = 0;
  }
  if (kind == HFT_MAKER || kind == HFT_QLEARNING) {
    hft_i64 skew = s->position.qty / 10, width = 2;
    if (kind == HFT_QLEARNING) {
      int state = hft_q_state(s, t, c), action = 0;
      for (i = 1; i < HFT_ACTIONS; ++i)
        if (c->model.q[state * 9 + i] > c->model.q[state * 9 + action])
          action = i;
      skew += (action % 3 - 1) * 2;
      width = 1 + action / 3 * 2;
    } else {
      double variance =
          (double)f->square / c->slow -
          ((double)f->sum_slow / c->slow) * ((double)f->sum_slow / c->slow);
      double gamma = .05, horizon = 1,
             spread = gamma * hft_max(0, variance) * horizon +
                      2 / gamma * hft_log(1 + gamma / 1.5);
      width = (hft_i64)hft_max(1, hft_min(20, spread * .5));
      skew += (hft_i64)(s->position.qty * gamma * hft_max(0, variance) *
                        horizon * .01);
    }
    out.is_quote = 1;
    out.bid = p - skew - width;
    out.ask = p - skew + width;
    out.qty = c->quantity;
    return out;
  }
  out.direction = dir;
  out.qty = dir ? c->quantity : 0;
  if (kind == HFT_PAIRS && (s->symbol & 1))
    out.qty =
        (int)hft_min(c->max_order, hft_max(1, c->quantity * s->pair_beta + .5));
  out.price = dir > 0 ? t.ask : t.bid;
  if (!out.confidence)
    out.confidence = dir ? 1 : 0;
  return out;
}
static inline int hft_risk(HFT_GLOBAL HftSymbol *s, HFT_CONST HftConfig *c,
                           HftOrder o, HftTop t, int emergency) {
  hft_i64 projected = s->position.qty + (o.side ? -o.qty : o.qty),
          p = hft_mid(t), pnl = s->position.cash + (hft_i64)s->position.qty * p;
  hft_u64 capacity = (hft_u64)c->max_orders_second * 1000000000UL, elapsed;
  if (emergency == 1)
    return 1;
  if (o.qty < 1 || o.qty > c->max_order ||
      hft_abs64(projected) > c->max_position ||
      hft_abs64(projected) * p > c->max_gross / c->symbols ||
      hft_abs64(o.price - p) > c->collar || pnl <= -c->max_loss / c->symbols ||
      s->halted)
    return 0;
  if (emergency == 2)
    return 1;
  if (!s->rate_initialized) {
    s->rate_credit = capacity;
    s->rate_last = s->time;
    s->rate_initialized = 1;
  }
  elapsed = s->time - s->rate_last;
  if (elapsed > 1000000000UL)
    elapsed = 1000000000UL;
  s->rate_credit += elapsed * (hft_u64)c->max_orders_second;
  if (s->rate_credit > capacity)
    s->rate_credit = capacity;
  s->rate_last = s->time;
  if (s->rate_credit < 1000000000UL)
    return 0;
  s->rate_credit -= 1000000000UL;
  return 1;
}
static inline void hft_emit(HFT_GLOBAL HftSymbol *s, HFT_CONST HftConfig *c,
                            HftOrder o, HftTop t, int naive, hft_u64 delay,
                            int emergency) {
  if (!hft_risk(s, c, o, t, emergency)) {
    ++s->rejected;
    return;
  }
  s->hash = hft_hash_order(s->hash, o);
  ++s->orders;
  if (c->mode && delay && !emergency) {
    if (s->delayed_count < 32) {
      o.ts += delay;
      s->delayed[s->delayed_count++] = o;
    } else
      ++s->delayed_dropped;
  } else
    hft_submit(s, c, o, 0, naive);
}
static inline void hft_pending(HFT_GLOBAL HftSymbol *s, HFT_CONST HftConfig *c,
                               int naive) {
  int i = 0;
  while (i < s->delayed_count) {
    if (s->delayed[i].ts <= s->time) {
      HftOrder o = s->delayed[i];
      int k;
      for (k = i + 1; k < s->delayed_count; ++k)
        s->delayed[k - 1] = s->delayed[k];
      --s->delayed_count;
      o.ts = s->time;
      if (hft_risk(s, c, o, hft_top(&s->book, c, naive), 2))
        hft_submit(s, c, o, 0, naive);
      else
        ++s->rejected;
    } else
      ++i;
  }
}
static inline void hft_background(HFT_GLOBAL HftSymbol *s,
                                  HFT_CONST HftConfig *c, int naive) {
  HftTop t = hft_top(&s->book, c, naive);
  HftEvent e = hft_generate(s, c, t);
  s->previous_mid = hft_mid(t);
  hft_apply_event(s, c, e, naive);
  hft_pending(s, c, naive);
  t = hft_top(&s->book, c, naive);
  hft_features(s, c, t, naive);
}
static inline void hft_decision(HFT_GLOBAL HftSymbol *s, HFT_CONST HftConfig *c,
                                hft_i64 pair_spread, int naive, hft_u64 delay) {
  HftTop t = hft_top(&s->book, c, naive);
  hft_i64 mark = hft_mid(t),
          pnl = s->position.cash + (hft_i64)s->position.qty * mark;
  HftSignal signal = hft_signal(s, c, t, pair_spread);
  HftOrder o;
  int k, old_qty = s->position.qty;
  if (pnl <= -c->max_loss / c->symbols)
    s->halted = 1;
  if (s->halted) {
    hft_book_cancel(&s->book, c, c->capacity);
    hft_book_cancel(&s->book, c, c->capacity - 1);
    if (s->position.qty) {
      o.id = (hft_u32)(c->capacity + 1 + s->event);
      o.symbol = (hft_u32)s->symbol;
      o.side = s->position.qty > 0;
      o.qty = (int)hft_min(c->max_order, (double)hft_abs64(s->position.qty));
      o.price = o.side ? s->book.low : s->book.low + c->depth - 1;
      o.ts = s->time;
      o.agent = 5;
      o.ioc = 1;
      hft_emit(s, c, o, t, naive, 0, 1);
    }
  } else if (signal.is_quote) {
    for (k = 0; k < 2; ++k) {
      hft_book_cancel(&s->book, c, c->capacity - k);
      o.id = (hft_u32)(c->capacity - k);
      o.symbol = (hft_u32)s->symbol;
      o.side = k;
      o.qty = signal.qty;
      o.price = k ? signal.ask : signal.bid;
      o.ts = s->time;
      o.agent = 5;
      o.ioc = 0;
      hft_emit(s, c, o, t, naive, delay, 0);
    }
  } else if (signal.direction) {
    o.id = (hft_u32)(c->capacity + 1 + s->event);
    o.symbol = (hft_u32)s->symbol;
    o.side = signal.direction < 0;
    o.qty = signal.qty;
    o.price = signal.price;
    o.ts = s->time;
    o.agent = 5;
    o.ioc = 1;
    hft_emit(s, c, o, t, naive, delay, 0);
  }
  if (c->strategy == HFT_EXECUTION && s->position.qty > old_qty)
    s->execution_done += s->position.qty - old_qty;
  mark = hft_mid(hft_top(&s->book, c, naive));
  pnl = s->position.cash + (hft_i64)s->position.qty * mark;
  if (pnl > s->peak)
    s->peak = pnl;
  if (s->peak - pnl > s->drawdown)
    s->drawdown = s->peak - pnl;
  if (s->event % (hft_u64)c->sample_every == 0) {
    int at = s->trace_head;
    HftCandle candle = s->active_candle;
    if (!candle.volume) {
      candle.open = candle.high = candle.low = candle.close = mark;
      candle.ts = s->time;
    }
    s->active_candle.volume = 0;
    s->candles[at] = candle;
    s->equity[at] = pnl;
    s->inventory[at] = s->position.qty;
    s->trace_head = (at + 1) % HFT_TRACE;
    if (s->trace_count < HFT_TRACE)
      ++s->trace_count;
  }
  ++s->event;
}
static inline void hft_group_inline(HFT_GLOBAL HftSymbol *states, int first,
                                    int count, HFT_CONST HftConfig *c,
                                    hft_u64 steps, int naive, hft_u64 delay) {
  hft_u64 e;
  for (e = 0; e < steps; ++e) {
    hft_i64 spread;
    hft_background(&states[first], c, naive);
    if (first + 1 < count)
      hft_background(&states[first + 1], c, naive);
    spread = 0;
    if (first + 1 < count && c->strategy == HFT_PAIRS) {
      int i, n = (int)hft_min(c->slow, (double)states[first].features.count);
      double sx = 0, sy = 0, sxx = 0, sxy = 0, syy = 0, beta, varx, vary, cov;
      for (i = 0; i < n; ++i) {
        hft_u64 at =
            (states[first].features.count - 1 - (hft_u64)i) % HFT_WINDOW;
        double x = states[first + 1].features.prices[at],
               y = states[first].features.prices[at];
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
        syy += y * y;
      }
      varx = sxx - sx * sx / n;
      vary = syy - sy * sy / n;
      cov = sxy - sx * sy / n;
      beta = varx > 1e-8 ? cov / varx : 1;
      states[first].pair_beta = states[first + 1].pair_beta =
          (float)hft_max(.1, hft_min(10, beta));
      if (beta > 0 && n > 2) {
        double residual =
            (double)hft_mid(hft_top(&states[first].book, c, naive)) - sy / n -
            beta *
                ((double)hft_mid(hft_top(&states[first + 1].book, c, naive)) -
                 sx / n);
        spread =
            (hft_i64)(100 * residual /
                      hft_max(1, hft_sqrt(hft_max(0, vary - beta * cov) / n)));
      }
    }
    hft_decision(&states[first], c, spread, naive, delay);
    if (first + 1 < count)
      hft_decision(&states[first + 1], c, spread, naive, delay);
  }
}
#endif
