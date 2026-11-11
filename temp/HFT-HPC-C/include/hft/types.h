/* Purpose: shared C and OpenCL C contracts. Units 1/3: portable layout.
 * Complexity: fixed storage per symbol; no pointers in device-visible records.
 */
#ifndef HFT_TYPES_H
#define HFT_TYPES_H
#ifdef __OPENCL_VERSION__
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL FP_CONTRACT OFF
typedef long hft_i64;
typedef ulong hft_u64;
typedef uint hft_u32;
#define HFT_GLOBAL __global
#define HFT_CONST __global const
#else
#include <stddef.h>
#include <stdint.h>
typedef int64_t hft_i64;
typedef uint64_t hft_u64;
typedef uint32_t hft_u32;
#define HFT_GLOBAL
#define HFT_CONST const
#endif
#define HFT_LEVELS 2048
#define HFT_ORDERS 1024
#define HFT_WINDOW 128
#define HFT_TRACE 256
#define HFT_FEATURES 32
#define HFT_WEIGHTS 4291
#define HFT_QSTATES 81
#define HFT_ACTIONS 9
#define HFT_STRATEGIES 13
#define HFT_ENGINES 8
enum {
  HFT_NAIVE,
  HFT_SERIAL,
  HFT_THREADS,
  HFT_OPENMP,
  HFT_MPI,
  HFT_HYBRID,
  HFT_OPENCL,
  HFT_NVIDIA
};
enum {
  HFT_SMA,
  HFT_EMA,
  HFT_REVERSION,
  HFT_BREAKOUT,
  HFT_RSI,
  HFT_MACD,
  HFT_IMBALANCE,
  HFT_MAKER,
  HFT_PAIRS,
  HFT_EXECUTION,
  HFT_MLP,
  HFT_QLEARNING,
  HFT_SELECTOR
};
enum {
  HFT_CALM,
  HFT_BULL,
  HFT_BEAR,
  HFT_SIDEWAYS,
  HFT_VOLATILE,
  HFT_CRASH,
  HFT_NEWS
};
enum { HFT_ADD, HFT_CANCEL, HFT_MARKET };
typedef struct {
  hft_u64 ts;
  hft_i64 price;
  hft_u32 id, symbol;
  int qty;
  unsigned char side, type, agent, flags;
} HftEvent;
typedef struct {
  hft_u32 id, symbol;
  hft_i64 price;
  hft_u64 ts;
  int qty, side, agent, ioc;
} HftOrder;
typedef struct {
  hft_u64 ts;
  hft_i64 price;
  hft_u32 id, symbol;
  int qty, side, maker, pad;
} HftFill;
typedef struct {
  HftOrder order;
  hft_u64 sequence;
  int next, previous, level, live;
} HftNode;
typedef struct {
  int head, tail, volume, pad;
} HftLevel;
typedef struct {
  hft_i64 bid, ask, last;
  int bid_qty, ask_qty;
  hft_u64 ts;
} HftTop;
typedef struct {
  hft_i64 cash, cost, realized, fees;
  int qty, pad;
} HftPosition;
typedef struct {
  float weights[HFT_WEIGHTS];
  float q[HFT_QSTATES * HFT_ACTIONS];
  float centroids[4 * 4];
  int trained, q_trained, version, pad;
} HftModel;
typedef struct {
  hft_u64 seed, events;
  int symbols, depth, capacity, strategy, engine, threads, batch, warmup,
      repetitions, fast, slow;
  int quantity, max_position, max_order, max_orders_second, collar, regime,
      auto_regime;
  int schedule, chunk, block_size, streams, pinned, topology, partition, mode;
  int parent_qty, parent_horizon, sample_every, train_epochs, train_samples;
  int weights[5];
  hft_i64 initial_price, max_gross, max_loss, initial_capital, fee;
  double tick_size, volatility, drift, hawkes_mu, hawkes_alpha, hawkes_beta,
      jump_rate;
  double jump_mean, jump_std, cancel_probability, z_enter, z_exit, imbalance;
  double latency_amplification, network_ns, network_jitter_ns, cpu_hourly,
      gpu_hourly;
    char real_data_path[256]; // path to CSV/JSON real market data
  HftModel model;
} HftConfig;
typedef struct {
  HftNode nodes[HFT_ORDERS + 1];
  HftLevel levels[2 * HFT_LEVELS];
  hft_u64 occupied[2 * (HFT_LEVELS / 64)], sequence;
  hft_i64 low, last;
  hft_u64 time;
  int live, rejections;
} HftBook;
typedef struct {
  hft_i64 prices[HFT_WINDOW], volume[HFT_WINDOW], sum_fast, sum_slow, square;
  hft_i64 ema_fast, ema_slow, macd_signal, gain, loss;
  hft_u64 count;
  float features[HFT_FEATURES];
  double norm_mean[HFT_FEATURES], norm_m2[HFT_FEATURES];
  hft_i64 high, low, vwap_num, vwap_den, last_mid;
} HftFeatures;
typedef struct {
  hft_u64 ts;
  hft_i64 open, high, low, close, volume;
} HftCandle;
typedef struct {
  HftBook book;
  HftFeatures features;
  HftPosition position;
  hft_u64 event, time, hash, market_hash, orders, fills, filled_qty, rejected,
      cancels;
  hft_u64 rate_last, rate_credit, next_decision;
  double fundamental, excitation;
  hft_i64 previous_mid, agent_inventory[5];
  int symbol, regime, halted, rate_initialized, trace_count, trace_head;
  int selected, detected, execution_done, pad;
  double bandit_reward[4 * 3];
  int bandit_count[4 * 3];
  hft_i64 bandit_last_pnl, peak, drawdown, slippage_sum, turnover;
  HftOrder delayed[32];
  int delayed_count, delayed_dropped;
  hft_i64 traded_qty, traded_value, signed_volume;
  float pair_beta;
  int reserved;
  HftCandle active_candle;
  HftFill last_fill;
  HftCandle candles[HFT_TRACE];
  hft_i64 equity[HFT_TRACE], inventory[HFT_TRACE];
} HftSymbol;
typedef struct {
  int direction, qty, is_quote, pad;
  hft_i64 price, bid, ask;
  float confidence;
  int selected;
} HftSignal;
#ifndef __OPENCL_VERSION__
_Static_assert(sizeof(HftEvent) == 32, "HftEvent must remain 32 bytes");
_Static_assert(sizeof(hft_i64) == 8, "64-bit prices required");
#endif
#endif
