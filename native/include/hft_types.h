#ifndef HFT_TYPES_H
#define HFT_TYPES_H
#include <stddef.h>
#include <stdint.h>

#define HFT_SCHEMA_VERSION 1
#define HFT_FEATURES 6
#define HFT_HIDDEN_MAX 128
#define HFT_ERROR_SIZE 512
enum HftEventType { HFT_LIMIT, HFT_MARKET, HFT_CANCEL, HFT_QUOTE, HFT_MODIFY };
typedef struct {
    uint64_t id, timestamp_ns, order_id, reference_id;
    int64_t price;
    int32_t quantity, instrument;
    int type, side;
} HftEvent;
typedef struct {
    HftEvent *events;
    size_t count;
    size_t instruments;
} HftDataset;
typedef struct {
    int hidden;
    double mean[HFT_FEATURES], scale[HFT_FEATURES];
    double w1[HFT_HIDDEN_MAX * HFT_FEATURES], b1[HFT_HIDDEN_MAX];
    double w2[3 * HFT_HIDDEN_MAX], b2[3];
} HftModel;
typedef struct {
    const char *input, *output, *engine, *cpu_mode, *strategy, *model_path, *components;
    int threads, position_limit, max_order, slippage_ticks;
    uint64_t latency_ns;
    double capital, fee_bps, max_loss, tick_value;
    int quiet;
} HftConfig;
typedef struct {
    double cash, cost, realized, fees, slippage;
    int64_t position;
    uint64_t trades, wins, closed;
} HftPortfolio;
typedef struct {
    double prices[17], returns[16];
    uint64_t count;
    double last;
} HftFeatures;
typedef struct {
    double pnl, equity, fees, realized, unrealized, drawdown, gross_exposure, slippage;
    uint64_t events, trades, orders, rejected, cancels, checksum, simulated_time_ns;
    int64_t position;
    double compute_seconds;
} HftInstrumentResult;
typedef struct {
    HftInstrumentResult *instruments;
    size_t instrument_count;
    double wall_seconds, load_seconds, compute_seconds, communication_seconds;
    double h2d_seconds, kernel_seconds, d2h_seconds, feature_seconds, match_seconds;
    double latency_us[3];
    double portfolio_drawdown, equity_curve[256];
    uint64_t equity_timestamps[256];
    size_t equity_samples;
    size_t peak_rss_bytes;
    long voluntary_switches, involuntary_switches;
    int workers, ranks, affinity_applied;
    uint64_t bytes_transferred;
    char error[HFT_ERROR_SIZE];
} HftResult;
#endif
