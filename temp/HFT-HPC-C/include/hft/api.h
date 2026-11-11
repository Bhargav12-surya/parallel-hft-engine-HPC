/* Purpose: public C engine, results, AI and analytics API. Units 1-4.
 * Ownership: caller frees results with hft_result_free; errors use return
 * codes. */
#ifndef HFT_API_H
#define HFT_API_H
#include "platform.h"
#include "types.h"
#define HFT_ERROR 1024
#define HFT_LAT_BINS 8192
typedef struct {
  hft_u64 counts[HFT_LAT_BINS], count, maximum;
  double sum, squares;
} HftHistogram;
typedef struct {
  int available, workers;
  char reason[256], name[128];
} HftCapability;
typedef struct {
  hft_u64 event;
  hft_i64 price, pnl;
  int inventory, regime, selected, pad;
} HftPoint;
typedef struct {
  HftConfig config;
  int cancelled, verified, workers, rank_count;
  double seconds, throughput, speedup_naive, speedup_serial, efficiency,
      karp_flatt;
  double h2d_seconds, kernel_seconds, d2h_seconds, communication_seconds,
      wait_seconds;
  double latency_overhead_ns, latency_clock_resolution_ns, batch_latency_ns,
      stage_seconds[8];
  double sharpe, sortino, max_drawdown, return_fraction, win_rate,
      profit_factor, slippage;
  hft_u64 events, orders, fills, rejected, hash, market_hash;
  hft_i64 pnl, fees, gross, latency_cost;
  HftHistogram latency;
  HftSymbol *states;
  HftPoint *points;
  int point_count;
  double *worker_seconds;
  int worker_count;
  char hardware[1024], timestamp[64], device[256], error[HFT_ERROR];
} HftResult;
typedef struct {
  HftTop top;
  HftCandle candles[HFT_TRACE];
  hft_i64 equity[HFT_TRACE], inventory[HFT_TRACE];
  HftFill last_fill;
  hft_i64 bids[20], asks[20];
  int bid_qty[20], ask_qty[20];
  double prices[64];
  int count, head, symbol, strategy, regime, selected, halted;
  hft_u64 events, orders, fills;
  hft_i64 pnl;
  double progress, events_second;
  char status[256];
} HftSnapshot;
typedef void (*HftProgress)(const HftSnapshot *, void *);
typedef struct {
  HftAtomic middle;
  int back, front;
  HftSnapshot slots[3];
} HftTripleBuffer;
void hft_snapshot_publish(HftTripleBuffer *, const HftSnapshot *);
int hft_snapshot_consume(HftTripleBuffer *, HftSnapshot *);
extern const char *const hft_engine_names[HFT_ENGINES];
extern const char *const hft_strategy_names[HFT_STRATEGIES];
extern const char *const hft_regime_names[7];
void hft_config_default(HftConfig *);
int hft_config_validate(const HftConfig *, char *);
int hft_config_load(HftConfig *, const char *, char *);
int hft_config_save(const HftConfig *, const char *, char *);
void hft_probe(int, HftCapability *);
int hft_run(const HftConfig *, HftResult *, HftAtomic *, HftProgress, void *);
int hft_run_cpu(const HftConfig *, HftResult *, HftAtomic *, HftProgress,
                void *);
int hft_run_opencl(const HftConfig *, HftResult *, HftAtomic *, HftProgress,
                   void *);
int hft_run_mpi_process(const HftConfig *, HftResult *, HftAtomic *,
                        HftProgress, void *, const char *, int);
void hft_opencl_probe(HftCapability *);
void hft_nvidia_probe(HftCapability *);
void hft_result_free(HftResult *);
int hft_result_export(const HftResult *, const char *, char *);
int hft_result_import(HftResult *, const char *, char *);
int hft_result_equivalent(const HftResult *, const HftResult *);
void hft_hist_record(HftHistogram *, hft_u64);
hft_u64 hft_hist_percentile(const HftHistogram *, double);
void hft_hist_merge(HftHistogram *, const HftHistogram *);
void hft_result_finalize(HftResult *);
void hft_make_snapshot(const HftResult *, int, HftSnapshot *);
int hft_initialize(const HftConfig *, HftResult *);
void hft_step_group(HFT_GLOBAL HftSymbol *, int, HFT_CONST HftConfig *, hft_u64,
                    int, hft_u64);
int hft_covariance(const double *, int, int, double *, int, int);
int hft_cholesky(const double *, int, double *);
int hft_cg(const double *, const double *, int, double *, int, double, int,
           int *, double *);
int hft_kmeans(const float *, int, int, int, int, float *, int *, int);
int hft_monte_carlo(const double *, const double *, int, int, hft_u64, double *,
                    double *, int);
int hft_risk_from_normals(const double *, const double *, const float *, int,
                          int, double *, double *);
int hft_nvidia_risk(const float *, int, int, float *, float *, int, hft_u64,
                    double *, char *);
int hft_opencl_matrix(const float *, int, int, float *, int, double *, char *);
int hft_opencl_infer(const HftModel *, const float *, int, float *, double *,
                     char *);
int hft_opencl_train(HftModel *, const float *, const int *, int, int, double *,
                     char *);
int hft_opencl_qtrain(HftModel *, int, int, hft_u64, double *, char *);
typedef struct {
  float *features;
  int *labels;
  int count, train_count;
} HftDataset;
int hft_dataset_make(const HftConfig *, int, HftDataset *, char *);
void hft_dataset_free(HftDataset *);
void hft_model_init(HftModel *, hft_u64);
void hft_mlp_predict(const HftModel *, const float *, float *);
double hft_mlp_gradient(const HftModel *, const float *, int, float *);
int hft_mlp_train(HftModel *, const HftDataset *, int, int, double *, double *,
                  HftAtomic *);
int hft_q_train(HftModel *, const HftConfig *, int, int, int, double *,
                HftAtomic *);
int hft_model_save(const HftModel *, const char *, char *);
int hft_model_load(HftModel *, const char *, char *);
int hft_openmp_sync(int, hft_u64, double *);
int hft_sync_benchmark(int, hft_u64, double *, char *);
int hft_write_png(const char *, const unsigned char *, int, int, char *);
int hft_svg_report(const HftResult *, const char *, char *);
int hft_source_export(const char *, char *);
int hft_bench_main(int, char **);
#endif
