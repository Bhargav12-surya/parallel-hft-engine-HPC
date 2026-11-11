#ifndef HFT_ENGINE_H
#define HFT_ENGINE_H
#include "hft_types.h"
#ifdef __cplusplus
extern "C" {
#endif
double hft_now(void);
int hft_logical_cpus(void);
int hft_pin_current_thread(int single_core);
int hft_dataset_load(const char *path, HftDataset *data, char *error);
void hft_dataset_free(HftDataset *data);
void hft_features_update(HftFeatures *state, double price, int side, int quantity,
                         double output[HFT_FEATURES]);
int hft_model_load(const char *path, HftModel *model, char *error);
void hft_model_predict(const HftModel *model, const double *features, double output[3]);
void hft_portfolio_fill(HftPortfolio *portfolio, int side, int64_t price, int quantity,
                         double fee_bps, double tick_value);
int hft_run(const HftConfig *config, const HftDataset *data,
             const HftModel *model, HftResult *result);
int hft_result_write(const char *path, const HftConfig *config,
                      const HftResult *result);
void hft_capabilities(void);
int hft_cuda_available(char *name, size_t capacity);
int hft_cuda_features(const HftEvent *events, size_t count, int instruments,
                       const HftModel *model, double *features, double *probabilities,
                       HftResult *timing);
#ifdef __cplusplus
}
#endif
#endif
