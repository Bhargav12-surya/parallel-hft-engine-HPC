/* Unit 4: temporal dataset, real backprop, validation and persisted C weights.
 */
#include "hft/api.h"
#include <stdio.h>
int main(void) {
  HftConfig c;
  HftDataset d;
  char error[HFT_ERROR] = {0};
  double loss, accuracy;
  hft_config_default(&c);
  if (!hft_dataset_make(&c, 512, &d, error))
    return 1;
  int ok = hft_mlp_train(&c.model, &d, 3, 1, &loss, &accuracy, NULL);
  if (ok)
    ok = hft_model_save(&c.model, "example-model.bin", error);
  printf("loss %.4f accuracy %.4f %s\n", loss, accuracy, error);
  hft_dataset_free(&d);
  return ok ? 0 : 1;
}
