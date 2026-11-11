// real_dataset.h – loader for CSV/JSON market data into HftDataset
#ifndef HFT_REAL_DATASET_H
#define HFT_REAL_DATASET_H

#include "hft/api.h"

/* Load a CSV file containing real market data into an HftDataset.
 * Expected columns (in order, comma‑separated):
 *   timestamp,mid_price,spread,volume,feat0,feat1,...,feat31[,label]
 * The optional label column will be computed from price deltas if omitted.
 *
 * Parameters:
 *   path   – path to the CSV file.
 *   out    – pointer to an HftDataset that will be allocated and filled.
 *   error  – buffer of size HFT_ERROR for error messages.
 * Returns 1 on success, 0 on failure.
 */
int hft_real_dataset_load(const char *path, HftDataset *out, char *error);

#endif // HFT_REAL_DATASET_H
