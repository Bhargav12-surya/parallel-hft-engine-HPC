/* real_dataset.c – load CSV/JSON market data into HftDataset */
#include "hft/api.h"
#include "hft/real_dataset.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE 4096

int hft_real_dataset_load(const char *path, HftDataset *out, char *error)
{
    FILE *f = fopen(path, "r");
    if (!f) {
        snprintf(error, HFT_ERROR, "Cannot open %s", path);
        return 0;
    }
    size_t cap = 1024;
    out->features = calloc(cap, 32 * sizeof(float));
    out->labels = calloc(cap, sizeof(int));
    if (!out->features || !out->labels) {
        strcpy(error, "Allocation failed");
        fclose(f);
        return 0;
    }
    char line[MAX_LINE];
    size_t n = 0;
    if (!fgets(line, sizeof(line), f)) { strcpy(error, "Empty CSV"); goto fail; }
    double price_prev = 0;
    while (fgets(line, sizeof(line), f)) {
        if (n == cap) {
            cap *= 2;
            out->features = realloc(out->features, cap * 32 * sizeof(float));
            out->labels = realloc(out->labels, cap * sizeof(int));
            if (!out->features || !out->labels) { strcpy(error, "Reallocation failed"); goto fail; }
        }
        char *tok = strtok(line, ",");
        int col = 0;
        float feats[32] = {0};
        double price_now = 0;
        while (tok && col < 34) {
            if (col < 32) {
                feats[col] = (float)atof(tok);
            } else if (col == 32) {
                price_now = atof(tok);
            } else if (col == 33) {
                out->labels[n] = atoi(tok);
            }
            tok = strtok(NULL, ",");
            ++col;
        }
        memcpy(out->features + n * 32, feats, 32 * sizeof(float));
        if (col <= 33) {
            double diff = price_now - price_prev;
            out->labels[n] = diff > 1 ? 2 : (diff < -1 ? 0 : 1);
        }
        price_prev = price_now;
        ++n;
    }
    out->count = (int)n;
    out->train_count = n * 3 / 4;
    fclose(f);
    return 1;
fail:
    fclose(f);
    free(out->features);
    free(out->labels);
    out->features = out->labels = NULL;
    return 0;
}

