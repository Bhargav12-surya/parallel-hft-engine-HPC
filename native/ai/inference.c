#include "hft_engine.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
void hft_features_update(HftFeatures *state, double price, int side, int quantity, double out[HFT_FEATURES]) {
    if (price <= 0) price = state->last > 0 ? state->last : 10000;
    uint64_t n = state->count;
    double ret = state->last > 0 ? (price - state->last) / state->last : 0;
    out[0] = ret;
    out[1] = n >= 4 ? (price - state->prices[(n - 4) % 17]) / state->prices[(n - 4) % 17] : 0;
    out[2] = n >= 16 ? (price - state->prices[(n - 16) % 17]) / state->prices[(n - 16) % 17] : 0;
    state->returns[n % 16] = ret;
    size_t samples = n + 1 < 16 ? (size_t)n + 1 : 16;
    double mean = 0, variance = 0;
    for (size_t j = 0; j < samples; ++j) mean += state->returns[j];
    mean /= (double)samples;
    for (size_t j = 0; j < samples; ++j) { double d = state->returns[j] - mean; variance += d * d; }
    out[3] = sqrt(variance / (double)samples);
    out[4] = (side ? -1.0 : 1.0) * quantity / (quantity + 100.0);
    out[5] = 2.0 / price;
    state->prices[n % 17] = price; state->last = price; ++state->count;
}
static int read_values(FILE *file, double *values, size_t count) {
    for (size_t i = 0; i < count; ++i) if (fscanf(file,"%lf",&values[i]) != 1 || !isfinite(values[i]) || fabs(values[i]) > 1e15) return 0;
    return 1;
}
int hft_model_load(const char *path, HftModel *m, char *error) {
    memset(m,0,sizeof(*m));
    FILE *file = fopen(path,"rb");
    if (!file) { snprintf(error,HFT_ERROR_SIZE,"Cannot open native model"); return 0; }
    char magic[32]; int version, inputs, outputs;
    if (fscanf(file,"%31s%d%d%d%d",magic,&version,&inputs,&m->hidden,&outputs) != 5 ||
        strcmp(magic,"HFTMODEL") || version != 1 || inputs != HFT_FEATURES || outputs != 3 ||
        m->hidden < 0 || m->hidden > HFT_HIDDEN_MAX) goto invalid;
    if (!read_values(file,m->mean,HFT_FEATURES) || !read_values(file,m->scale,HFT_FEATURES)) goto invalid;
    for (int j = 0; j < HFT_FEATURES; ++j) if (m->scale[j] < 1e-15) goto invalid;
    if (m->hidden && (!read_values(file,m->w1,(size_t)m->hidden * HFT_FEATURES) || !read_values(file,m->b1,(size_t)m->hidden))) goto invalid;
    if (!read_values(file,m->w2,3 * (size_t)(m->hidden ? m->hidden : HFT_FEATURES)) || !read_values(file,m->b2,3)) goto invalid;
    { char extra; if (fscanf(file," %c",&extra) == 1) goto invalid; }
    fclose(file); return 1;
invalid:
    snprintf(error,HFT_ERROR_SIZE,"Invalid native model format, dimensions, or numerical values"); fclose(file); return 0;
}
void hft_model_predict(const HftModel *m, const double *features, double output[3]) {
    double x[HFT_FEATURES], hidden[HFT_HIDDEN_MAX];
    for (int j = 0; j < HFT_FEATURES; ++j) x[j] = (features[j] - m->mean[j]) / m->scale[j];
    for (int i = 0; i < m->hidden; ++i) {
        double value = m->b1[i];
        for (int j = 0; j < HFT_FEATURES; ++j) value += m->w1[i * HFT_FEATURES + j] * x[j];
        hidden[i] = value > 0 ? value : 0;
    }
    int width = m->hidden ? m->hidden : HFT_FEATURES;
    const double *input = m->hidden ? hidden : x;
    double maximum = -1e300;
    for (int i = 0; i < 3; ++i) {
        double value = m->b2[i];
        for (int j = 0; j < width; ++j) value += m->w2[i * width + j] * input[j];
        output[i] = value; if (value > maximum) maximum = value;
    }
    double sum = 0;
    for (int i = 0; i < 3; ++i) { output[i] = exp(output[i] - maximum); sum += output[i]; }
    for (int i = 0; i < 3; ++i) output[i] /= sum;
}
void hft_portfolio_fill(HftPortfolio *p, int side, int64_t price, int quantity, double fee_bps, double tick_value) {
    int64_t delta = side ? -quantity : quantity, old = p->position;
    double notional = (double)price * quantity * tick_value;
    double fee = notional * fee_bps * .0001;
    p->cash -= (side ? -notional : notional) + fee; p->fees += fee;
    if (!old || (old > 0) == (delta > 0)) p->cost += notional;
    else {
        int64_t closed = (old < 0 ? -old : old) < quantity ? (old < 0 ? -old : old) : quantity;
        double cost = p->cost * (double)closed / (double)(old < 0 ? -old : old);
        double profit = (old > 0 ? 1 : -1) * ((double)closed * price * tick_value - cost);
        p->realized += profit; p->cost -= cost; ++p->closed; if (profit > 0) ++p->wins;
        if ((int64_t)quantity > (old < 0 ? -old : old)) p->cost = ((double)quantity - (double)(old < 0 ? -old : old)) * price * tick_value;
    }
    p->position += delta; ++p->trades;
}
