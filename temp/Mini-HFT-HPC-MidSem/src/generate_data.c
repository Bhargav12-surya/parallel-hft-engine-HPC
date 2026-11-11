#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static uint32_t rng_state = 123456789u;

static uint32_t next_u32(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

int main(int argc, char **argv) {
    int symbols = argc > 1 ? atoi(argv[1]) : 4;
    int events_per_symbol = argc > 2 ? atoi(argv[2]) : 10000;
    const char *path = argc > 3 ? argv[3] : "data/market.csv";

    if (symbols < 1 || events_per_symbol < 2) {
        fprintf(stderr, "Usage: %s [symbols>=1] [events_per_symbol>=2] [output.csv]\n", argv[0]);
        return 1;
    }

    FILE *fp = fopen(path, "w");
    if (!fp) {
        perror("fopen");
        return 1;
    }

    double *mid = malloc((size_t)symbols * sizeof(*mid));
    if (!mid) {
        fclose(fp);
        return 1;
    }

    for (int s = 0; s < symbols; ++s) mid[s] = 100.0 + 5.0 * s;

    fprintf(fp, "step,symbol,bid,ask,bid_qty,ask_qty\n");

    for (int step = 0; step < events_per_symbol; ++step) {
        for (int s = 0; s < symbols; ++s) {
            uint32_t r = next_u32() % 100u;
            if (r < 35u) mid[s] -= 0.01;
            else if (r >= 65u) mid[s] += 0.01;

            if (mid[s] < 1.0) mid[s] = 1.0;

            double half_spread = (next_u32() % 5u == 0u) ? 0.015 : 0.005;
            double bid = mid[s] - half_spread;
            double ask = mid[s] + half_spread;
            int bid_qty = 1 + (int)(next_u32() % 20u);
            int ask_qty = 1 + (int)(next_u32() % 20u);

            fprintf(fp, "%d,%d,%.3f,%.3f,%d,%d\n",
                    step, s, bid, ask, bid_qty, ask_qty);
        }
    }

    free(mid);
    fclose(fp);

    printf("Generated %d symbols x %d events = %d rows in %s\n",
           symbols, events_per_symbol, symbols * events_per_symbol, path);
    return 0;
}
