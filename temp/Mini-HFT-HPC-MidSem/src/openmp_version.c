#include "mini_hft.h"

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "data/market.csv";
    int thread_count = argc > 2 ? atoi(argv[2]) : 2;
    if (thread_count < 1) thread_count = 1;

    MarketData market;
    if (!load_market_data(path, &market)) {
        fprintf(stderr, "Could not load %s\n", path);
        return 1;
    }

    omp_set_num_threads(thread_count);

    long long trades = 0;
    int position = 0;
    double pnl = 0.0;

    double start = now_seconds();

    #pragma omp parallel for reduction(+:trades, position, pnl) schedule(static)
    for (int s = 0; s < market.symbol_count; ++s) {
        SimResult r = simulate_symbol(&market.symbols[s]);
        trades += r.trades;
        position += r.final_position;
        pnl += r.pnl;
    }

    double seconds = now_seconds() - start;
    print_summary("openmp", thread_count, market.total_events,
                  trades, position, pnl, seconds);

    free_market_data(&market);
    return 0;
}
