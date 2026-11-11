#include "mini_hft.h"
#include <stdio.h>

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "data/market.csv";
    MarketData market;

    if (!load_market_data(path, &market)) {
        fprintf(stderr, "Could not load %s\n", path);
        return 1;
    }

    long long trades = 0;
    int position = 0;
    double pnl = 0.0;

    double start = now_seconds();
    for (int s = 0; s < market.symbol_count; ++s) {
        SimResult r = simulate_symbol(&market.symbols[s]);
        trades += r.trades;
        position += r.final_position;
        pnl += r.pnl;
    }
    double seconds = now_seconds() - start;

    print_summary("sequential", 1, market.total_events,
                  trades, position, pnl, seconds);
    free_market_data(&market);
    return 0;
}
