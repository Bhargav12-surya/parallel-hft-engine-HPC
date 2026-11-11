#define _POSIX_C_SOURCE 200809L
#include "mini_hft.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int parse_event(const char *line, QuoteEvent *e) {
    return sscanf(line, "%d,%d,%lf,%lf,%d,%d",
                  &e->step, &e->symbol, &e->bid, &e->ask,
                  &e->bid_qty, &e->ask_qty) == 6;
}

int load_market_data(const char *path, MarketData *market) {
    memset(market, 0, sizeof(*market));

    FILE *fp = fopen(path, "r");
    if (!fp) {
        perror("fopen");
        return 0;
    }

    char line[256];
    if (!fgets(line, sizeof(line), fp)) {
        fclose(fp);
        return 0;
    }

    int max_symbol = -1;
    size_t total = 0;
    QuoteEvent e;

    while (fgets(line, sizeof(line), fp)) {
        if (!parse_event(line, &e) || e.symbol < 0) continue;
        if (e.symbol > max_symbol) max_symbol = e.symbol;
        total++;
    }

    if (max_symbol < 0 || total == 0) {
        fclose(fp);
        return 0;
    }

    int symbol_count = max_symbol + 1;
    size_t *counts = calloc((size_t)symbol_count, sizeof(*counts));
    if (!counts) {
        fclose(fp);
        return 0;
    }

    rewind(fp);
    fgets(line, sizeof(line), fp);
    while (fgets(line, sizeof(line), fp)) {
        if (parse_event(line, &e) && e.symbol >= 0 && e.symbol < symbol_count)
            counts[e.symbol]++;
    }

    SymbolData *symbols = calloc((size_t)symbol_count, sizeof(*symbols));
    size_t *used = calloc((size_t)symbol_count, sizeof(*used));
    if (!symbols || !used) {
        free(symbols);
        free(used);
        free(counts);
        fclose(fp);
        return 0;
    }

    for (int s = 0; s < symbol_count; ++s) {
        symbols[s].count = counts[s];
        symbols[s].events = calloc(counts[s], sizeof(QuoteEvent));
        if (counts[s] && !symbols[s].events) {
            for (int j = 0; j < s; ++j) free(symbols[j].events);
            free(symbols);
            free(used);
            free(counts);
            fclose(fp);
            return 0;
        }
    }

    rewind(fp);
    fgets(line, sizeof(line), fp);
    while (fgets(line, sizeof(line), fp)) {
        if (!parse_event(line, &e)) continue;
        if (e.symbol < 0 || e.symbol >= symbol_count) continue;
        symbols[e.symbol].events[used[e.symbol]++] = e;
    }

    fclose(fp);
    free(used);
    free(counts);

    market->symbols = symbols;
    market->symbol_count = symbol_count;
    market->total_events = total;
    return 1;
}

void free_market_data(MarketData *market) {
    if (!market || !market->symbols) return;
    for (int s = 0; s < market->symbol_count; ++s)
        free(market->symbols[s].events);
    free(market->symbols);
    memset(market, 0, sizeof(*market));
}

void update_order_book(OrderBook *book, const QuoteEvent *event) {
    book->best_bid = event->bid;
    book->best_ask = event->ask;
    book->bid_qty = event->bid_qty;
    book->ask_qty = event->ask_qty;
}

SimResult simulate_symbol(const SymbolData *data) {
    OrderBook book = {0};
    SimResult result = {0};
    double previous_mid = 0.0;
    const int max_position = 10;

    for (size_t i = 0; i < data->count; ++i) {
        update_order_book(&book, &data->events[i]);
        double mid = 0.5 * (book.best_bid + book.best_ask);

        if (previous_mid > 0.0) {
            if (mid > previous_mid && result.final_position < max_position) {
                result.cash -= book.best_ask;   /* buy 1 at the ask */
                result.final_position += 1;
                result.trades += 1;
            } else if (mid < previous_mid && result.final_position > -max_position) {
                result.cash += book.best_bid;   /* sell 1 at the bid */
                result.final_position -= 1;
                result.trades += 1;
            }
        }

        previous_mid = mid;
        result.last_mid = mid;
    }

    result.pnl = result.cash + result.final_position * result.last_mid;
    return result;
}

double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

void print_summary(const char *name, int workers, size_t events,
                   long long trades, int position, double pnl, double seconds) {
    double throughput = seconds > 0.0 ? (double)events / seconds : 0.0;
    printf("backend=%s\n", name);
    printf("workers=%d\n", workers);
    printf("events=%zu\n", events);
    printf("trades=%lld\n", trades);
    printf("final_position=%d\n", position);
    printf("pnl=%.6f\n", pnl);
    printf("time_seconds=%.6f\n", seconds);
    printf("throughput_events_per_second=%.0f\n", throughput);
}
