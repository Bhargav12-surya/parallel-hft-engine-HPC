#ifndef MINI_HFT_H
#define MINI_HFT_H

#include <stddef.h>

typedef struct {
    int step;
    int symbol;
    double bid;
    double ask;
    int bid_qty;
    int ask_qty;
} QuoteEvent;

typedef struct {
    double best_bid;
    double best_ask;
    int bid_qty;
    int ask_qty;
} OrderBook;

typedef struct {
    QuoteEvent *events;
    size_t count;
} SymbolData;

typedef struct {
    long long trades;
    int final_position;
    double cash;
    double last_mid;
    double pnl;
} SimResult;

typedef struct {
    SymbolData *symbols;
    int symbol_count;
    size_t total_events;
} MarketData;

int load_market_data(const char *path, MarketData *market);
void free_market_data(MarketData *market);
void update_order_book(OrderBook *book, const QuoteEvent *event);
SimResult simulate_symbol(const SymbolData *data);
double now_seconds(void);
void print_summary(const char *name, int workers, size_t events,
                   long long trades, int position, double pnl, double seconds);

#endif
