#include "mini_hft.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    const MarketData *market;
    int thread_id;
    int thread_count;
    long long trades;
    int position;
    double pnl;
} ThreadWork;

static void *worker(void *arg) {
    ThreadWork *w = (ThreadWork *)arg;

    for (int s = w->thread_id; s < w->market->symbol_count; s += w->thread_count) {
        SimResult r = simulate_symbol(&w->market->symbols[s]);
        w->trades += r.trades;
        w->position += r.final_position;
        w->pnl += r.pnl;
    }
    return NULL;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "data/market.csv";
    int thread_count = argc > 2 ? atoi(argv[2]) : 2;
    if (thread_count < 1) thread_count = 1;

    MarketData market;
    if (!load_market_data(path, &market)) {
        fprintf(stderr, "Could not load %s\n", path);
        return 1;
    }

    pthread_t *threads = calloc((size_t)thread_count, sizeof(*threads));
    ThreadWork *work = calloc((size_t)thread_count, sizeof(*work));
    if (!threads || !work) return 1;

    double start = now_seconds();

    for (int t = 0; t < thread_count; ++t) {
        work[t].market = &market;
        work[t].thread_id = t;
        work[t].thread_count = thread_count;
        pthread_create(&threads[t], NULL, worker, &work[t]);
    }

    long long trades = 0;
    int position = 0;
    double pnl = 0.0;

    for (int t = 0; t < thread_count; ++t) {
        pthread_join(threads[t], NULL);
        trades += work[t].trades;
        position += work[t].position;
        pnl += work[t].pnl;
    }

    double seconds = now_seconds() - start;
    print_summary("pthreads", thread_count, market.total_events,
                  trades, position, pnl, seconds);

    free(threads);
    free(work);
    free_market_data(&market);
    return 0;
}
