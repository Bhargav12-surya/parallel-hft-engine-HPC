#include "mini_hft.h"

#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const char *path = argc > 1 ? argv[1] : "data/market.csv";
    MarketData market;

    if (!load_market_data(path, &market)) {
        if (rank == 0) fprintf(stderr, "Could not load %s\n", path);
        MPI_Finalize();
        return 1;
    }

    long long local_trades = 0;
    int local_position = 0;
    double local_pnl = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    for (int s = rank; s < market.symbol_count; s += size) {
        SimResult r = simulate_symbol(&market.symbols[s]);
        local_trades += r.trades;
        local_position += r.final_position;
        local_pnl += r.pnl;
    }

    long long total_trades = 0;
    int total_position = 0;
    double total_pnl = 0.0;

    MPI_Reduce(&local_trades, &total_trades, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_position, &total_position, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_pnl, &total_pnl, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double local_seconds = MPI_Wtime() - start;
    double max_seconds = 0.0;
    MPI_Reduce(&local_seconds, &max_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        print_summary("mpi", size, market.total_events,
                      total_trades, total_position, total_pnl, max_seconds);
    }

    free_market_data(&market);
    MPI_Finalize();
    return 0;
}
