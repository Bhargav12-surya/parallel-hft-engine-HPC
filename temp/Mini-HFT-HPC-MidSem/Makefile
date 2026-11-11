CC ?= gcc
MPICC ?= mpicc
CFLAGS ?= -O2 -std=c17 -Wall -Wextra -pedantic -Iinclude
COMMON = src/mini_hft.c
BIN = bin

.PHONY: all clean data mpi run

all: $(BIN)/generate_data $(BIN)/sequential $(BIN)/pthreads $(BIN)/openmp

$(BIN):
	mkdir -p $(BIN)

$(BIN)/generate_data: src/generate_data.c | $(BIN)
	$(CC) $(CFLAGS) $< -o $@

$(BIN)/sequential: src/sequential.c $(COMMON) include/mini_hft.h | $(BIN)
	$(CC) $(CFLAGS) src/sequential.c $(COMMON) -o $@

$(BIN)/pthreads: src/pthreads_version.c $(COMMON) include/mini_hft.h | $(BIN)
	$(CC) $(CFLAGS) src/pthreads_version.c $(COMMON) -pthread -o $@

$(BIN)/openmp: src/openmp_version.c $(COMMON) include/mini_hft.h | $(BIN)
	$(CC) $(CFLAGS) src/openmp_version.c $(COMMON) -fopenmp -o $@

mpi: $(BIN)/mpi

$(BIN)/mpi: src/mpi_version.c $(COMMON) include/mini_hft.h | $(BIN)
	$(MPICC) $(CFLAGS) src/mpi_version.c $(COMMON) -o $@

data: $(BIN)/generate_data
	mkdir -p data
	./$(BIN)/generate_data 4 10000 data/market.csv

run: all data
	./$(BIN)/sequential data/market.csv
	./$(BIN)/pthreads data/market.csv 2
	./$(BIN)/openmp data/market.csv 2

clean:
	rm -f $(BIN)/generate_data $(BIN)/sequential $(BIN)/pthreads $(BIN)/openmp $(BIN)/mpi
	rm -f data/market.csv results/*.txt
