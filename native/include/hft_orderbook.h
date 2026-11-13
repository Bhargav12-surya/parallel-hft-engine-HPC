#ifndef HFT_ORDERBOOK_H
#define HFT_ORDERBOOK_H
#include "hft_types.h"
typedef struct {
    uint64_t id, sequence;
    int64_t price;
    int quantity, side, agent, live;
    size_t heap_index, free_next;
} HftOrder;
typedef struct {
    HftOrder *nodes;
    size_t *heaps[2], heap_counts[2], capacity, next_slot, free_head;
    uint64_t *keys;
    size_t *values, map_capacity, map_used, map_tombstones;
    uint64_t sequence;
} HftBook;
typedef void (*HftFillFn)(void *context, const HftOrder *maker,
                          const HftOrder *taker, int64_t price, int quantity);
/* Return 1 to fill, 0 to reject the remaining taker quantity, or -1 to
 * cancel the maker and continue matching the taker. */
typedef int (*HftFillCheckFn)(void *context, const HftOrder *maker,
                             const HftOrder *taker, int64_t price, int quantity);
int hft_book_init(HftBook *book);
void hft_book_destroy(HftBook *book);
int hft_book_cancel(HftBook *book, uint64_t id);
int hft_book_submit(HftBook *book, HftOrder order, int market,
                    HftFillFn fill, void *context);
int hft_book_submit_checked(HftBook *book, HftOrder order, int market,
                            HftFillCheckFn check, HftFillFn fill, void *context);
const HftOrder *hft_book_best(const HftBook *book, int side);
int hft_book_validate(const HftBook *book);
#endif
