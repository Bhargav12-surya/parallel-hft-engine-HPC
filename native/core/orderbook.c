/* Binary heaps preserve price/time priority; an open-addressed ID index gives
 * expected O(1) lookup, O(log live_orders) submission/cancellation. */
#include "hft_orderbook.h"
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#define TOMBSTONE UINT64_MAX
static size_t hash_id(uint64_t x) {
    x ^= x >> 30; x *= UINT64_C(0xbf58476d1ce4e5b9);
    x ^= x >> 27; x *= UINT64_C(0x94d049bb133111eb);
    return (size_t)(x ^ (x >> 31));
}
static size_t map_lookup(const HftBook *b, uint64_t id) {
    if (!id || id == TOMBSTONE) return 0;
    size_t p = hash_id(id) & (b->map_capacity - 1);
    for (size_t k = 0; k < b->map_capacity; ++k) {
        if (!b->keys[p]) return 0;
        if (b->keys[p] == id) return b->values[p];
        p = (p + 1) & (b->map_capacity - 1);
    }
    return 0;
}
static int map_rebuild(HftBook *b, size_t capacity) {
    if (capacity > SIZE_MAX / sizeof(uint64_t) || capacity > SIZE_MAX / sizeof(size_t)) return 0;
    uint64_t *keys = calloc(capacity, sizeof(*keys));
    size_t *values = calloc(capacity, sizeof(*values));
    if (!keys || !values) { free(keys); free(values); return 0; }
    for (size_t i = 0; i < b->map_capacity; ++i) {
        uint64_t id = b->keys[i];
        if (id && id != TOMBSTONE) {
            size_t p = hash_id(id) & (capacity - 1);
            while (keys[p]) p = (p + 1) & (capacity - 1);
            keys[p] = id; values[p] = b->values[i];
        }
    }
    free(b->keys); free(b->values);
    b->keys = keys; b->values = values; b->map_capacity = capacity; b->map_tombstones = 0;
    return 1;
}
static int map_insert(HftBook *b, uint64_t id, size_t value) {
    if ((b->map_used + b->map_tombstones + 1) * 2 >= b->map_capacity) {
        size_t n = (b->map_used + 1) * 2 >= b->map_capacity ? b->map_capacity * 2 : b->map_capacity;
        if (n < b->map_capacity || !map_rebuild(b, n)) return 0;
    }
    size_t p = hash_id(id) & (b->map_capacity - 1), tomb = SIZE_MAX;
    while (b->keys[p]) {
        if (b->keys[p] == TOMBSTONE && tomb == SIZE_MAX) tomb = p;
        p = (p + 1) & (b->map_capacity - 1);
    }
    if (tomb != SIZE_MAX) { p = tomb; --b->map_tombstones; }
    b->keys[p] = id; b->values[p] = value; ++b->map_used;
    return 1;
}
static void map_delete(HftBook *b, uint64_t id) {
    size_t p = hash_id(id) & (b->map_capacity - 1);
    while (b->keys[p]) {
        if (b->keys[p] == id) { b->keys[p] = TOMBSTONE; --b->map_used; ++b->map_tombstones; return; }
        p = (p + 1) & (b->map_capacity - 1);
    }
}
static int priority(const HftBook *b, size_t a, size_t z, int side) {
    const HftOrder *x = &b->nodes[a], *y = &b->nodes[z];
    if (x->price != y->price) return side == 0 ? x->price > y->price : x->price < y->price;
    return x->sequence < y->sequence;
}
static void swap_heap(HftBook *b, int side, size_t a, size_t z) {
    size_t *h = b->heaps[side], t = h[a]; h[a] = h[z]; h[z] = t;
    b->nodes[h[a]].heap_index = a; b->nodes[h[z]].heap_index = z;
}
static void heap_up(HftBook *b, int side, size_t p) {
    while (p && priority(b, b->heaps[side][p], b->heaps[side][(p - 1) / 2], side)) {
        swap_heap(b, side, p, (p - 1) / 2); p = (p - 1) / 2;
    }
}
static void heap_down(HftBook *b, int side, size_t p) {
    size_t n = b->heap_counts[side];
    for (;;) {
        size_t q = 2 * p + 1;
        if (q >= n) break;
        if (q + 1 < n && priority(b, b->heaps[side][q + 1], b->heaps[side][q], side)) ++q;
        if (!priority(b, b->heaps[side][q], b->heaps[side][p], side)) break;
        swap_heap(b, side, p, q); p = q;
    }
}
static void remove_node(HftBook *b, size_t slot) {
    HftOrder *n = &b->nodes[slot]; int side = n->side;
    size_t p = n->heap_index, last = --b->heap_counts[side];
    if (p != last) {
        b->heaps[side][p] = b->heaps[side][last];
        b->nodes[b->heaps[side][p]].heap_index = p;
        if (p && priority(b, b->heaps[side][p], b->heaps[side][(p - 1) / 2], side)) heap_up(b, side, p);
        else heap_down(b, side, p);
    }
    map_delete(b, n->id); n->live = 0;
    n->free_next = b->free_head; b->free_head = slot;
}
static int grow(HftBook *b) {
    size_t n = b->capacity * 2;
    if (n < b->capacity || n > SIZE_MAX / sizeof(HftOrder)) return 0;
    HftOrder *nodes = calloc(n, sizeof(*nodes));
    size_t *h0 = malloc(n * sizeof(*h0)), *h1 = malloc(n * sizeof(*h1));
    if (!nodes || !h0 || !h1) { free(nodes); free(h0); free(h1); return 0; }
    memcpy(nodes, b->nodes, b->capacity * sizeof(*nodes));
    memcpy(h0, b->heaps[0], b->heap_counts[0] * sizeof(*h0));
    memcpy(h1, b->heaps[1], b->heap_counts[1] * sizeof(*h1));
    free(b->nodes); free(b->heaps[0]); free(b->heaps[1]);
    b->nodes = nodes; b->heaps[0] = h0; b->heaps[1] = h1; b->capacity = n;
    return 1;
}
int hft_book_init(HftBook *b) {
    memset(b, 0, sizeof(*b)); b->capacity = 64; b->next_slot = 1;
    b->nodes = calloc(b->capacity, sizeof(*b->nodes));
    b->heaps[0] = malloc(b->capacity * sizeof(size_t));
    b->heaps[1] = malloc(b->capacity * sizeof(size_t));
    if (!b->nodes || !b->heaps[0] || !b->heaps[1] || !map_rebuild(b, 128)) {
        hft_book_destroy(b); return 0;
    }
    return 1;
}
void hft_book_destroy(HftBook *b) {
    free(b->nodes); free(b->heaps[0]); free(b->heaps[1]); free(b->keys); free(b->values);
    memset(b, 0, sizeof(*b));
}
const HftOrder *hft_book_best(const HftBook *b, int side) {
    return b->heap_counts[side] ? &b->nodes[b->heaps[side][0]] : NULL;
}
int hft_book_cancel(HftBook *b, uint64_t id) {
    size_t n = map_lookup(b, id); if (!n) return 0;
    remove_node(b, n); return 1;
}
int hft_book_submit(HftBook *b, HftOrder o, int market, HftFillFn fill, void *context) {
    if (!o.id || o.id == TOMBSTONE || o.quantity <= 0 || o.quantity > 1000000000 ||
        o.price <= 0 || (o.side != 0 && o.side != 1) || map_lookup(b, o.id)) return 0;
    while (o.quantity > 0 && b->heap_counts[1 - o.side]) {
        size_t slot = b->heaps[1 - o.side][0];
        HftOrder maker = b->nodes[slot];
        if (!market && (o.side == 0 ? maker.price > o.price : maker.price < o.price)) break;
        int q = maker.quantity < o.quantity ? maker.quantity : o.quantity;
        if (fill) fill(context, &maker, &o, maker.price, q);
        o.quantity -= q; b->nodes[slot].quantity -= q;
        if (!b->nodes[slot].quantity) remove_node(b, slot);
    }
    if (!o.quantity || market) return 1;
    if (!b->free_head && b->next_slot == b->capacity && !grow(b)) return -1;
    size_t slot = b->free_head;
    if (slot) b->free_head = b->nodes[slot].free_next; else slot = b->next_slot++;
    if (!map_insert(b, o.id, slot)) { b->nodes[slot].free_next = b->free_head; b->free_head = slot; return -1; }
    o.sequence = ++b->sequence; o.live = 1; o.heap_index = b->heap_counts[o.side];
    b->nodes[slot] = o; b->heaps[o.side][b->heap_counts[o.side]++] = slot;
    heap_up(b, o.side, o.heap_index); return 1;
}
int hft_book_validate(const HftBook *b) {
    for (int side = 0; side < 2; ++side) {
        for (size_t i = 0; i < b->heap_counts[side]; ++i) {
            size_t n = b->heaps[side][i];
            if (!b->nodes[n].live || b->nodes[n].heap_index != i || map_lookup(b, b->nodes[n].id) != n) return 0;
            if (i && priority(b, n, b->heaps[side][(i - 1) / 2], side)) return 0;
        }
    }
    return 1;
}
