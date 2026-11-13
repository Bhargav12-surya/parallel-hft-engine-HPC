#include "hft_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
static int parse_u64(const char *text, uint64_t *result) {
    if (!*text || *text == '-' || *text == '+') return 0;
    char *end; errno = 0; unsigned long long value = strtoull(text, &end, 10);
    if (errno || *end) return 0;
    *result = (uint64_t)value; return 1;
}
static int event_type(const char *text) {
    if (!strcmp(text,"LIMIT")) return HFT_LIMIT;
    if (!strcmp(text,"MARKET")) return HFT_MARKET;
    if (!strcmp(text,"CANCEL")) return HFT_CANCEL;
    if (!strcmp(text,"QUOTE")) return HFT_QUOTE;
    if (!strcmp(text,"MODIFY")) return HFT_MODIFY;
    return -1;
}
static int compare_id(const void *a,const void *b) {
    uint64_t x=*(const uint64_t*)a,y=*(const uint64_t*)b;
    return x<y?-1:x>y;
}
int hft_dataset_load(const char *path, HftDataset *data, char *error) {
    memset(data, 0, sizeof(*data));
    FILE *file = fopen(path, "rb");
    if (!file) { snprintf(error,HFT_ERROR_SIZE,"Cannot open dataset: %s",strerror(errno)); return 0; }
    char line[2048]; size_t capacity = 0, number = 0;
    if (!fgets(line,sizeof(line),file)) { snprintf(error,HFT_ERROR_SIZE,"Empty dataset"); fclose(file); return 0; }
    line[strcspn(line,"\r\n")] = 0;
    const char *header="event_id,timestamp_ns,instrument_id,event_type,side,price_ticks,quantity,order_id,reference_order_id";
    if (strcmp(line,header)) { snprintf(error,HFT_ERROR_SIZE,"Invalid canonical CSV header"); fclose(file); return 0; }
    uint64_t last_ts = 0, last_id = 0; int ids_increasing = 1;
    while (fgets(line,sizeof(line),file)) {
        ++number;
        if (!strchr(line,'\n') && !feof(file)) goto malformed;
        line[strcspn(line,"\r\n")] = 0;
        char *fields[9], *cursor = line;
        for (int i = 0; i < 9; ++i) {
            fields[i] = cursor;
            char *comma = strchr(cursor,',');
            if (i < 8) { if (!comma) goto malformed; *comma = 0; cursor = comma + 1; }
            else if (comma) goto malformed;
        }
        HftEvent e; memset(&e,0,sizeof(e));
        uint64_t instrument, price, quantity;
        if (!parse_u64(fields[0],&e.id) || !parse_u64(fields[1],&e.timestamp_ns) ||
            !parse_u64(fields[2],&instrument) || !parse_u64(fields[5],&price) ||
            !parse_u64(fields[6],&quantity) || !parse_u64(fields[7],&e.order_id) ||
            !parse_u64(fields[8],&e.reference_id)) goto malformed;
        e.type = event_type(fields[3]);
        e.side = !strcmp(fields[4],"BUY") ? 0 : !strcmp(fields[4],"SELL") ? 1 : -1;
        if (e.type < 0 || e.side < 0 || !e.id || e.id > INT64_MAX || e.timestamp_ns > INT64_MAX || e.timestamp_ns < last_ts ||
            instrument > 65535 || price > INT64_C(1000000000000) || quantity > 1000000000 ||
            e.order_id > INT64_MAX || e.reference_id > INT64_MAX ||
            (e.type != HFT_CANCEL && (!price || !quantity)) ||
            ((e.type == HFT_LIMIT || e.type == HFT_MARKET || e.type == HFT_MODIFY) && !e.order_id) ||
            ((e.type == HFT_CANCEL || e.type == HFT_MODIFY) && !e.reference_id) ||
            ((e.type != HFT_CANCEL && e.type != HFT_MODIFY) && e.reference_id)) goto malformed;
        e.instrument = (int32_t)instrument; e.price = (int64_t)price; e.quantity = (int32_t)quantity;
        if (e.id <= last_id) ids_increasing = 0;
        last_ts = e.timestamp_ns; last_id = e.id;
        if (data->count == capacity) {
            size_t n = capacity ? capacity * 2 : 4096;
            if (n < capacity || n > SIZE_MAX / sizeof(HftEvent)) goto allocation;
            HftEvent *next = realloc(data->events,n * sizeof(*next));
            if (!next) goto allocation;
            data->events = next; capacity = n;
        }
        data->events[data->count++] = e;
        if (instrument + 1 > data->instruments) data->instruments = (size_t)instrument + 1;
    }
    if (ferror(file)) { snprintf(error,HFT_ERROR_SIZE,"Dataset read failed"); goto failure; }
    if (!data->count) { snprintf(error,HFT_ERROR_SIZE,"Dataset contains no events"); goto failure; }
    if (!ids_increasing) {
        if (data->count > SIZE_MAX/sizeof(uint64_t)) goto allocation;
        uint64_t *ids = malloc(data->count*sizeof(*ids)); if(!ids) goto allocation;
        for(size_t i=0;i<data->count;++i) ids[i]=data->events[i].id;
        qsort(ids,data->count,sizeof(*ids),compare_id);
        for(size_t i=1;i<data->count;++i) if(ids[i]==ids[i-1]) {
            free(ids);snprintf(error,HFT_ERROR_SIZE,"Duplicate event ID in dataset");goto failure;
        }
        free(ids);
    }
    fclose(file); return 1;
malformed:
    snprintf(error,HFT_ERROR_SIZE,"Malformed or unordered event at CSV line %zu",number + 1); goto failure;
allocation:
    snprintf(error,HFT_ERROR_SIZE,"Dataset exceeds available address space/memory");
failure:
    fclose(file); hft_dataset_free(data); return 0;
}
void hft_dataset_free(HftDataset *data) { free(data->events); memset(data,0,sizeof(*data)); }
