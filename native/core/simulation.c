#include "hft_engine.h"
#include "hft_orderbook.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <inttypes.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#else
#include <pthread.h>
#include <sys/resource.h>
#endif
#ifdef HFT_HAVE_OPENMP
#include <omp.h>
#endif
#ifdef HFT_HAVE_MPI
#include <mpi.h>
#endif

typedef struct {
    HftBook book;
    HftPortfolio portfolio;
    HftFeatures features;
    const HftConfig *config;
    HftInstrumentResult result;
    double mark, initial_mark, capital, peak;
    uint64_t sequence;
    int instrument, halted, failure;
    HftOrder pending[32];
    uint64_t due[32];
    int pending_market[32], pending_count;
} Instrument;
typedef struct {
    const HftConfig *config;
    const HftDataset *data;
    const HftModel *model;
    HftResult *result;
    size_t *indices, *offsets, groups, group_width, next_group, completed;
    uint64_t completed_trades, completed_orders, completed_timestamp;
    int rank, ranks, failed, force_sequential, use_precomputed;
    double *features, *probabilities, *signals, *latencies, *equity_deltas;
    size_t sample_stride, sample_count;
    double started, completed_pnl;
#ifdef _WIN32
    CRITICAL_SECTION mutex;
#else
    pthread_mutex_t mutex;
#endif
} Work;
static void lock(Work *w) {
#ifdef _WIN32
    EnterCriticalSection(&w->mutex);
#else
    pthread_mutex_lock(&w->mutex);
#endif
}
static void unlock(Work *w) {
#ifdef _WIN32
    LeaveCriticalSection(&w->mutex);
#else
    pthread_mutex_unlock(&w->mutex);
#endif
}
static uint64_t hash_step(uint64_t hash, uint64_t value) {
    for (int j = 0; j < 8; ++j) { hash ^= value & 255; hash *= UINT64_C(1099511628211); value >>= 8; }
    return hash;
}
static void on_fill(void *context, const HftOrder *maker, const HftOrder *taker, int64_t price, int quantity) {
    Instrument *s = context;
    const HftOrder *orders[2] = { maker, taker };
    for (int j = 0; j < 2; ++j) if (orders[j]->agent) {
        int side = orders[j]->side;
        int64_t adjusted = price + (side ? -1 : 1) * s->config->slippage_ticks;
        if (adjusted < 1) adjusted = 1;
        hft_portfolio_fill(&s->portfolio,side,adjusted,quantity,s->config->fee_bps,s->config->tick_value);
        s->portfolio.slippage += fabs((double)adjusted - price) * quantity * s->config->tick_value;
        s->result.checksum = hash_step(s->result.checksum,orders[j]->id);
        s->result.checksum = hash_step(s->result.checksum,(uint64_t)adjusted);
        s->result.checksum = hash_step(s->result.checksum,(uint64_t)quantity | ((uint64_t)side << 32));
    }
}
static void submit(Instrument *s, HftOrder order, int market) {
    int result = hft_book_submit(&s->book,order,market,on_fill,s);
    if (result < 0) s->failure = 1;
    if (result <= 0) ++s->result.rejected;
}
static uint64_t quote_id(int side, int agent) { return UINT64_C(0xe000000000000000) + (uint64_t)agent * 2 + (uint64_t)side; }
static HftOrder new_order(uint64_t id, int64_t price, int quantity, int side, int agent) {
    HftOrder order; memset(&order,0,sizeof(order));
    order.id = id; order.price = price; order.quantity = quantity; order.side = side; order.agent = agent;
    return order;
}
static void strategy_cancel_quotes(Instrument *s) {
    for (int side = 0; side < 2; ++side) s->result.cancels += hft_book_cancel(&s->book,quote_id(side,1));
}
static void mark_to_market(Instrument *s) {
    double equity = s->portfolio.cash + s->portfolio.position * s->mark * s->config->tick_value;
    double pnl = equity - s->capital;
    if (equity > s->peak) s->peak = equity;
    double drawdown = s->peak - equity;
    if (drawdown > s->result.drawdown) s->result.drawdown = drawdown;
    s->result.pnl = pnl; s->result.equity = equity; s->result.position = s->portfolio.position;
    s->result.trades = s->portfolio.trades; s->result.fees = s->portfolio.fees;
    s->result.realized = s->portfolio.realized;
    s->result.unrealized = s->portfolio.position ?
        (s->portfolio.position > 0 ? 1 : -1) * (fabs((double)s->portfolio.position) * s->mark * s->config->tick_value - s->portfolio.cost) : 0;
    s->result.slippage = s->portfolio.slippage;
    s->result.gross_exposure = fabs((double)s->portfolio.position) * s->mark * s->config->tick_value;
    if (pnl < -s->config->max_loss && !s->halted) { s->halted = 1; strategy_cancel_quotes(s); s->pending_count = 0; }
}
static int risk_check(Instrument *s, int side, int quantity, int64_t price) {
    int64_t next = s->portfolio.position + (side ? -quantity : quantity);
    double gross = fabs((double)next) * price * s->config->tick_value;
    if (s->halted || quantity <= 0 || quantity > s->config->max_order ||
        llabs(next) > s->config->position_limit || gross > s->capital * 2 ||
        (!side && quantity * (double)price * s->config->tick_value * (1 + s->config->fee_bps * .0001) > s->portfolio.cash)) {
        ++s->result.rejected; return 0;
    }
    return 1;
}
static void send_strategy(Instrument *s, HftOrder order, int market, uint64_t now) {
    if (!risk_check(s,order.side,order.quantity,order.price)) return;
    ++s->result.orders;
    if (s->config->latency_ns) {
        if (s->pending_count == 32 || UINT64_MAX - now < s->config->latency_ns) { ++s->result.rejected; return; }
        int j = s->pending_count++; s->pending[j] = order;
        s->due[j] = now + s->config->latency_ns; s->pending_market[j] = market;
    } else submit(s,order,market);
}
static void deliver_pending(Instrument *s, uint64_t now) {
    int j = 0;
    while (j < s->pending_count) {
        if (s->due[j] > now) { ++j; continue; }
        HftOrder order = s->pending[j];
        if (risk_check(s,order.side,order.quantity,order.price)) {
            if (!s->pending_market[j]) hft_book_cancel(&s->book,order.id);
            submit(s,order,s->pending_market[j]);
        }
        --s->pending_count;
        memmove(&s->pending[j],&s->pending[j+1],(size_t)(s->pending_count-j)*sizeof(*s->pending));
        memmove(&s->due[j],&s->due[j+1],(size_t)(s->pending_count-j)*sizeof(*s->due));
        memmove(&s->pending_market[j],&s->pending_market[j+1],(size_t)(s->pending_count-j)*sizeof(*s->pending_market));
    }
}
static void external_event(Instrument *s, const HftEvent *event) {
    deliver_pending(s,event->timestamp_ns);
    if (event->type != HFT_CANCEL && event->price > 0) s->mark = (double)event->price;
    if (!s->initial_mark && s->mark) s->initial_mark = s->mark;
    if (event->type == HFT_QUOTE) {
        for (int side = 0; side < 2; ++side) hft_book_cancel(&s->book,quote_id(side,0));
        int64_t mid = event->price;
        submit(s,new_order(quote_id(0,0),mid > 1 ? mid - 1 : 1,event->quantity,0,0),0);
        submit(s,new_order(quote_id(1,0),mid + 1,event->quantity,1,0),0);
    } else if (event->type == HFT_CANCEL) {
        if (hft_book_cancel(&s->book,event->reference_id)) ++s->result.cancels; else ++s->result.rejected;
    } else if (event->type == HFT_MODIFY) {
        if (hft_book_cancel(&s->book,event->reference_id)) {
            ++s->result.cancels;
            submit(s,new_order(event->order_id,event->price,event->quantity,event->side,0),0);
        } else ++s->result.rejected;
    } else submit(s,new_order(event->order_id,event->price,event->quantity,event->side,0),event->type == HFT_MARKET);
    ++s->result.events; s->result.simulated_time_ns = event->timestamp_ns;
    s->result.checksum = hash_step(s->result.checksum,event->id);
    mark_to_market(s);
}
static double independent_signal(const HftConfig *c, const double *f, const double *p) {
    if (!strcmp(c->strategy,"ai")) {
        int best = p[2] > p[0] ? 2 : 0;
        if (p[1] > p[best]) best = 1;
        return p[best] >= .45 ? (double)(best - 1) : 0;
    }
    double threshold = fmax(2 * c->fee_bps * .0001 + f[5],.0002);
    if (!strcmp(c->strategy,"momentum")) return f[1] > threshold ? 1 : f[1] < -threshold ? -1 : 0;
    if (!strcmp(c->strategy,"mean-reversion")) return f[2] > threshold ? -1 : f[2] < -threshold ? 1 : 0;
    /* Market makers use this causal volatility estimate to widen quotes. */
    return f[3];
}
static void trade_target(Instrument *s, int target, uint64_t now) {
    int64_t delta = target - s->portfolio.position;
    if (!delta || s->pending_count) return;
    int side = delta < 0, quantity = (int)(llabs(delta) > s->config->max_order ? s->config->max_order : llabs(delta));
    const HftOrder *opposite = hft_book_best(&s->book,1-side);
    if (!opposite) return;
    strategy_cancel_quotes(s);
    uint64_t id = UINT64_C(0xc000000000000000) + ++s->sequence;
    send_strategy(s,new_order(id,opposite->price,quantity,side,1),1,now);
}
static void strategy_event(Instrument *s, const HftEvent *event, const double *features, double signal) {
    if (s->halted || s->result.events < 17) return;
    if (!strcmp(s->config->strategy,"hold")) return;
    if (!strcmp(s->config->strategy,"market-making")) {
        if (s->result.events % 8 || s->pending_count) return;
        strategy_cancel_quotes(s);
        int width = 2 + (int)fmin(100.0,ceil(fabs(signal) * s->mark * .5));
        int skew = (int)(s->portfolio.position / (s->config->max_order ? s->config->max_order : 1));
        for (int side = 0; side < 2; ++side) {
            int64_t price = (int64_t)s->mark + (side ? width : -width) - skew;
            if (price < 1) price = 1;
            send_strategy(s,new_order(quote_id(side,1),price,s->config->max_order,side,1),0,event->timestamp_ns);
        }
    } else if (strcmp(s->config->strategy,"stat-arb") && s->result.events % 4 == 0) {
        int desired = (signal > 0 ? 1 : signal < 0 ? -1 : 0) * s->config->max_order;
        trade_target(s,desired,event->timestamp_ns);
    }
    (void)features;
    mark_to_market(s);
}
static void precompute_group(Work *w, size_t group) {
    HftFeatures state[2]; memset(state,0,sizeof(state));
    for (size_t j = w->offsets[group]; j < w->offsets[group+1]; ++j) {
        size_t index = w->indices[j]; const HftEvent *e = &w->data->events[index];
        double *f = w->features + index * HFT_FEATURES;
        hft_features_update(&state[(size_t)e->instrument % w->group_width],e->type==HFT_CANCEL?0:(double)e->price,e->side,e->quantity,f);
        if (w->model) hft_model_predict(w->model,f,w->probabilities+index*3);
    }
}
static void progress(Work *w, size_t events, double pnl, uint64_t trades, uint64_t orders, uint64_t timestamp) {
    lock(w); w->completed += events; w->completed_pnl += pnl;
    w->completed_trades += trades; w->completed_orders += orders;
    if (timestamp > w->completed_timestamp) w->completed_timestamp = timestamp;
    if (!w->config->quiet && w->rank == 0) {
        double elapsed = hft_now() - w->started;
        printf("{\"schema_version\":1,\"type\":\"progress\",\"phase\":\"matching\",\"events_processed\":%zu,\"total_events\":%zu,\"elapsed_seconds\":%.9g,\"pnl\":%.12g,\"equity\":%.12g,\"trades\":%" PRIu64 ",\"orders\":%" PRIu64 ",\"timestamp_ns\":%" PRIu64 "}\n",
            w->completed,w->data->count,elapsed,w->completed_pnl,w->config->capital+w->completed_pnl,w->completed_trades,w->completed_orders,w->completed_timestamp);
        fflush(stdout);
    }
    unlock(w);
}
static void simulate_group(Work *w, size_t group) {
    double started = hft_now();
    Instrument states[2]; memset(states,0,sizeof(states));
    size_t actual = w->data->instruments - group * w->group_width;
    if (actual > w->group_width) actual = w->group_width;
    for (size_t j = 0; j < actual; ++j) {
        Instrument *s = &states[j]; s->config = w->config; s->instrument = (int)(group*w->group_width+j);
        s->capital = w->config->capital / (double)w->data->instruments;
        s->portfolio.cash = s->capital; s->peak = s->capital;
        s->result.checksum = UINT64_C(14695981039346656037);
        if (!hft_book_init(&s->book)) s->failure = 1;
    }
    double pair_history[32] = {0}; size_t pair_count = 0; double previous_group_pnl = 0;
    for (size_t j = w->offsets[group]; j < w->offsets[group+1]; ++j) {
        size_t index = w->indices[j]; const HftEvent *e = &w->data->events[index];
        Instrument *s = &states[(size_t)e->instrument % w->group_width];
        if (s->failure) break;
        int sample = index % w->sample_stride == 0;
        double before = sample ? hft_now() : 0;
        external_event(s,e);
        double buffer[HFT_FEATURES], probabilities[3] = {0};
        const double *features, *p;
        if (w->use_precomputed) { features = w->features+index*HFT_FEATURES; p = w->probabilities+index*3; }
        else {
            hft_features_update(&s->features,e->type==HFT_CANCEL?0:(double)e->price,e->side,e->quantity,buffer);
            features = buffer; p = probabilities;
            if (w->model) hft_model_predict(w->model,features,probabilities);
        }
        double signal = w->signals ? w->signals[index] : independent_signal(w->config,features,p);
        strategy_event(s,e,features,signal);
        if (!strcmp(w->config->strategy,"stat-arb") && actual == 2 && states[0].initial_mark && states[1].initial_mark) {
            double spread = log(states[0].mark/states[0].initial_mark) - log(states[1].mark/states[1].initial_mark);
            pair_history[pair_count % 32] = spread; ++pair_count;
            size_t n = pair_count < 32 ? pair_count : 32; double mean = 0, variance = 0;
            for (size_t k = 0; k < n; ++k) mean += pair_history[k]; mean /= (double)n;
            for (size_t k = 0; k < n; ++k) { double d = pair_history[k]-mean; variance += d*d; }
            double sd = sqrt(variance/(double)n), z = sd > 1e-12 ? (spread-mean)/sd : 0;
            if (pair_count >= 32 && pair_count % 4 == 0) {
                int direction = z > 1.5 ? -1 : z < -1.5 ? 1 : fabs(z)<.5 ? 0 : 2;
                if (direction != 2) {
                    trade_target(&states[0],direction*w->config->max_order,e->timestamp_ns);
                    /* Dollar-neutral target rounded to units; execution of legs is
                     * independent and exposes actual leg/liquidity risk. */
                    int hedge = (int)fmin(w->config->position_limit,floor(w->config->max_order*states[0].mark/states[1].mark));
                    trade_target(&states[1],-direction*hedge,e->timestamp_ns);
                }
            }
        }
        if (sample) w->latencies[index/w->sample_stride] = (hft_now()-before)*1e6;
        double group_pnl = 0;
        for (size_t k=0;k<actual;++k) { mark_to_market(&states[k]); group_pnl += states[k].result.pnl; }
        w->equity_deltas[index] = group_pnl - previous_group_pnl;
        previous_group_pnl = group_pnl;
    }
    double pnl = 0; int failure = 0; uint64_t trades=0,orders=0,timestamp=0;
    for (size_t j = 0; j < actual; ++j) {
        Instrument *s = &states[j]; mark_to_market(s);
        s->result.rejected += (uint64_t)s->pending_count;
        s->result.compute_seconds = hft_now()-started;
        w->result->instruments[group*w->group_width+j] = s->result;
        pnl += s->result.pnl; failure |= s->failure; hft_book_destroy(&s->book);
        trades+=s->result.trades; orders+=s->result.orders;
        if(s->result.simulated_time_ns>timestamp)timestamp=s->result.simulated_time_ns;
    }
    if (failure) { lock(w); w->failed = 1; unlock(w); }
    progress(w,w->offsets[group+1]-w->offsets[group],pnl,trades,orders,timestamp);
}
static void run_worker(Work *w) {
    int pin = !strncmp(w->config->cpu_mode,"single-",7);
    int affinity = hft_pin_current_thread(pin);
    lock(w); w->result->affinity_applied += affinity; unlock(w);
    for (;;) {
        lock(w); size_t g = w->next_group++;
        while (g < w->groups && (int)(g % (size_t)w->ranks) != w->rank) g = w->next_group++;
        unlock(w);
        if (g >= w->groups) break;
        simulate_group(w,g);
    }
}
#ifdef _WIN32
static DWORD WINAPI worker_entry(LPVOID pointer) { run_worker(pointer); return 0; }
#else
static void *worker_entry(void *pointer) { run_worker(pointer); return NULL; }
#endif
static int run_threads(Work *w, int workers) {
#ifdef _WIN32
    HANDLE *handles = calloc((size_t)workers,sizeof(*handles));
#else
    pthread_t *handles = calloc((size_t)workers,sizeof(*handles));
#endif
    if (!handles) return 0;
    int started = 0;
    for (int i = 0; i < workers; ++i) {
#ifdef _WIN32
        handles[i] = CreateThread(NULL,0,worker_entry,w,0,NULL); if (!handles[i]) break;
#else
        if (pthread_create(&handles[i],NULL,worker_entry,w)) break;
#endif
        ++started;
    }
    for (int i = 0; i < started; ++i) {
#ifdef _WIN32
        WaitForSingleObject(handles[i],INFINITE); CloseHandle(handles[i]);
#else
        pthread_join(handles[i],NULL);
#endif
    }
    free(handles); return started == workers;
}
static int compare_double(const void *a, const void *b) { double x=*(const double*)a,y=*(const double*)b; return x<y?-1:x>y; }
static int component(const char *list, const char *wanted) {
    size_t n = strlen(wanted); const char *p = list;
    while (p && *p) { if (!strncmp(p,wanted,n) && (!p[n] || p[n]==',')) return 1; p=strchr(p,','); if(p)++p; }
    return 0;
}
#ifdef HFT_HAVE_MPI
static int collective_ok(int local, int mpi) {
    if (!mpi) return local;
    int all = 0;
    if (MPI_Allreduce(&local,&all,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD) != MPI_SUCCESS) {
        MPI_Abort(MPI_COMM_WORLD,3); return 0;
    }
    return all;
}
static int reduce_doubles(double *values, size_t count, int root) {
    while (count) { int n = count > 100000000 ? 100000000 : (int)count;
        int rank; MPI_Comm_rank(MPI_COMM_WORLD,&rank);
        int status = MPI_Reduce(rank==root?MPI_IN_PLACE:values,values,n,MPI_DOUBLE,MPI_SUM,root,MPI_COMM_WORLD);
        if (status != MPI_SUCCESS) return 0; values += n; count -= (size_t)n; }
    return 1;
}
#else
static int collective_ok(int local, int mpi) { (void)mpi; return local; }
#endif
int hft_run(const HftConfig *config, const HftDataset *data, const HftModel *model, HftResult *result) {
    Work w; memset(&w,0,sizeof(w));
    w.config=config; w.data=data; w.model=model; w.result=result; w.ranks=1;
    w.group_width=!strcmp(config->strategy,"stat-arb")?2:1;
    w.groups=(data->instruments+w.group_width-1)/w.group_width;
    int hybrid=!strcmp(config->engine,"hybrid");
    int cpu=hybrid&&component(config->components,"cpu");
    int omp=!strcmp(config->engine,"openmp")||(hybrid&&component(config->components,"openmp"));
    int mpi=!strcmp(config->engine,"mpi")||(hybrid&&component(config->components,"mpi"));
    int cuda=!strcmp(config->engine,"cuda")||(hybrid&&component(config->components,"cuda"));
    if (hybrid && cpu+omp+mpi+cuda<2) { snprintf(result->error,sizeof(result->error),"Hybrid requires two or more distinct components"); return 0; }
#ifndef HFT_HAVE_OPENMP
    if(omp){snprintf(result->error,sizeof(result->error),"OpenMP was not compiled");return 0;}
#endif
#ifndef HFT_HAVE_MPI
    if(mpi){snprintf(result->error,sizeof(result->error),"MPI was not compiled");return 0;}
#else
    if(mpi){MPI_Comm_rank(MPI_COMM_WORLD,&w.rank);MPI_Comm_size(MPI_COMM_WORLD,&w.ranks);}
#endif
    if(!collective_ok(!cuda || hft_cuda_available(NULL,0),mpi)){snprintf(result->error,sizeof(result->error),"CUDA runtime/device unavailable on one or more ranks, or CUDA not compiled");return 0;}
    w.sample_stride=data->count/65536+1; w.sample_count=(data->count+w.sample_stride-1)/w.sample_stride;
    result->instrument_count=data->instruments; result->ranks=w.ranks;
    result->instruments=calloc(data->instruments,sizeof(*result->instruments));
    w.offsets=calloc(w.groups+1,sizeof(*w.offsets)); w.indices=malloc(data->count*sizeof(*w.indices));
    w.latencies=calloc(w.sample_count,sizeof(double));
    w.equity_deltas=calloc(data->count,sizeof(double));
    if(!collective_ok(result->instruments&&w.offsets&&w.indices&&w.latencies&&w.equity_deltas,mpi))goto allocation;
    for(size_t i=0;i<data->count;++i)++w.offsets[(size_t)data->events[i].instrument/w.group_width+1];
    for(size_t i=1;i<=w.groups;++i)w.offsets[i]+=w.offsets[i-1];
    size_t *cursor=malloc(w.groups*sizeof(*cursor));if(!collective_ok(cursor!=NULL,mpi)){free(cursor);goto allocation;}
    memcpy(cursor,w.offsets,w.groups*sizeof(*cursor));
    for(size_t i=0;i<data->count;++i)w.indices[cursor[(size_t)data->events[i].instrument/w.group_width]++]=i;
    free(cursor);
#ifdef _WIN32
    InitializeCriticalSection(&w.mutex);
#else
    pthread_mutex_init(&w.mutex,NULL);
#endif
    w.started=hft_now();
    int threads=config->threads>0?config->threads:hft_logical_cpus();
    if(threads>64)threads=64;
    if((size_t)threads>w.groups)threads=(int)w.groups;
    if(threads<1)threads=1;
    if ((!strcmp(config->engine,"sequential") || !strcmp(config->engine,"threaded")) &&
        (!strcmp(config->cpu_mode,"single-single") || !strcmp(config->cpu_mode,"multi-single"))) threads=1;
    result->workers=threads;
    w.use_precomputed=hybrid||cuda;
    if(w.use_precomputed){
        if(!collective_ok(data->count<=SIZE_MAX/(HFT_FEATURES*sizeof(double)),mpi))goto allocation_locked;
        w.features=calloc(data->count*HFT_FEATURES,sizeof(double));
        w.probabilities=calloc(data->count*3,sizeof(double));
        if(!collective_ok(w.features&&w.probabilities,mpi))goto allocation_locked;
        double feature_start=hft_now();
        if(cuda){
            /* Pack only this rank's independent groups: no duplicate device work. */
            size_t local_count=0;
            for(size_t g=(size_t)w.rank;g<w.groups;g+=(size_t)w.ranks)local_count+=w.offsets[g+1]-w.offsets[g];
            HftEvent *packed=malloc(local_count*sizeof(*packed));
            size_t *mapping=malloc(local_count*sizeof(*mapping));
            double *local_features=calloc(local_count*HFT_FEATURES,sizeof(double)),*local_p=calloc(local_count*3,sizeof(double));
            if(!collective_ok(!local_count||(packed&&mapping&&local_features&&local_p),mpi)){free(packed);free(mapping);free(local_features);free(local_p);goto allocation_locked;}
            size_t k=0;for(size_t g=(size_t)w.rank;g<w.groups;g+=(size_t)w.ranks)for(size_t j=w.offsets[g];j<w.offsets[g+1];++j){mapping[k]=w.indices[j];packed[k]=data->events[mapping[k]];++k;}
            int ok=!local_count||hft_cuda_features(packed,local_count,(int)data->instruments,model,local_features,local_p,result);
            if(ok)for(size_t j=0;j<local_count;++j){memcpy(w.features+mapping[j]*HFT_FEATURES,local_features+j*HFT_FEATURES,HFT_FEATURES*sizeof(double));memcpy(w.probabilities+mapping[j]*3,local_p+j*3,3*sizeof(double));}
            free(packed);free(mapping);free(local_features);free(local_p);
            if(!collective_ok(ok,mpi))goto failure_locked;
        }else if(omp){
#ifdef HFT_HAVE_OPENMP
            int g;
#pragma omp parallel num_threads(threads)
            {
#pragma omp single
                result->workers=omp_get_num_threads();
#pragma omp for schedule(static)
            for(g=0;g<(int)w.groups;++g)if((int)((size_t)g%(size_t)w.ranks)==w.rank)precompute_group(&w,(size_t)g);
            }
#endif
        }else for(size_t g=(size_t)w.rank;g<w.groups;g+=(size_t)w.ranks)precompute_group(&w,g);
        result->feature_seconds=hft_now()-feature_start;
#ifdef HFT_HAVE_MPI
        if(mpi&&cpu){
            double begin=hft_now();
            if(!reduce_doubles(w.features,data->count*HFT_FEATURES,0)||!reduce_doubles(w.probabilities,data->count*3,0))goto failure_locked;
            result->communication_seconds+=hft_now()-begin;
            result->bytes_transferred+=(uint64_t)data->count*(HFT_FEATURES+3)*sizeof(double)*(uint64_t)(w.ranks-1);
        }
#endif
        /* OpenMP has a separate signal-generation role when CUDA extracted
         * features. Matching remains deterministic on the CPU if selected. */
        if(omp&&cuda){
            w.signals=calloc(data->count,sizeof(double));if(!collective_ok(w.signals!=NULL,mpi))goto allocation_locked;
#ifdef HFT_HAVE_OPENMP
            int g;
#pragma omp parallel num_threads(threads)
            {
#pragma omp single
                result->workers=omp_get_num_threads();
#pragma omp for schedule(static)
            for(g=0;g<(int)w.groups;++g)for(size_t j=w.offsets[g];j<w.offsets[g+1];++j){size_t i=w.indices[j];w.signals[i]=independent_signal(config,w.features+i*HFT_FEATURES,w.probabilities+i*3);}
            }
#endif
        }
    }
    {double begin=hft_now();
        if(mpi&&cpu){
            if(w.rank==0){w.ranks=1;for(size_t g=0;g<w.groups;++g)simulate_group(&w,g);w.ranks=result->ranks;}
        }else if(omp&&!cpu){
#ifdef HFT_HAVE_OPENMP
            int g;
#pragma omp parallel num_threads(threads)
            {
#pragma omp single
                result->workers=omp_get_num_threads();
#pragma omp for schedule(dynamic,1)
            for(g=0;g<(int)w.groups;++g)if((int)((size_t)g%(size_t)w.ranks)==w.rank)simulate_group(&w,(size_t)g);
            }
#endif
        }else if(!strcmp(config->engine,"threaded")||(!strcmp(config->engine,"sequential")&&strstr(config->cpu_mode,"multi")&&strcmp(config->cpu_mode,"multi-single"))){
            if(!run_threads(&w,threads))w.failed=1;
        }else{
            if(!omp)result->workers=1;
            result->affinity_applied=hft_pin_current_thread(!strncmp(config->cpu_mode,"single-",7));
            for(size_t g=(size_t)w.rank;g<w.groups;g+=(size_t)w.ranks)simulate_group(&w,g);
        }
        result->match_seconds=hft_now()-begin; result->compute_seconds=result->feature_seconds+result->match_seconds;
    }
#ifdef HFT_HAVE_MPI
    if(mpi){
        double begin=hft_now(); int failed=0;
        MPI_Allreduce(&w.failed,&failed,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD); w.failed=failed;
        if(!cpu){
            size_t bytes=data->instruments*sizeof(HftInstrumentResult);
            if(bytes>INT32_MAX||(size_t)w.ranks>SIZE_MAX/bytes){w.failed=1;}
            else{
                HftInstrumentResult *all=w.rank==0?calloc((size_t)w.ranks*data->instruments,sizeof(*all)):NULL;
                int alloc_ok=w.rank!=0||all!=NULL;MPI_Bcast(&alloc_ok,1,MPI_INT,0,MPI_COMM_WORLD);
                if(!alloc_ok)w.failed=1;
                else if(MPI_Gather(result->instruments,(int)bytes,MPI_BYTE,all,(int)bytes,MPI_BYTE,0,MPI_COMM_WORLD)!=MPI_SUCCESS)w.failed=1;
                if(w.rank==0&&all){for(size_t i=0;i<data->instruments;++i){size_t owner=(i/w.group_width)%(size_t)w.ranks;result->instruments[i]=all[owner*data->instruments+i];}free(all);}
                result->bytes_transferred+=(uint64_t)bytes*(uint64_t)(w.ranks-1);
            }
            reduce_doubles(w.latencies,w.sample_count,0);
            reduce_doubles(w.equity_deltas,data->count,0);
        }
        double compute=result->compute_seconds,maximum=0;MPI_Reduce(&compute,&maximum,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);if(w.rank==0)result->compute_seconds=maximum;
        result->communication_seconds+=hft_now()-begin;
    }
#endif
    result->wall_seconds=hft_now()-w.started;
    if(w.rank==0){
        double equity=config->capital,peak=equity;size_t sample_every=data->count/255+1;
        result->equity_curve[0]=equity;result->equity_timestamps[0]=data->events[0].timestamp_ns;result->equity_samples=1;
        for(size_t i=0;i<data->count;++i){
            equity+=w.equity_deltas[i];if(equity>peak)peak=equity;
            if(peak-equity>result->portfolio_drawdown)result->portfolio_drawdown=peak-equity;
            if((i%sample_every==0||i+1==data->count)&&result->equity_samples<256){size_t j=result->equity_samples++;result->equity_curve[j]=equity;result->equity_timestamps[j]=data->events[i].timestamp_ns;}
        }
    }
    qsort(w.latencies,w.sample_count,sizeof(double),compare_double);
    result->latency_us[0]=w.latencies[(w.sample_count-1)*50/100];result->latency_us[1]=w.latencies[(w.sample_count-1)*95/100];result->latency_us[2]=w.latencies[(w.sample_count-1)*99/100];
#ifdef _WIN32
    {PROCESS_MEMORY_COUNTERS memory;if(GetProcessMemoryInfo(GetCurrentProcess(),&memory,sizeof(memory)))result->peak_rss_bytes=memory.PeakWorkingSetSize;}
#else
    {struct rusage usage;if(!getrusage(RUSAGE_SELF,&usage)){result->peak_rss_bytes=(size_t)usage.ru_maxrss*1024;result->voluntary_switches=usage.ru_nvcsw;result->involuntary_switches=usage.ru_nivcsw;}}
#endif
    if(w.failed)snprintf(result->error,sizeof(result->error),"Worker memory allocation or thread/process execution failed");
    goto cleanup;
allocation_locked:snprintf(result->error,sizeof(result->error),"Insufficient memory for native work buffers");
failure_locked:w.failed=1;
cleanup:
#ifdef _WIN32
    DeleteCriticalSection(&w.mutex);
#else
    pthread_mutex_destroy(&w.mutex);
#endif
    free(w.features);free(w.probabilities);free(w.signals);free(w.latencies);free(w.equity_deltas);free(w.indices);free(w.offsets);
    return !w.failed;
allocation:
    snprintf(result->error,sizeof(result->error),"Insufficient memory for native workload");
    free(w.offsets);free(w.indices);free(w.latencies);free(w.equity_deltas);return 0;
}
