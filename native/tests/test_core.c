/* Native tests intentionally use explicit checks: release NDEBUG must never
 * compile correctness assertions out of the test executable. */
#include "hft_engine.h"
#include "hft_orderbook.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#x);return 1;}}while(0)
typedef struct{int count,quantity;uint64_t first,second;} Fills;
static void fill(void *context,const HftOrder *maker,const HftOrder *taker,int64_t price,int quantity){
    Fills *f=context;if(f->count==0)f->first=maker->id;if(f->count==1)f->second=maker->id;++f->count;f->quantity+=quantity;(void)taker;(void)price;
}
static HftOrder order(uint64_t id,int64_t price,int quantity,int side){HftOrder o;memset(&o,0,sizeof(o));o.id=id;o.price=price;o.quantity=quantity;o.side=side;return o;}
static int book_test(void){
    HftBook b;Fills fills={0};CHECK(hft_book_init(&b));
    CHECK(hft_book_submit(&b,order(1,101,7,1),0,fill,&fills)==1);
    CHECK(hft_book_submit(&b,order(2,101,9,1),0,fill,&fills)==1);
    CHECK(hft_book_submit(&b,order(3,102,5,1),0,fill,&fills)==1);
    CHECK(hft_book_submit(&b,order(4,101,11,0),0,fill,&fills)==1);
    CHECK(fills.count==2&&fills.quantity==11&&fills.first==1&&fills.second==2);
    CHECK(hft_book_best(&b,1)->id==2&&hft_book_best(&b,1)->quantity==5);
    CHECK(hft_book_submit(&b,order(2,99,1,0),0,fill,&fills)==0);
    CHECK(hft_book_cancel(&b,2));CHECK(!hft_book_cancel(&b,2));
    CHECK(hft_book_best(&b,1)->id==3);
    CHECK(hft_book_submit(&b,order(5,99,8,0),0,fill,&fills)==1);
    CHECK(hft_book_best(&b,0)->id==5);
    CHECK(hft_book_submit(&b,order(6,1,10,0),1,fill,&fills)==1);
    CHECK(!hft_book_best(&b,1));CHECK(hft_book_validate(&b));
    CHECK(hft_book_submit(&b,order(7,-1,1,0),0,fill,&fills)==0);
    CHECK(hft_book_submit(&b,order(7,1,0,0),0,fill,&fills)==0);
    for(uint64_t i=100;i<10100;++i)CHECK(hft_book_submit(&b,order(i,(int64_t)i,1,0),0,NULL,NULL)==1);
    for(uint64_t i=100;i<10100;i+=2)CHECK(hft_book_cancel(&b,i));
    CHECK(hft_book_validate(&b));
    for(uint64_t i=20000;i<30000;++i)CHECK(hft_book_submit(&b,order(i,(int64_t)i,1,1),0,NULL,NULL)==1);
    CHECK(hft_book_validate(&b));hft_book_destroy(&b);return 0;
}
static int portfolio_test(void){
    HftPortfolio p;memset(&p,0,sizeof(p));p.cash=1000;
    hft_portfolio_fill(&p,0,10000,10,1,.01);
    CHECK(fabs(p.cash+0.1)<1e-9&&p.position==10&&fabs(p.fees-.1)<1e-12);
    hft_portfolio_fill(&p,1,10100,4,1,.01);
    CHECK(p.position==6&&fabs(p.realized-4)<1e-12&&fabs(p.cost-600)<1e-12);
    hft_portfolio_fill(&p,1,9900,8,0,.01);
    CHECK(p.position==-2&&fabs(p.realized+2)<1e-12&&fabs(p.cost-198)<1e-12);
    hft_portfolio_fill(&p,0,9800,2,0,.01);
    CHECK(p.position==0&&fabs(p.realized)<1e-12&&fabs(p.cost)<1e-12);return 0;
}
static int feature_test(void){
    HftFeatures state;memset(&state,0,sizeof(state));double f[HFT_FEATURES];
    hft_features_update(&state,100,0,100,f);
    CHECK(f[0]==0&&f[1]==0&&f[2]==0&&f[3]==0&&f[4]==.5&&f[5]==.02);
    hft_features_update(&state,101,1,100,f);CHECK(fabs(f[0]-.01)<1e-15&&fabs(f[3]-.005)<1e-15&&f[4]==-.5);
    for(int i=2;i<=16;++i)hft_features_update(&state,100+i,0,100,f);
    CHECK(fabs(f[2]-.16)<1e-15&&fabs(f[1]-4.0/112.0)<1e-15);
    hft_features_update(&state,0,0,0,f);CHECK(f[0]==0&&f[4]==0);return 0;
}
static int model_test(void){
    HftModel model;memset(&model,0,sizeof(model));for(int j=0;j<HFT_FEATURES;++j)model.scale[j]=1;
    model.w2[0]=1;model.w2[HFT_FEATURES]=-1;model.w2[2*HFT_FEATURES]=0;
    double features[HFT_FEATURES]={1000,0,0,0,0,0},p[3];hft_model_predict(&model,features,p);
    CHECK(isfinite(p[0])&&p[0]>.999&&p[0]+p[1]+p[2]>.999999999999);
    model.hidden=2;memset(model.w2,0,sizeof(model.w2));model.w1[0]=1;model.w1[HFT_FEATURES]=-1;model.w2[0]=1;model.w2[3]=1;
    features[0]=-10;hft_model_predict(&model,features,p);CHECK(p[1]>.99);return 0;
}
static int parser_test(void){
    const char *path="hftlab-native-parser-fixture.csv";
    const char *header="event_id,timestamp_ns,instrument_id,event_type,side,price_ticks,quantity,order_id,reference_order_id\n";
    FILE *f=fopen(path,"wb");CHECK(f);fputs(header,f);
    fputs("9,100,0,QUOTE,BUY,10000,100,0,0\n3,100,0,QUOTE,SELL,10001,100,0,0\n",f);fclose(f);
    HftDataset data;char error[HFT_ERROR_SIZE];CHECK(hft_dataset_load(path,&data,error));
    CHECK(data.count==2&&data.events[0].id==9&&data.events[1].id==3);hft_dataset_free(&data);
    f=fopen(path,"wb");CHECK(f);fputs(header,f);fputs("1,100,0,QUOTE,BUY,10000,100,0,0\n1,101,0,QUOTE,BUY,10000,100,0,0\n",f);fclose(f);
    CHECK(!hft_dataset_load(path,&data,error));
    f=fopen(path,"wb");CHECK(f);fputs(header,f);fputs("1,100,0,QUOTE,BUY,10000,100,0,0\n2,99,0,QUOTE,BUY,10000,100,0,0\n",f);fclose(f);
    CHECK(!hft_dataset_load(path,&data,error));
    f=fopen(path,"wb");CHECK(f);fputs(header,f);fputs("1,100,0,QUOTE,BUY,-1,100,0,0\n",f);fclose(f);
    CHECK(!hft_dataset_load(path,&data,error));remove(path);return 0;
}
static int simulation_test(void){
    HftEvent events[128];memset(events,0,sizeof(events));
    for(int i=0;i<128;++i){int step=i/2;events[i].id=(uint64_t)i+1;events[i].timestamp_ns=(uint64_t)i*1000;events[i].instrument=i%2;events[i].type=HFT_QUOTE;events[i].price=10000+(step<32?step:64-step)*10;events[i].quantity=100;events[i].side=i%2;}
    HftDataset data={events,128,2};HftConfig c;memset(&c,0,sizeof(c));c.engine="sequential";c.cpu_mode="multi-single";c.strategy="momentum";c.capital=100000;c.max_order=10;c.position_limit=20;c.fee_bps=1;c.tick_value=.01;c.max_loss=10000;c.threads=2;c.quiet=1;c.components="cpu,openmp";
    HftResult baseline;memset(&baseline,0,sizeof(baseline));CHECK(hft_run(&c,&data,NULL,&baseline));
    CHECK(baseline.instruments[0].trades>0&&baseline.instruments[0].fees>0);
    CHECK(llabs(baseline.instruments[0].position)<=c.position_limit);
    double peak=c.capital,drawdown=0;
    for(size_t i=0;i<baseline.equity_samples;++i){double equity=baseline.equity_curve[i];if(equity>peak)peak=equity;if(peak-equity>drawdown)drawdown=peak-equity;}
    CHECK(fabs(drawdown-baseline.portfolio_drawdown)<1e-9);
    const char *backends[]={"threaded",
#ifdef HFT_HAVE_OPENMP
        "openmp","hybrid",
#endif
        NULL};
    for(int engine=0;backends[engine];++engine){
        c.engine=backends[engine];c.cpu_mode="multi-multi";HftResult actual;memset(&actual,0,sizeof(actual));CHECK(hft_run(&c,&data,NULL,&actual));
        for(int j=0;j<2;++j){CHECK(actual.instruments[j].checksum==baseline.instruments[j].checksum);CHECK(fabs(actual.instruments[j].pnl-baseline.instruments[j].pnl)<1e-12);CHECK(actual.instruments[j].events==64);}
        CHECK(fabs(actual.portfolio_drawdown-baseline.portfolio_drawdown)<1e-12);free(actual.instruments);
    }
    free(baseline.instruments);c.engine="sequential";c.cpu_mode="multi-single";c.strategy="hold";memset(&baseline,0,sizeof(baseline));CHECK(hft_run(&c,&data,NULL,&baseline));
    CHECK(baseline.instruments[0].trades==0&&baseline.instruments[0].pnl==0&&baseline.portfolio_drawdown==0);free(baseline.instruments);return 0;
}
int main(void){CHECK(!book_test());CHECK(!portfolio_test());CHECK(!feature_test());CHECK(!model_test());CHECK(!parser_test());CHECK(!simulation_test());puts("PASS book priority/partial/cancel/growth, accounting, causal AI, model stability, parser robustness, CPU/OpenMP/hybrid replay, chronological drawdown");return 0;}
