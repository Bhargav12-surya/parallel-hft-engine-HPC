#include "hft_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <limits.h>
#include <inttypes.h>
#ifdef HFT_HAVE_MPI
#include <mpi.h>
#endif
static void error_json(const char *message) {
    fprintf(stderr,"HFT-LAB: %s\n",message);
    fputs("{\"schema_version\":1,\"type\":\"error\",\"message\":\"",stdout);
    for(const unsigned char *p=(const unsigned char*)message;*p;++p){if(*p=='"'||*p=='\\')putchar('\\');if(*p>=32)putchar(*p);}
    fputs("\"}\n",stdout);
}
static int integer(const char *text,int minimum,int maximum,int *result){
    char *end;errno=0;long n=strtol(text,&end,10);if(errno||*end||n<minimum||n>maximum)return 0;*result=(int)n;return 1;
}
static int real_number(const char *text,double minimum,double maximum,double *result){
    char *end;errno=0;double n=strtod(text,&end);if(errno||*end||!isfinite(n)||n<minimum||n>maximum)return 0;*result=n;return 1;
}
static int one_of(const char *text,const char *const *options){for(int j=0;options[j];++j)if(!strcmp(text,options[j]))return 1;return 0;}
static int infer(const char *model_path,const char *features_path){
    HftModel model;char error[HFT_ERROR_SIZE];
    if(!model_path||!features_path){error_json("--infer requires --model and --features");return 2;}
    if(!hft_model_load(model_path,&model,error)){error_json(error);return 2;}
    FILE *file=fopen(features_path,"rb");if(!file){error_json("Cannot open inference feature file");return 2;}
    double *probabilities=NULL;size_t count=0,capacity=0;int status=0;
    for(;;){
        double features[HFT_FEATURES],p[3];int first=fscanf(file,"%lf",&features[0]);
        if(first==EOF)break;
        if(first!=1||!isfinite(features[0])||fabs(features[0])>1e20){status=2;break;}
        for(int j=1;j<HFT_FEATURES;++j)if(fscanf(file,"%lf",&features[j])!=1||!isfinite(features[j])||fabs(features[j])>1e20){status=2;break;}
        if(status)break;
        if(count==capacity){size_t n=capacity?capacity*2:64;if(n<capacity||n>SIZE_MAX/(3*sizeof(double))){status=3;break;}double *next=realloc(probabilities,n*3*sizeof(double));if(!next){status=3;break;}probabilities=next;capacity=n;}
        hft_model_predict(&model,features,p);memcpy(probabilities+3*count,p,3*sizeof(double));++count;
    }
    if(ferror(file))status=3;fclose(file);
    if(status){free(probabilities);error_json("Malformed features or inference memory/I/O error");return status;}
    printf("{\"schema_version\":1,\"probabilities\":[");
    for(size_t i=0;i<count;++i)printf("%s[%.17g,%.17g,%.17g]",i?",":"",probabilities[3*i],probabilities[3*i+1],probabilities[3*i+2]);
    printf("]}\n");free(probabilities);return 0;
}
int main(int argc,char **argv){
    HftConfig config;memset(&config,0,sizeof(config));
    config.engine="sequential";config.cpu_mode="single-single";config.strategy="momentum";config.components="cpu,openmp";
    config.capital=100000;config.fee_bps=1;config.position_limit=100;config.max_order=10;config.max_loss=10000;config.tick_value=.01;
    const char *features_path=NULL;int inference=0;
    for(int i=1;i<argc;++i){const char *arg=argv[i];
        if(!strcmp(arg,"--capabilities")){hft_capabilities();return 0;}
        if(!strcmp(arg,"--infer")){inference=1;continue;}
        if(!strcmp(arg,"--quiet")){config.quiet=1;continue;}
        if(!strcmp(arg,"--help")){puts("HFT-LAB native runner v1\n--input canonical.csv --output result.json --engine sequential|threaded|openmp|mpi|cuda|hybrid\n--cpu-mode single-single|single-multi|multi-single|multi-multi --threads N\n--strategy hold|market-making|momentum|mean-reversion|stat-arb|ai --model model.hftmodel\n--capital N --fee-bps N --slippage-ticks N --latency-ns N --position-limit N --max-order N --max-loss N --tick-value N\n--components cpu,openmp,mpi,cuda --capabilities\n--infer --model model.hftmodel --features features.txt");return 0;}
        if(i+1>=argc){error_json("Missing argument value");return 2;}const char *value=argv[++i];int valid=1;
        if(!strcmp(arg,"--input"))config.input=value;
        else if(!strcmp(arg,"--output"))config.output=value;
        else if(!strcmp(arg,"--engine"))config.engine=value;
        else if(!strcmp(arg,"--cpu-mode"))config.cpu_mode=value;
        else if(!strcmp(arg,"--strategy"))config.strategy=value;
        else if(!strcmp(arg,"--model"))config.model_path=value;
        else if(!strcmp(arg,"--features"))features_path=value;
        else if(!strcmp(arg,"--components"))config.components=value;
        else if(!strcmp(arg,"--threads"))valid=integer(value,1,1024,&config.threads);
        else if(!strcmp(arg,"--position-limit"))valid=integer(value,1,100000000,&config.position_limit);
        else if(!strcmp(arg,"--max-order"))valid=integer(value,1,100000000,&config.max_order);
        else if(!strcmp(arg,"--slippage-ticks"))valid=integer(value,0,1000000,&config.slippage_ticks);
        else if(!strcmp(arg,"--latency-ns")){char *end;errno=0;unsigned long long n=strtoull(value,&end,10);valid=!errno&&!*end&&*value!='-'&&n<=UINT64_C(1000000000000);config.latency_ns=(uint64_t)n;}
        else if(!strcmp(arg,"--capital"))valid=real_number(value,.01,1e15,&config.capital);
        else if(!strcmp(arg,"--fee-bps"))valid=real_number(value,0,10000,&config.fee_bps);
        else if(!strcmp(arg,"--max-loss"))valid=real_number(value,0,1e15,&config.max_loss);
        else if(!strcmp(arg,"--tick-value"))valid=real_number(value,1e-12,1e6,&config.tick_value);
        else{error_json("Unknown native argument");return 2;}
        if(!valid){error_json("Argument value outside supported range");return 2;}
    }
    if(inference)return infer(config.model_path,features_path);
    const char *engines[]={"sequential","threaded","openmp","mpi","cuda","hybrid",NULL};
    const char *modes[]={"single-single","single-multi","multi-single","multi-multi",NULL};
    const char *strategies[]={"hold","market-making","momentum","mean-reversion","stat-arb","ai",NULL};
    if(!config.input||!config.output||!one_of(config.engine,engines)||!one_of(config.cpu_mode,modes)||!one_of(config.strategy,strategies)){
        error_json("Invalid engine/CPU mode/strategy or missing --input/--output");return 2;}
    if(!strcmp(config.engine,"hybrid")){
        const char *valid_components[]={"cpu","openmp","mpi","cuda",NULL};char copy[128];
        if(strlen(config.components)>=sizeof(copy)){error_json("Hybrid component list is too long");return 2;}
        strcpy(copy,config.components);int seen[4]={0},n=0;char *part=strtok(copy,",");
        while(part){int found=-1;for(int j=0;j<4;++j)if(!strcmp(part,valid_components[j]))found=j;
            if(found<0||seen[found]){error_json("Unknown or repeated hybrid component");return 2;}seen[found]=1;++n;part=strtok(NULL,",");}
        if(n<2){error_json("Hybrid requires two or more components");return 2;}
    }
    HftDataset data;HftResult result;HftModel model;HftModel *model_pointer=NULL;
    memset(&data,0,sizeof(data));memset(&result,0,sizeof(result));
    int rank=0,ranks=1,mpi=!strcmp(config.engine,"mpi")||(!strcmp(config.engine,"hybrid")&&strstr(config.components,"mpi"));
#ifndef HFT_HAVE_MPI
    if(mpi){error_json("MPI engine unavailable: rebuild with an MPI C implementation");return 2;}
#else
    if(mpi){if(MPI_Init(&argc,&argv)!=MPI_SUCCESS){error_json("MPI_Init failed");return 3;}MPI_Comm_set_errhandler(MPI_COMM_WORLD,MPI_ERRORS_RETURN);MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&ranks);}
#endif
    int ok=1;double start=hft_now();
    if(rank==0)ok=hft_dataset_load(config.input,&data,result.error);
#ifdef HFT_HAVE_MPI
    if(mpi){
        double begin=hft_now();MPI_Bcast(&ok,1,MPI_INT,0,MPI_COMM_WORLD);
        uint64_t dimensions[2]={(uint64_t)data.count,(uint64_t)data.instruments};
        if(ok){MPI_Bcast(dimensions,2,MPI_UINT64_T,0,MPI_COMM_WORLD);if(rank){data.count=(size_t)dimensions[0];data.instruments=(size_t)dimensions[1];if(data.count>SIZE_MAX/sizeof(HftEvent))ok=0;else data.events=malloc(data.count*sizeof(HftEvent));if(!data.events)ok=0;}
            int all_ok=0;MPI_Allreduce(&ok,&all_ok,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);ok=all_ok;
            if(ok){size_t bytes=data.count*sizeof(HftEvent),offset=0;while(bytes){int chunk=bytes>100000000?100000000:(int)bytes;if(MPI_Bcast((char*)data.events+offset,chunk,MPI_BYTE,0,MPI_COMM_WORLD)!=MPI_SUCCESS){ok=0;break;}bytes-=(size_t)chunk;offset+=(size_t)chunk;}result.bytes_transferred+=(uint64_t)data.count*sizeof(HftEvent)*(uint64_t)(ranks-1);}}
        result.communication_seconds+=hft_now()-begin;
    }
#endif
    result.load_seconds=hft_now()-start;
    if(ok&&!strcmp(config.strategy,"stat-arb")&&data.instruments<2){snprintf(result.error,sizeof(result.error),"Statistical arbitrage requires at least two instruments");ok=0;}
    if(ok&&!strcmp(config.strategy,"ai")){
        if(!config.model_path){snprintf(result.error,sizeof(result.error),"AI strategy requires --model");ok=0;}
        else{ok=hft_model_load(config.model_path,&model,result.error);model_pointer=&model;}
    }
#ifdef HFT_HAVE_MPI
    if(mpi){int all_ok=0;MPI_Allreduce(&ok,&all_ok,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);ok=all_ok;}
#endif
    if(ok)ok=hft_run(&config,&data,model_pointer,&result);
    if(ok&&rank==0)ok=hft_result_write(config.output,&config,&result);
    if(rank==0){
        if(ok)puts("{\"schema_version\":1,\"type\":\"result\",\"status\":\"COMPLETED\"}");
        else error_json(*result.error?result.error:"Native execution or output write failed");
    }
    hft_dataset_free(&data);free(result.instruments);
#ifdef HFT_HAVE_MPI
    if(mpi)MPI_Finalize();
#endif
    return ok?0:3;
}
