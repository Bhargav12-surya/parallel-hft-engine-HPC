#include "hft_engine.h"
#include <stdio.h>
#include <inttypes.h>
#include <string.h>
#include <stdlib.h>
#ifdef _WIN32
extern int hft_atomic_replace(const char*,const char*);
#endif
void hft_capabilities(void){
    int openmp=0,mpi=0;
#ifdef HFT_HAVE_OPENMP
    openmp=1;
#endif
#ifdef HFT_HAVE_MPI
    mpi=1;
#endif
    char device[256]={0};int cuda=hft_cuda_available(device,sizeof(device));
    /* JSON booleans describe compiled and usable backends, not installed Python
     * packages. CUDA execution still validates every runtime API call. */
    printf("{\"schema_version\":1,\"engines\":{\"sequential\":true,\"threaded\":true,\"openmp\":%s,\"mpi\":%s,\"cuda\":%s,\"hybrid\":%s},\"logical_cpus\":%d,\"model_format\":\"HFTMODEL 1\",\"features\":6,\"cuda_device\":\"%s\",\"compiler\":\"",
      openmp?"true":"false",mpi?"true":"false",cuda?"true":"false",openmp||mpi||cuda?"true":"false",hft_logical_cpus(),device);
#ifdef _MSC_VER
    printf("MSVC %d",_MSC_VER);
#else
    printf("%s",__VERSION__);
#endif
    printf("\"}\n");
}
int hft_result_write(const char *path,const HftConfig *config,const HftResult *r){
    size_t n=strlen(path);if(n>4096)return 0;
    char *temporary=malloc(n+16);if(!temporary)return 0;
    snprintf(temporary,n+16,"%s.native.tmp",path);
    FILE *file=fopen(temporary,"wb");if(!file){free(temporary);return 0;}
    uint64_t events=0,trades=0,orders=0,rejected=0,cancels=0,checksum=UINT64_C(14695981039346656037),time=0;
    double pnl=0,equity=0,fees=0,drawdown=0,realized=0,unrealized=0,gross=0,slippage=0;
    for(size_t i=0;i<r->instrument_count;++i){const HftInstrumentResult *s=&r->instruments[i];events+=s->events;trades+=s->trades;orders+=s->orders;rejected+=s->rejected;cancels+=s->cancels;
        checksum^=s->checksum;checksum*=UINT64_C(1099511628211);pnl+=s->pnl;equity+=s->equity;fees+=s->fees;realized+=s->realized;unrealized+=s->unrealized;gross+=s->gross_exposure;slippage+=s->slippage;
        if(s->simulated_time_ns>time)time=s->simulated_time_ns;}
    fprintf(file,"{\n\"schema_version\":1,\"status\":\"COMPLETED\",\"engine\":\"%s\",\"cpu_mode\":\"%s\",\"strategy\":\"%s\",\n",config->engine,config->cpu_mode,config->strategy);
    fprintf(file,"\"events_processed\":%" PRIu64 ",\"trades\":%" PRIu64 ",\"orders\":%" PRIu64 ",\"rejected\":%" PRIu64 ",\"cancels\":%" PRIu64 ",\"simulated_time_ns\":%" PRIu64 ",\n",events,trades,orders,rejected,cancels,time);
    drawdown=r->portfolio_drawdown;
    fprintf(file,"\"wall_seconds\":%.17g,\"throughput\":%.17g,\"pnl\":%.17g,\"equity\":%.17g,\"fees\":%.17g,\"drawdown\":%.17g,\"realized_pnl\":%.17g,\"unrealized_pnl\":%.17g,\"gross_exposure\":%.17g,\"slippage\":%.17g,\"checksum\":\"%016" PRIx64 "\",\n",r->wall_seconds,r->wall_seconds>0?events/r->wall_seconds:0,pnl,equity,fees,drawdown,realized,unrealized,gross,slippage,checksum);
    fprintf(file,"\"latency_us\":{\"p50\":%.17g,\"p95\":%.17g,\"p99\":%.17g,\"sampled\":true,\"scope\":\"CPU event matching and decision; GPU preprocessing excluded\"},\n",r->latency_us[0],r->latency_us[1],r->latency_us[2]);
    fprintf(file,"\"profiling\":{\"load_seconds\":%.17g,\"compute_seconds\":%.17g,\"feature_seconds\":%.17g,\"matching_seconds\":%.17g,\"mpi_communication_seconds\":%.17g,\"cuda_h2d_seconds\":%.17g,\"cuda_kernel_seconds\":%.17g,\"cuda_d2h_seconds\":%.17g,\"bytes_transferred\":%" PRIu64 ",\"peak_rss_bytes\":%zu,\"workers\":%d,\"ranks\":%d,\"affinity_applied_workers\":%d,",r->load_seconds,r->compute_seconds,r->feature_seconds,r->match_seconds,r->communication_seconds,r->h2d_seconds,r->kernel_seconds,r->d2h_seconds,r->bytes_transferred,r->peak_rss_bytes,r->workers,r->ranks,r->affinity_applied);
#ifdef _WIN32
    fprintf(file,"\"voluntary_context_switches\":null,\"involuntary_context_switches\":null},\n");
#else
    fprintf(file,"\"voluntary_context_switches\":%ld,\"involuntary_context_switches\":%ld},\n",r->voluntary_switches,r->involuntary_switches);
#endif
    fprintf(file,"\"equity_curve\":[");for(size_t i=0;i<r->equity_samples;++i)fprintf(file,"%s{\"timestamp_ns\":%" PRIu64 ",\"equity\":%.17g,\"pnl\":%.17g}",i?",":"",r->equity_timestamps[i],r->equity_curve[i],r->equity_curve[i]-config->capital);fprintf(file,"],\n");
    fprintf(file,"\"instruments\":[");
    for(size_t i=0;i<r->instrument_count;++i){const HftInstrumentResult *s=&r->instruments[i];fprintf(file,"%s{\"instrument_id\":%zu,\"events\":%" PRIu64 ",\"position\":%" PRId64 ",\"pnl\":%.17g,\"equity\":%.17g,\"trades\":%" PRIu64 ",\"fees\":%.17g,\"drawdown\":%.17g}",i?",":"",i,s->events,s->position,s->pnl,s->equity,s->trades,s->fees,s->drawdown);}
    fprintf(file,"]\n}\n");
    int ok=!ferror(file);if(fclose(file))ok=0;
    if(ok){
#ifdef _WIN32
        /* MoveFileEx supports atomic replacement on the same filesystem. */
        ok=hft_atomic_replace(temporary,path);
#else
        ok=rename(temporary,path)==0;
#endif
    }
    if(!ok)remove(temporary);free(temporary);return ok;
}
