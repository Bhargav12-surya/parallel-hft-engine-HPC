/* Independent event features use a predecessor index: no event reordering or
 * dependent exchange matching occurs on the GPU. One CUDA thread per event. */
#include "hft_engine.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <climits>
static __global__ void feature_kernel(const double *prices,const int64_t *previous,
    const int *sides,const int *quantities,size_t count,const HftModel *model,
    double *features,double *probabilities) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    double f[HFT_FEATURES]={0},ret[16]={0};double price=prices[i];
    int64_t predecessor=previous[i];
    f[0]=predecessor>=0?(price-prices[predecessor])/prices[predecessor]:0;
    int64_t cursor=(int64_t)i;int samples=0;
    for(int k=0;k<16&&cursor>=0;++k){
        int64_t p=previous[cursor];
        ret[k]=p>=0?(prices[cursor]-prices[p])/prices[p]:0;++samples;cursor=p;
        if(k==3&&cursor>=0)f[1]=(price-prices[cursor])/prices[cursor];
        if(k==15&&cursor>=0)f[2]=(price-prices[cursor])/prices[cursor];
    }
    double mean=0,variance=0;for(int k=0;k<samples;++k)mean+=ret[k];mean/=samples;
    for(int k=0;k<samples;++k){double d=ret[k]-mean;variance+=d*d;}
    f[3]=sqrt(variance/samples);f[4]=(sides[i]?-1.0:1.0)*quantities[i]/(quantities[i]+100.0);f[5]=2/price;
    for(int k=0;k<HFT_FEATURES;++k)features[i*HFT_FEATURES+k]=f[k];
    if(!model)return;
    double x[HFT_FEATURES],hidden[HFT_HIDDEN_MAX];
    for(int k=0;k<HFT_FEATURES;++k)x[k]=(f[k]-model->mean[k])/model->scale[k];
    for(int h=0;h<model->hidden;++h){double value=model->b1[h];for(int k=0;k<HFT_FEATURES;++k)value+=model->w1[h*HFT_FEATURES+k]*x[k];hidden[h]=value>0?value:0;}
    int width=model->hidden?model->hidden:HFT_FEATURES;const double *input=model->hidden?hidden:x;
    double logits[3],maximum=-1e300;for(int c=0;c<3;++c){double value=model->b2[c];for(int k=0;k<width;++k)value+=model->w2[c*width+k]*input[k];logits[c]=value;if(value>maximum)maximum=value;}
    double sum=0;for(int c=0;c<3;++c){logits[c]=exp(logits[c]-maximum);sum+=logits[c];}
    for(int c=0;c<3;++c)probabilities[i*3+c]=logits[c]/sum;
}
extern "C" int hft_cuda_available(char *name,size_t capacity){
    int count=0;if(cudaGetDeviceCount(&count)!=cudaSuccess||count<1)return 0;
    cudaDeviceProp properties;if(cudaGetDeviceProperties(&properties,0)!=cudaSuccess)return 0;
    if(name&&capacity)std::snprintf(name,capacity,"%s",properties.name);return 1;
}
extern "C" int hft_cuda_features(const HftEvent *events,size_t count,int instruments,
    const HftModel *model,double *features,double *probabilities,HftResult *timing){
    if(!count)return 1;
    double *prices=(double*)std::malloc(count*sizeof(double));
    int64_t *previous=(int64_t*)std::malloc(count*sizeof(int64_t)),*last=(int64_t*)std::malloc((size_t)instruments*sizeof(int64_t));
    double *marks=(double*)std::calloc((size_t)instruments,sizeof(double));
    int *sides=(int*)std::malloc(count*sizeof(int)),*quantities=(int*)std::malloc(count*sizeof(int));
    double *dp=NULL,*df=NULL,*dprob=NULL;int64_t *dprevious=NULL;int *ds=NULL,*dq=NULL;HftModel *dm=NULL;
    cudaEvent_t begin=NULL,end=NULL;cudaError_t status=cudaSuccess;float milliseconds=0;int ok=0;
    if(!prices||!previous||!last||!marks||!sides||!quantities){std::snprintf(timing->error,sizeof(timing->error),"CUDA host allocation failed");goto cleanup;}
    for(int s=0;s<instruments;++s)last[s]=-1;
    for(size_t i=0;i<count;++i){int s=events[i].instrument;previous[i]=last[s];last[s]=(int64_t)i;if(events[i].type!=HFT_CANCEL&&events[i].price>0)marks[s]=(double)events[i].price;prices[i]=marks[s]>0?marks[s]:10000;sides[i]=events[i].side;quantities[i]=events[i].quantity;}
#define CUDA_CHECK(call) do{status=(call);if(status!=cudaSuccess)goto cuda_error;}while(0)
    CUDA_CHECK(cudaEventCreate(&begin));CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaMalloc((void**)&dp,count*sizeof(double)));CUDA_CHECK(cudaMalloc((void**)&dprevious,count*sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc((void**)&ds,count*sizeof(int)));CUDA_CHECK(cudaMalloc((void**)&dq,count*sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&df,count*HFT_FEATURES*sizeof(double)));CUDA_CHECK(cudaMalloc((void**)&dprob,count*3*sizeof(double)));
    if(model){CUDA_CHECK(cudaMalloc((void**)&dm,sizeof(HftModel)));}
    CUDA_CHECK(cudaEventRecord(begin));
    CUDA_CHECK(cudaMemcpy(dp,prices,count*sizeof(double),cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(dprevious,previous,count*sizeof(int64_t),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ds,sides,count*sizeof(int),cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(dq,quantities,count*sizeof(int),cudaMemcpyHostToDevice));
    if(model){CUDA_CHECK(cudaMemcpy(dm,model,sizeof(HftModel),cudaMemcpyHostToDevice));}
    CUDA_CHECK(cudaEventRecord(end));CUDA_CHECK(cudaEventSynchronize(end));CUDA_CHECK(cudaEventElapsedTime(&milliseconds,begin,end));timing->h2d_seconds+=milliseconds*.001;
    CUDA_CHECK(cudaEventRecord(begin));
    feature_kernel<<<(unsigned int)((count+255)/256),256>>>(dp,dprevious,ds,dq,count,dm,df,dprob);
    CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaEventRecord(end));CUDA_CHECK(cudaEventSynchronize(end));CUDA_CHECK(cudaEventElapsedTime(&milliseconds,begin,end));timing->kernel_seconds+=milliseconds*.001;
    CUDA_CHECK(cudaEventRecord(begin));
    CUDA_CHECK(cudaMemcpy(features,df,count*HFT_FEATURES*sizeof(double),cudaMemcpyDeviceToHost));
    if(model){CUDA_CHECK(cudaMemcpy(probabilities,dprob,count*3*sizeof(double),cudaMemcpyDeviceToHost));}
    CUDA_CHECK(cudaEventRecord(end));CUDA_CHECK(cudaEventSynchronize(end));CUDA_CHECK(cudaEventElapsedTime(&milliseconds,begin,end));timing->d2h_seconds+=milliseconds*.001;
    timing->bytes_transferred+=(uint64_t)count*(sizeof(double)+sizeof(int64_t)+2*sizeof(int)+HFT_FEATURES*sizeof(double)+(model?3*sizeof(double):0))+(model?sizeof(HftModel):0);
    ok=1;goto cleanup;
cuda_error:std::snprintf(timing->error,sizeof(timing->error),"CUDA error: %s",cudaGetErrorString(status));
cleanup:
    if(begin)cudaEventDestroy(begin);if(end)cudaEventDestroy(end);
    cudaFree(dp);cudaFree(dprevious);cudaFree(ds);cudaFree(dq);cudaFree(df);cudaFree(dprob);cudaFree(dm);
    std::free(prices);std::free(previous);std::free(last);std::free(marks);std::free(sides);std::free(quantities);
    return ok;
}
