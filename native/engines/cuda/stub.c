#include "hft_engine.h"
#include <stdio.h>
int hft_cuda_available(char *name,size_t capacity){(void)name;(void)capacity;return 0;}
int hft_cuda_features(const HftEvent *events,size_t count,int instruments,const HftModel *model,double *features,double *probabilities,HftResult *timing){
    (void)events;(void)count;(void)instruments;(void)model;(void)features;(void)probabilities;
    snprintf(timing->error,sizeof(timing->error),"CUDA was not compiled; configure -DHFT_CUDA=ON with nvcc");return 0;
}
