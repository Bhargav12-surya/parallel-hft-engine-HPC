/* Run explicitly on an available OpenCL device. CPU OpenCL is labeled as such. */
#include "hft/api.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
static int errors;
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x);++errors;}}while(0)
int main(void){HftCapability cap;hft_opencl_probe(&cap);if(!cap.available){puts(cap.reason);return 77;}printf("Device: %s (%s)\n",cap.name,cap.reason);HftConfig c;hft_config_default(&c);c.symbols=4;c.events=500;c.batch=64;c.block_size=2;for(int s=0;s<13;++s){HftResult cpu={0},gpu={0};c.strategy=s;c.engine=HFT_SERIAL;CHECK(hft_run(&c,&cpu,NULL,NULL,NULL));c.engine=HFT_OPENCL;c.pinned=s%2;c.streams=s%3;int ok=hft_run(&c,&gpu,NULL,NULL,NULL);CHECK(ok);if(!ok)fprintf(stderr,"%s\n",gpu.error);else CHECK(hft_result_equivalent(&cpu,&gpu));printf("%-15s %s\n",hft_strategy_names[s],ok&&hft_result_equivalent(&cpu,&gpu)?"MATCH":"FAIL");hft_result_free(&cpu);hft_result_free(&gpu);}
 float input[64*32],output[64*3],expected[3],cov[32*32];double x[64*32],reference[32*32],seconds;char error[HFT_ERROR]={0};for(int i=0;i<64*32;++i)x[i]=input[i]=(float)sin(i*.13);CHECK(hft_covariance(x,64,32,reference,0,0));CHECK(hft_opencl_matrix(input,64,32,cov,1,&seconds,error));for(int i=0;i<32*32;++i)CHECK(fabs(cov[i]-reference[i])<1e-5);CHECK(hft_opencl_infer(&c.model,input,64,output,&seconds,error));for(int i=0;i<64;++i){hft_mlp_predict(&c.model,input+i*32,expected);for(int j=0;j<3;++j)CHECK(fabs(output[i*3+j]-expected[j])<1e-5);}
 int labels[64];for(int i=0;i<64;++i)labels[i]=i%3;HftDataset d={input,labels,64,48};HftModel m=c.model,n=c.model;double loss,accuracy;CHECK(hft_mlp_train(&m,&d,2,1,&loss,&accuracy,NULL));CHECK(hft_opencl_train(&n,input,labels,48,2,&seconds,error));for(int i=0;i<HFT_WEIGHTS;++i)CHECK(fabs(m.weights[i]-n.weights[i])<1e-5);CHECK(hft_opencl_qtrain(&n,2,100,42,&seconds,error));CHECK(n.q_trained);printf("Device regression: %d failures\n",errors);return errors?1:0;}
