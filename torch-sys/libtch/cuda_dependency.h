#ifndef __TCH_CUDA_DEPENDENCY_H__
#define __TCH_CUDA_DEPENDENCY_H__

#include<stdio.h>
#include<stdint.h>

#ifdef __cplusplus
#include<ATen/cuda/CUDAGraph.h>
extern "C" {
typedef at::cuda::CUDAGraph *cuda_graph;
#else
typedef void *cuda_graph;
#endif

void dummy_cuda_dependency();
cuda_graph atcg_new();
void atcg_free(cuda_graph); 
void atcg_capture_begin(cuda_graph);
void atcg_capture_end(cuda_graph);
void atcg_replay(cuda_graph);
void atcg_reset(cuda_graph);

#ifdef __cplusplus
}
#endif

#endif
