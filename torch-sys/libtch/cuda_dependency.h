#ifndef __TCH_CUDA_DEPENDENCY_H__
#define __TCH_CUDA_DEPENDENCY_H__

#include<stdio.h>
#include<stdint.h>

#ifdef __cplusplus

#ifdef __TCH_ACTUAL_CUDA_DEPENDENCY__
#include<ATen/cuda/CUDAGraph.h>
#include<c10/cuda/CUDAStream.h>
typedef at::cuda::CUDAGraph *cuda_graph;
typedef c10::cuda::CUDAStream *cuda_stream;
#else
typedef void *cuda_graph;
typedef void *cuda_stream;
#endif

extern "C" {

#else
typedef void *cuda_graph;
typedef void *cuda_stream;
#endif

void dummy_cuda_dependency();
cuda_graph atcg_new();
void atcg_free(cuda_graph); 
void atcg_capture_begin(cuda_graph);
void atcg_capture_end(cuda_graph);
void atcg_replay(cuda_graph);
void atcg_reset(cuda_graph);

void atcs_free(cuda_stream); 
cuda_stream atcs_get_stream_from_pool(int, int);
cuda_stream atcs_get_default_stream(int);
cuda_stream atcs_get_current_stream(int);
void atcs_set_current_stream(cuda_stream);

#ifdef __cplusplus
}
#endif

#endif
