#ifndef __TORCH_C10_CUDA_API_H__
#define __TORCH_C10_CUDA_API_H__

#include<torch/csrc/cuda/CUDAPluggableAllocator.h>

extern "C" {
    void emptyCache();
}

#endif