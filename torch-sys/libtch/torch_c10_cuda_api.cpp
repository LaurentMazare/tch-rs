
#include "torch_c10_cuda_api.h"


void emptyCache(){
    c10::cuda::CUDACachingAllocator::emptyCache();
}