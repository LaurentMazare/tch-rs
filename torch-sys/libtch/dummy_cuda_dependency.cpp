#include<stdio.h>
#include<stdint.h>
using namespace std;
extern "C" {
    void dummy_cuda_dependency();
}

namespace at {
    namespace cuda {
        void *getCurrentCUDABlasHandle();
        int warp_size();
    }
}
char * magma_strerror(int err);
void dummy_cuda_dependency() {
    at::cuda::getCurrentCUDABlasHandle();
    at::cuda::warp_size();
}
