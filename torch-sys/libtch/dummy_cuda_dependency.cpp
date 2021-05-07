#include<stdio.h>
#include<stdint.h>
using namespace std;
extern "C" {
    void dummy_cuda_dependency();
}

struct cublasContext;

namespace at {
    namespace cuda {
        cublasContext* getCurrentCUDABlasHandle();
        int warp_size();
    }
}
char * magma_strerror(int err);
void dummy_cuda_dependency() {
    at::cuda::getCurrentCUDABlasHandle();
    at::cuda::warp_size();
}
