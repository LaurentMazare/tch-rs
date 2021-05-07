#include<stdio.h>
#include<stdint.h>
using namespace std;
extern "C" {
    void dummy_cuda_dependency();
}

struct cublasContext;

namespace at {
    class Tensor;
    namespace native {
        class at::Tensor searchsorted_cuda(class at::Tensor const &,class at::Tensor const &,bool,bool);
    }
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
