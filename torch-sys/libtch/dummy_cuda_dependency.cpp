#include<stdio.h>
#include<stdint.h>
#include<stdexcept>
#include<iostream>
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
  try {
    at::cuda::getCurrentCUDABlasHandle();
    at::cuda::warp_size();
  }
  catch (std::exception &e) {
    if (getenv("TCH_PRINT_CUDA_INIT_ERROR") != nullptr) {
      std::cerr << "error initializing cuda: " << e.what() << std::endl;
    }
  }
}
