#define __TCH_ACTUAL_CUDA_DEPENDENCY__
#include "cuda_dependency.h"
#include "torch_api.h"

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
    std::cerr << "error initializing cuda: " << e.what() << std::endl;
  }
}

cuda_graph atcg_new() {
  PROTECT(
  return new at::cuda::CUDAGraph();
  )
  return nullptr;
}

void atcg_free(cuda_graph c) {
  delete c;
}
void atcg_capture_begin(cuda_graph c) {
  PROTECT(
      c->capture_begin();
  )
}

void atcg_capture_end(cuda_graph c) {
  PROTECT(
      c->capture_end();
  )
}

void atcg_replay(cuda_graph c) {
  PROTECT(
      c->replay();
  )
}

void atcg_reset(cuda_graph c) {
  PROTECT(
      c->reset();
  )
}
