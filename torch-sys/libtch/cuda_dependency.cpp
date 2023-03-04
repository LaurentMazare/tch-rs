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

void atcs_free(cuda_stream s) {
  delete s;
}

cuda_stream atcs_get_stream_from_pool(int high_priority, int device) {
  PROTECT (
      return new c10::cuda::CUDAStream(c10::cuda::getStreamFromPool(high_priority, device));
  )
  return nullptr;
}

cuda_stream atcs_get_default_stream(int device) {
  PROTECT (
      return new c10::cuda::CUDAStream(c10::cuda::getDefaultCUDAStream(device));
  )
  return nullptr;
}

cuda_stream atcs_get_current_stream(int device) {
  PROTECT (
      return new c10::cuda::CUDAStream(c10::cuda::getCurrentCUDAStream(device));
  )
  return nullptr;
}

void atcs_set_current_stream(cuda_stream s) {
  PROTECT (
      c10::cuda::setCurrentCUDAStream(*s);
  )
}
