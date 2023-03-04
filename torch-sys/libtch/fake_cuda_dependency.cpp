#include "cuda_dependency.h"

void dummy_cuda_dependency() {
}

cuda_graph atcg_new() {
  return nullptr;
}

void atcg_free(cuda_graph) {}
void atcg_capture_begin(cuda_graph) {}
void atcg_capture_end(cuda_graph) {}
void atcg_replay(cuda_graph) {}
void atcg_reset(cuda_graph) {}
