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

void atcs_free(cuda_stream); {}
cuda_stream atcs_get_stream_from_pool(int, int) {
  return nullptr;
}
cuda_stream atcs_get_default_stream(int) {
  return nullptr;
}
cuda_stream atcs_get_current_stream(int) {
  return nullptr;
}
void atcs_set_current_stream(cuda_stream) {}
