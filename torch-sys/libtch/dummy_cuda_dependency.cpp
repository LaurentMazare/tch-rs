extern "C" {
    void dummy_cuda_dependency();
}

namespace at {
    namespace cuda {
        int warp_size();
    }
}
void dummy_cuda_dependency() {
  at::cuda::warp_size();
}
