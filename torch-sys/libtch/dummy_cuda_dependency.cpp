extern "C" {
    void dummy_cuda_dependency();
}

namespace at {
    namespace cuda {
        void getDeviceProperties(long);
    }
}
void dummy_cuda_dependency() {
    at::cuda::getDeviceProperties(0);
}
