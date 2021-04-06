#include "torch_api.h"
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

typedef c10::cuda::CUDAStream *cuda_stream;
typedef c10::cuda::CUDAStreamGuard *cuda_stream_guard;

#include "cuda_dependency.h"
using namespace std;

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

cuda_stream *get_stream_from_pool(int high_priority,  int device) {
    PROTECT (
      return new c10::cuda::CUDAStream(at::cuda::getStreamFromPool(high_priority, device));
    )
    return nullptr;
}

void delete_stream(cuda_stream *c) {
    delete(c);
}

cuda_stream_guard *get_stream_guard(cuda_stream *c) {
    PROTECT (
      return new c10::cuda::CUDAStreamGuard(*c);
    )
    return nullptr;
}

void delete_stream_guard(cuda_stream_guard * g) {
    delete(g);
}
