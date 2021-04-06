typedef void *cuda_stream;
typedef void *cuda_stream_guard;

#include "cuda_dependency.h"

void dummy_cuda_dependency() {
}

cuda_stream *get_stream_from_pool(int,  int) {
    return nullptr;
}

void delete_stream(cuda_stream *) {
}

cuda_stream_guard *get_stream_guard(cuda_stream *) {
    return nullptr;
}

void delete_stream_guard(cuda_stream_guard *) {
}
