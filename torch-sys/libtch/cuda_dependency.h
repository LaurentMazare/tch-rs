#include<stdio.h>
#include<stdint.h>

#ifdef __cplusplus
extern "C" {
#else
typedef void cuda_stream;
typedef void cuda_stream_guard;
#endif

void dummy_cuda_dependency();

cuda_stream *get_stream_from_pool(int,  int);
void delete_stream(cuda_stream *);

cuda_stream_guard *get_stream_guard(cuda_stream *);
void delete_stream_guard(cuda_stream_guard *);

#ifdef __cplusplus
}
#endif
