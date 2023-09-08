#ifndef _AVA_GUESTLIB_EXTENSIONS_CUDART_101_UTILITIES_H_
#define _AVA_GUESTLIB_EXTENSIONS_CUDART_101_UTILITIES_H_

#include <cuda_runtime_api.h>
void set_stream_synchronize_safe_async(cudaStream_t stream, bool safe_ava_async);
bool get_stream_synchronize_safe_async(cudaStream_t stream);

#endif  // _AVA_GUESTLIB_EXTENSIONS_CUDART_101_UTILITIES_H_
