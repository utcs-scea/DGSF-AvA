#include "cudart_10.1_utilities.hpp"

#include <absl/container/flat_hash_map.h>
absl::Mutex stream_synchronize_safe_async_mu;
absl::flat_hash_map<cudaStream_t, bool> stream_synchronize_safe_async;

void set_stream_synchronize_safe_async(cudaStream_t stream, bool safe_ava_async) {
  absl::MutexLock lk(&stream_synchronize_safe_async_mu);
  stream_synchronize_safe_async[stream] = safe_ava_async;
}

bool get_stream_synchronize_safe_async(cudaStream_t stream) {
  absl::MutexLock lk(&stream_synchronize_safe_async_mu);
  auto search = stream_synchronize_safe_async.find(stream);
  if (search == stream_synchronize_safe_async.end()) {
    stream_synchronize_safe_async[stream] = true;
  }
  return stream_synchronize_safe_async[stream];
}