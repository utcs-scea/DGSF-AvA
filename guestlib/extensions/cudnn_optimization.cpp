#include "common/extensions/cudnn_optimization.h"

#include <glib.h>
#include <stdint.h>
// #include <deque>
#include <absl/synchronization/mutex.h>

#include "common/endpoint_lib.hpp"

absl::Mutex conv_desc_pool_mu;
GQueue *convolution_descriptor_pool ABSL_GUARDED_BY(conv_desc_pool_mu);
GQueue *idle_convolution_descriptor_pool ABSL_GUARDED_BY(conv_desc_pool_mu);

absl::Mutex pooling_desc_pool_mu;
GQueue *pooling_descriptor_pool ABSL_GUARDED_BY(pooling_desc_pool_mu);
GQueue *idle_pooling_descriptor_pool ABSL_GUARDED_BY(pooling_desc_pool_mu);

absl::Mutex tensor_desc_pool_mu;
GQueue *tensor_descriptor_pool ABSL_GUARDED_BY(tensor_desc_pool_mu);
GQueue *idle_tensor_descriptor_pool ABSL_GUARDED_BY(tensor_desc_pool_mu);

absl::Mutex filter_desc_pool_mu;
GQueue *filter_descriptor_pool ABSL_GUARDED_BY(filter_desc_pool_mu);
GQueue *idle_filter_descriptor_pool ABSL_GUARDED_BY(filter_desc_pool_mu);

absl::Mutex reduce_tensor_desc_pool_mu;
GQueue* reduce_tensor_descriptor_pool ABSL_GUARDED_BY(reduce_tensor_desc_pool_mu);
GQueue* idle_reduce_tensor_descriptor_pool ABSL_GUARDED_BY(reduce_tensor_desc_pool_mu);

void worker_cudnn_opt_init(void) {}

void guestlib_cudnn_opt_init(void) {
  /* Pool descriptors */
  convolution_descriptor_pool = g_queue_new();
  idle_convolution_descriptor_pool = g_queue_new();

  pooling_descriptor_pool = g_queue_new();
  idle_pooling_descriptor_pool = g_queue_new();

  tensor_descriptor_pool = g_queue_new();
  idle_tensor_descriptor_pool = g_queue_new();

  filter_descriptor_pool = g_queue_new();
  idle_filter_descriptor_pool = g_queue_new();

  reduce_tensor_descriptor_pool = g_queue_new();
  idle_reduce_tensor_descriptor_pool = g_queue_new();
}

void guestlib_cudnn_opt_fini(void) {
  /* Free descriptors */
  free_convolution_descriptor_pool(convolution_descriptor_pool);
  free_convolution_descriptor_pool(idle_convolution_descriptor_pool);
  g_queue_free(convolution_descriptor_pool);
  g_queue_free(idle_convolution_descriptor_pool);

  free_pooling_descriptor_pool(pooling_descriptor_pool);
  free_pooling_descriptor_pool(idle_pooling_descriptor_pool);
  g_queue_free(pooling_descriptor_pool);
  g_queue_free(idle_pooling_descriptor_pool);

  free_tensor_descriptor_pool(tensor_descriptor_pool);
  free_tensor_descriptor_pool(idle_tensor_descriptor_pool);
  g_queue_free(tensor_descriptor_pool);
  g_queue_free(idle_tensor_descriptor_pool);

  free_filter_descriptor_pool(filter_descriptor_pool);
  free_filter_descriptor_pool(idle_filter_descriptor_pool);
  g_queue_free(filter_descriptor_pool);
  g_queue_free(idle_filter_descriptor_pool);

  free_reduce_tensor_descriptor_pool(reduce_tensor_descriptor_pool);
  free_reduce_tensor_descriptor_pool(idle_reduce_tensor_descriptor_pool);
  g_queue_free(reduce_tensor_descriptor_pool);
  g_queue_free(idle_reduce_tensor_descriptor_pool);
}

int free_convolution_descriptor_pool(GQueue *pool) ABSL_EXCLUSIVE_LOCKS_REQUIRED(conv_desc_pool_mu) {
  gpointer element;
  cudnnConvolutionDescriptor_t *desc;
  int i = 0;

  if (g_queue_is_empty(pool)) {
    return CUDNN_STATUS_SUCCESS;
  }

  desc = (cudnnConvolutionDescriptor_t *)malloc(sizeof(cudnnConvolutionDescriptor_t) * pool->length);

  while ((element = g_queue_pop_head(pool))) {
    desc[i++] = (cudnnConvolutionDescriptor_t)element;
  }

  auto ret = __pool_cudnnDestroyConvolutionDescriptor(desc, i);
  return ret;
}

int free_pooling_descriptor_pool(GQueue *pool) ABSL_EXCLUSIVE_LOCKS_REQUIRED(pooling_desc_pool_mu) {
  gpointer element;
  cudnnPoolingDescriptor_t *desc;
  int i = 0;

  if (g_queue_is_empty(pool)) {
    return CUDNN_STATUS_SUCCESS;
  }

  desc = (cudnnPoolingDescriptor_t *)malloc(sizeof(cudnnPoolingDescriptor_t) * pool->length);

  while ((element = g_queue_pop_head(pool))) {
    desc[i++] = (cudnnPoolingDescriptor_t)element;
  }

  auto ret = __pool_cudnnDestroyPoolingDescriptor(desc, i);
  return ret;
}

int free_tensor_descriptor_pool(GQueue *pool) ABSL_EXCLUSIVE_LOCKS_REQUIRED(tensor_desc_pool_mu) {
  gpointer element;
  cudnnTensorDescriptor_t *desc;
  int i = 0;
  
  if (g_queue_is_empty(pool)) {
    return CUDNN_STATUS_SUCCESS;
  }

  desc = (cudnnTensorDescriptor_t *)malloc(sizeof(cudnnTensorDescriptor_t) * pool->length);

  while ((element = g_queue_pop_head(pool))) {
    desc[i++] = (cudnnTensorDescriptor_t)element;
  }

  auto ret = __pool_cudnnDestroyTensorDescriptor(desc, i);
  return ret;
}

int free_filter_descriptor_pool(GQueue *pool) ABSL_EXCLUSIVE_LOCKS_REQUIRED(filter_desc_pool_mu) {
  gpointer element;
  cudnnFilterDescriptor_t *desc;
  int i = 0;

  if (g_queue_is_empty(pool)) {
    return CUDNN_STATUS_SUCCESS;
  }

  desc = (cudnnFilterDescriptor_t *)malloc(sizeof(cudnnFilterDescriptor_t) * pool->length);

  while ((element = g_queue_pop_head(pool))) {
    desc[i++] = (cudnnFilterDescriptor_t)element;
  }

  auto ret = __pool_cudnnDestroyFilterDescriptor(desc, i);
  return ret;
}

int free_reduce_tensor_descriptor_pool(GQueue *pool) ABSL_EXCLUSIVE_LOCKS_REQUIRED(reduce_tensor_desc_pool_mu) {
  gpointer element;
  cudnnReduceTensorDescriptor_t *desc;
  int i = 0;

  if (g_queue_is_empty(pool)) {
    return CUDNN_STATUS_SUCCESS;
  }

  desc = (cudnnReduceTensorDescriptor_t *)malloc(sizeof(cudnnReduceTensorDescriptor_t) * pool->length);
  while ((element = g_queue_pop_head(pool))) {
    desc[i++] = (cudnnReduceTensorDescriptor_t)element;
  }

  auto ret = __pool_cudnnDestroyReduceTensorDescriptor(desc, i);
  return ret;
}
