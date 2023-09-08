/**
 * This file implements the handle pooling optimization for cuDNN and related APIs
 * used in TensorFlow 1.14 and ONNXruntime 1.2.0.
 * The underlying dependencies are CUDA 10.1 and cuDNN 7.6.5.
 * The optimization is applied in `cava/samples/onnxruntime/onnx_opt.c`.
 */
#ifndef AVA_EXTENSIONS_CUDNN_OPTIMIZATION_H_
#define AVA_EXTENSIONS_CUDNN_OPTIMIZATION_H_

#include <cublas_v2.h>
#include <cuda.h>
#include <cudnn.h>
#include <glib.h>
#include <deque>

#ifdef __cplusplus
extern "C" {
#endif

#define DESCRITPOR_POOL_SIZE 64

void guestlib_cudnn_opt_init(void);
void guestlib_cudnn_opt_fini(void);
void worker_cudnn_opt_init(uint32_t n_handles);
void worker_cudnn_opt_cleanup();

cudnnStatus_t __pool_cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t *convDesc, size_t count);
cudnnStatus_t __pool_cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t *convDesc, size_t count);
int free_convolution_descriptor_pool(GQueue *pool);

cudnnStatus_t __pool_cudnnCreatePoolingDescriptor(cudnnPoolingDescriptor_t *poolingDesc, size_t count);
cudnnStatus_t __pool_cudnnDestroyPoolingDescriptor(cudnnPoolingDescriptor_t *poolingDesc, size_t count);
int free_pooling_descriptor_pool(GQueue *pool);

cudnnStatus_t __pool_cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t *tensorDesc, size_t count);
cudnnStatus_t __pool_cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t *tensorDesc, size_t count);
int free_tensor_descriptor_pool(GQueue *pool);

cudnnStatus_t __pool_cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t *filterDesc, size_t count);
cudnnStatus_t __pool_cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t *filterDesc, size_t count);
int free_filter_descriptor_pool(GQueue *pool);

cudnnStatus_t __pool_cudnnCreateReduceTensorDescriptor(cudnnReduceTensorDescriptor_t *reduceTensorDesc, size_t count);
cudnnStatus_t __pool_cudnnDestroyReduceTensorDescriptor(cudnnReduceTensorDescriptor_t *reduceTensorDesc, size_t count);
int free_reduce_tensor_descriptor_pool(GQueue *pool);

cudnnStatus_t __cudnnCreate(cudnnHandle_t *handle);
cublasStatus_t __cublasCreate(cublasHandle_t *handle);
cublasStatus_t __helper_cublasDestroy(cublasHandle_t handle);
cudnnStatus_t __helper_cudnnDestroy(cudnnHandle_t handle);

cudnnHandle_t __get_cudnn_handle(cudnnHandle_t handle);
cublasHandle_t __get_cublas_handle(cublasHandle_t handle);


#ifdef __cplusplus
}
#endif

#endif  // AVA_EXTENSIONS_CUDNN_OPTIMIZATION_H_
