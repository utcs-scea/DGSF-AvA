#ifndef AVA_COMMON_EXTENSIONS_CUDART_10_1_UTILITIES_HPP_
#define AVA_COMMON_EXTENSIONS_CUDART_10_1_UTILITIES_HPP_

#include <absl/container/flat_hash_map.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <glib.h>

#include <algorithm>

#define MAX_KERNEL_ARG 30
#define MAX_KERNEL_NAME_LEN 1024
#define MAX_ASYNC_BUFFER_NUM 16

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

int deference_int_pointer(int *p);

#define cu_in_out_buffer(x, y)                                                \
  ({                                                                          \
    if (ava_is_in)                                                            \
      ava_buffer(x);                                                          \
    else                                                                      \
      ava_buffer(std::min(x, y == (void *)0 ? x : deference_int_pointer(y))); \
  })

struct fatbin_wrapper {
  uint32_t magic;
  uint32_t seq;
  uint64_t ptr;
  uint64_t data_ptr;
};

struct kernel_arg {
  char is_handle;
  uint32_t size;
};

struct fatbin_function {
  int argc;
  struct kernel_arg args[MAX_KERNEL_ARG];

  // this is barely any space, so leave 4 as default
  CUfunction cufunc[4];
  void *hostfunc[4];
  CUmodule module[4];
};

cudaError_t __helper_create_stream(cudaStream_t *pStream, unsigned int flag, int priority);
cudaError_t __helper_destroy_stream(cudaStream_t stream);
CUresult __helper_destroy_custream(CUstream stream);

size_t __helper_fatbin_size(const void *cubin);

void __helper_print_kernel_info(struct fatbin_function *func, void **args);

cudaError_t __helper_launch_kernel(struct fatbin_function *func, const void *hostFun, dim3 gridDim, dim3 blockDim,
                                   void **args, size_t sharedMem, cudaStream_t stream);

int __helper_cubin_num(void **cubin_handle);

void __helper_print_fatcubin_info(void *fatCubin, void **ret);

void __helper_unregister_fatbin(void **fatCubinHandle);

void __helper_parse_function_args(const char *name, struct kernel_arg *args);

size_t __helper_launch_extra_size(void **extra);

void *__helper_cu_mem_host_alloc_portable(size_t size);

void __helper_cu_mem_host_free(void *ptr);

void __helper_assosiate_function_dump(GHashTable *funcs, struct fatbin_function **func, void *local,
                                      const char *deviceName);

void __helper_register_function(struct fatbin_function *func, const char *hostFun, CUmodule *module,
                                const char *deviceName);

/* Async buffer address list */
struct async_buffer_list {
  int num_buffers;
  void *buffers[MAX_ASYNC_BUFFER_NUM]; /* array of buffer addresses */
  size_t buffer_sizes[MAX_ASYNC_BUFFER_NUM];
};

void __helper_register_async_buffer(struct async_buffer_list *buffers, void *buffer, size_t size);

struct async_buffer_list *__helper_load_async_buffer_list(struct async_buffer_list *buffers);

int __helper_a_last_dim_size(cublasOperation_t transa, int k, int m);

int __helper_b_last_dim_size(cublasOperation_t transb, int k, int n);

int __helper_type_size(cudaDataType dataType);

cudaError_t __helper_func_get_attributes(struct cudaFuncAttributes *attr, struct fatbin_function *func,
                                         const void *hostFun);

cudaError_t __helper_occupancy_max_active_blocks_per_multiprocessor(int *numBlocks, struct fatbin_function *func,
                                                                    const void *hostFun, int blockSize,
                                                                    size_t dynamicSMemSize);

cudaError_t __helper_occupancy_max_active_blocks_per_multiprocessor_with_flags(int *numBlocks,
                                                                               struct fatbin_function *func,
                                                                               const void *hostFun, int blockSize,
                                                                               size_t dynamicSMemSize,
                                                                               unsigned int flags);

void __helper_print_pointer_attributes(const struct cudaPointerAttributes *attributes, const void *ptr);

cudaError_t __helper_cuda_memcpy_async_host_to_host(void *dst, const void *src, size_t count, cudaStream_t stream);
cudaError_t __helper_cuda_memcpy_async_host_to_device(void *dst, const void *src, size_t count, cudaStream_t stream);
cudaError_t __helper_cuda_memcpy_async_device_to_host(void *dst, const void *src, size_t count, cudaStream_t stream);
cudaError_t __helper_cuda_memcpy_async_device_to_device(void *dst, const void *src, size_t count, cudaStream_t stream);
cudaError_t __helper_cuda_memcpy_async_default(void *dst, const void *src, size_t count, cudaStream_t stream,
                                               bool dst_is_gpu, bool src_is_gpu);

void __helper_record_module_path(CUmodule module, const char *fname);
void __helper_parse_module_function_args(CUmodule module, const char *name, struct fatbin_function **func);
void __helper_init_module(struct fatbin_wrapper *fatCubin, void **handle, CUmodule *module);
CUresult __helper_cuModuleLoad(CUmodule *module, const char *fname);
CUresult __internal_cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                   unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                                   unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra);
cudaError_t __internal_cudaMalloc(void **devPtr, size_t size);
CUresult __internal_cuMemFree(CUdeviceptr dptr);
cudaError_t __internal_cudaFree(void *devPtr);

uint32_t __internal_getCurrentDeviceIndex();
CUresult __helper_cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int *flags, int *active);
CUresult __helper_cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev);
CUresult __helper_cuDeviceGetName(char *name, int len, CUdevice dev);
CUresult __helper_cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev);
CUresult __helper_cuDeviceGetUuid(CUuuid *uuid, CUdevice dev);
CUresult __helper_cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags);
CUresult __helper_cuDeviceTotalMem(size_t *bytes, CUdevice dev);
CUresult __helper_cuDeviceGetPCIBusId(char *pciBusId, int len, CUdevice dev);
CUresult __helper_cuDeviceComputeCapability(int *major, int *minor, CUdevice device);
cudaError_t __helper_cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device);
cudaError_t __helper_cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr, int device);
CUresult __helper_cuDeviceGet(CUdevice *device, int ordinal);
cudaError_t __helper_cudaPointerGetAttributes(struct cudaPointerAttributes *attributes, const void *ptr);
void __helper_record_tensor_desc(cudnnTensorDescriptor_t desc, cudnnDataType_t data_type);
bool __helper_get_tensor_type(cudnnTensorDescriptor_t desc, cudnnDataType_t *data_type);

cudaError_t __helper_cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags);
cudaError_t __helper_cudaEventDestroy(cudaEvent_t event);
cudaError_t __helper_cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
cudaError_t __helper_cudaEventSynchronize(cudaEvent_t event);

cudaEvent_t __helper_translate_event(cudaEvent_t event);
cudaStream_t __helper_translate_stream(cudaStream_t stream);

cudaError_t __helper_cudaStreamSynchronize_sync(cudaStream_t stream);
cudaError_t __helper_cudaStreamSynchronize_async(cudaStream_t stream);

cublasStatus_t __helper_cublasSetStream(cublasHandle_t handle, cudaStream_t streamId);
cudnnStatus_t __helper_cudnnSetStream(cudnnHandle_t handle, cudaStream_t streamId);

cublasStatus_t __helper_cublasSgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m,
                                       int n, int k, const float *alpha, /* host or device pointer */
                                       const float *A, int lda, const float *B, int ldb,
                                       const float *beta, /* host or device pointer */
                                       float *C, int ldc, bool alpha_is_gpu, bool beta_is_gpu);

cublasStatus_t __helper_cublasSgemmStridedBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
    const float *alpha,                                                /* host or device pointer */
    const float *A, int lda, long long int strideA,                    /* purposely signed */
    const float *B, int ldb, long long int strideB, const float *beta, /* host or device pointer */
    float *C, int ldc, long long int strideC, int batchCount, bool alpha_is_gpu, bool beta_is_gpu);

cudnnStatus_t cudnnBatchNormalizationForwardInference_double(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, const double *alpha, /* alpha[0] = result blend factor */
    const double *beta,                                                   /* beta[0] = dest layer blend factor */
    const cudnnTensorDescriptor_t xDesc, const void *x,                   /* NxCxHxW */
    const cudnnTensorDescriptor_t yDesc, void *y,                         /* NxCxHxW */
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale, const void *bnBias,
    const void *estimatedMean, const void *estimatedVariance, double epsilon);

cudnnStatus_t cudnnBatchNormalizationForwardInference_float(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, const float *alpha, /* alpha[0] = result blend factor */
    const float *beta,                                                   /* beta[0] = dest layer blend factor */
    const cudnnTensorDescriptor_t xDesc, const void *x,                  /* NxCxHxW */
    const cudnnTensorDescriptor_t yDesc, void *y,                        /* NxCxHxW */
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale, const void *bnBias,
    const void *estimatedMean, const void *estimatedVariance, double epsilon);

cudnnStatus_t __helper_cudnnConvolutionForward_double(cudnnHandle_t handle, const double *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionFwdAlgo_t algo, void *workSpace,
    size_t workSpaceSizeInBytes, const double *beta,
    const cudnnTensorDescriptor_t yDesc, void *y);

cudnnStatus_t __helper_cudnnConvolutionForward_float(cudnnHandle_t handle, const float *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionFwdAlgo_t algo, void *workSpace,
    size_t workSpaceSizeInBytes, const float *beta,
    const cudnnTensorDescriptor_t yDesc, void *y);

cudnnStatus_t cudnnPoolingForward_double(cudnnHandle_t handle, const cudnnPoolingDescriptor_t poolingDesc,
                                         const double *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
                                         const double *beta, const cudnnTensorDescriptor_t yDesc, void *y);

cudnnStatus_t cudnnPoolingForward_float(cudnnHandle_t handle, const cudnnPoolingDescriptor_t poolingDesc,
                                        const float *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
                                        const float *beta, const cudnnTensorDescriptor_t yDesc, void *y);

cudnnStatus_t cudnnAddTensor_double(cudnnHandle_t handle, const double *alpha, const cudnnTensorDescriptor_t aDesc,
                                    const void *A, const double *beta, const cudnnTensorDescriptor_t cDesc, void *C);

cudnnStatus_t cudnnAddTensor_float(cudnnHandle_t handle, const float *alpha, const cudnnTensorDescriptor_t aDesc,
                                   const void *A, const float *beta, const cudnnTensorDescriptor_t cDesc, void *C);

cudnnStatus_t cudnnReduceTensor_double(cudnnHandle_t handle, const cudnnReduceTensorDescriptor_t reduceTensorDesc,
                                       void *indices, size_t indicesSizeInBytes, void *workspace,
                                       size_t workspaceSizeInBytes, const double *alpha,
                                       const cudnnTensorDescriptor_t aDesc, const void *A, const double *beta,
                                       const cudnnTensorDescriptor_t cDesc, void *C);
      
cudnnStatus_t cudnnReduceTensor_float(cudnnHandle_t handle, const cudnnReduceTensorDescriptor_t reduceTensorDesc,
                                      void *indices, size_t indicesSizeInBytes, void *workspace,
                                      size_t workspaceSizeInBytes, const float *alpha,
                                      const cudnnTensorDescriptor_t aDesc, const void *A, const float *beta,
                                      const cudnnTensorDescriptor_t cDesc, void *C);

cudnnStatus_t cudnnScaleTensor_double(cudnnHandle_t handle, const cudnnTensorDescriptor_t yDesc, void *y,
                                      const double *alpha);

cudnnStatus_t cudnnScaleTensor_float(cudnnHandle_t handle, const cudnnTensorDescriptor_t yDesc, void *y,
                                     const float *alpha);

cudnnStatus_t cudnnConvolutionBiasActivationForward_double(
    cudnnHandle_t handle, const double *alpha1, const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionFwdAlgo_t algo, void *workSpace, size_t workSpaceSizeInBytes, const double *alpha2,
    const cudnnTensorDescriptor_t zDesc, const void *z, const cudnnTensorDescriptor_t biasDesc, const void *bias,
    const cudnnActivationDescriptor_t activationDesc, const cudnnTensorDescriptor_t yDesc, void *y);

cudnnStatus_t cudnnConvolutionBiasActivationForward_float(
    cudnnHandle_t handle, const float *alpha1, const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionFwdAlgo_t algo, void *workSpace, size_t workSpaceSizeInBytes, const float *alpha2,
    const cudnnTensorDescriptor_t zDesc, const void *z, const cudnnTensorDescriptor_t biasDesc, const void *bias,
    const cudnnActivationDescriptor_t activationDesc, const cudnnTensorDescriptor_t yDesc, void *y);

cudnnStatus_t cudnnSoftmaxForward_double(cudnnHandle_t handle, cudnnSoftmaxAlgorithm_t algo, cudnnSoftmaxMode_t mode,
                                         const double *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
                                         const double *beta, const cudnnTensorDescriptor_t yDesc, void *y);

cudnnStatus_t cudnnSoftmaxForward_float(cudnnHandle_t handle, cudnnSoftmaxAlgorithm_t algo, cudnnSoftmaxMode_t mode,
                                        const float *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
                                        const float *beta, const cudnnTensorDescriptor_t yDesc, void *y);

#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif  // AVA_COMMON_EXTENSIONS_CUDART_10_1_UTILITIES_HPP_
