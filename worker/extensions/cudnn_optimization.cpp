#include "common/extensions/cudnn_optimization.h"

#include <absl/synchronization/mutex.h>
#include <fmt/core.h>
#include <glib.h>
#include <stdint.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <gsl/gsl>
#include <iostream>

#include "common/endpoint_lib.hpp"
#include "worker/extensions/memory_server/client.hpp"

template <class handle_type>
struct HandleSet {
    std::vector<handle_type> handles;

    HandleSet();
    void cleanup();

    bool containsHandle(handle_type guest_handle) {
        if (__internal_allContextsEnabled()) {
            for (uint32_t gpuid = 0; gpuid < __internal_getDeviceCount(); gpuid++) {
                if (handles[gpuid] == guest_handle) return true;
            }
            return false;
        } else {
            return guest_handle == handles[0];
        }
    }

    handle_type getCurrentGPUHandle() {
        if (__internal_allContextsEnabled()) {
            return handles[__internal_getCurrentDevice()];
        } else {
            return handles[0];
        }
    }

    ~HandleSet() { cleanup(); } 

    static gint finder(gpointer a, gpointer b) {
        HandleSet* set = a;
        handle_type handle = b;
        //return 0 if it contains
        return set->containsHandle(handle) ? 0 : 1;
    }
};

template <>
HandleSet<cudnnHandle_t>::HandleSet() {
// if we need to create one per gpu
    if (__internal_allContextsEnabled()) {
        handles.reserve(__internal_getDeviceCount());
        for (uint32_t gpuid = 0; gpuid < __internal_getDeviceCount(); gpuid++) {
            if(cudaSetDevice(gpuid) != 0) {
                fprintf(stderr, "### error on cudaSetDevice at HandleSet cublas, skipping!\n");
                continue;
            }
            cudnnHandle_t cudnn_handle;
            cudnnStatus_t cudnn_ret = cudnnCreate(&cudnn_handle);
            if (cudnn_ret == CUDNN_STATUS_SUCCESS)
                handles.push_back(cudnn_handle);
            else
                fprintf(stderr, "### Failed to create CUDNN handle!\n");
        }
        cudaSetDevice(__internal_getCurrentDevice());
    }
    // otherwise keep it simple
    else {
        cudnnHandle_t cudnn_handle;
        cudnnStatus_t cudnn_ret = cudnnCreate(&cudnn_handle);
        if (cudnn_ret == CUDNN_STATUS_SUCCESS)
            handles.push_back(cudnn_handle);
        else
            fprintf(stderr, "### Failed to create CUDNN handle!\n");
    }
}

template <>
void HandleSet<cudnnHandle_t>::cleanup() {
    if (__internal_allContextsEnabled()) {
        for (uint32_t gpuid = 0; gpuid < __internal_getDeviceCount(); gpuid++) {
            if(cudaSetDevice(gpuid) != 0) {
                fprintf(stderr, "### error on cudaSetDevice at HandleSet cublas, skipping!\n");
                continue;
                cudnnDestroy(handles[gpuid]);
            }
        }
        cudaSetDevice(__internal_getCurrentDevice());
    } else {
        cudnnDestroy(handles[0]);
    }
}

template <>
HandleSet<cublasHandle_t>::HandleSet() {
// if we need to create one per gpu
    if (__internal_allContextsEnabled()) {
        handles.reserve(__internal_getDeviceCount());
        for (uint32_t gpuid = 0; gpuid < __internal_getDeviceCount(); gpuid++) {
            if(cudaSetDevice(gpuid) != 0) {
                fprintf(stderr, "### error on cudaSetDevice at HandleSet cublas, skipping!\n");
                continue;
            }
            cublasHandle_t cublas_handle;
            cublasStatus_t cublas_ret = cublasCreate(&cublas_handle);
            if (cublas_ret == CUBLAS_STATUS_SUCCESS)
                handles.push_back(cublas_handle);
            else
                fprintf(stderr, "### Failed to create CUDNN handle!\n");
        }
        cudaSetDevice(__internal_getCurrentDevice());
    }
    // otherwise keep it simple
    else {
        cublasHandle_t cublas_handle;
        cublasStatus_t cublas_ret = cublasCreate(&cublas_handle);
        if (cublas_ret == CUDNN_STATUS_SUCCESS)
            handles.push_back(cublas_handle);
        else
            fprintf(stderr, "### Failed to create CUDNN handle!\n");
    }
}

template <>
void HandleSet<cublasHandle_t>::cleanup() {
    if (__internal_allContextsEnabled()) {
        for (uint32_t gpuid = 0; gpuid < __internal_getDeviceCount(); gpuid++) {
            if(cudaSetDevice(gpuid) != 0) {
                fprintf(stderr, "### error on cudaSetDevice at HandleSet cublas, skipping!\n");
                continue;
                cublasDestroy(handles[gpuid]);
            }
        }
        cudaSetDevice(__internal_getCurrentDevice());
    } else {
        cublasDestroy(handles[0]);
    }
}

absl::Mutex cudnn_handles_mu;
GQueue *used_cudnn_handles ABSL_GUARDED_BY(cudnn_handles_mu);
GQueue *idle_cudnn_handles ABSL_GUARDED_BY(cudnn_handles_mu);

absl::Mutex cublas_handles_mu;
GQueue *used_cublas_handles ABSL_GUARDED_BY(cublas_handles_mu);
GQueue *idle_cublas_handles ABSL_GUARDED_BY(cublas_handles_mu);

// TODO(#86): Better way to avoid linking issue (referenced in spec utilities).
void guestlib_cudnn_opt_init(void) {}
void guestlib_cudnn_opt_fini(void) {}

static inline int __gettid() { return gsl::narrow_cast<int>(syscall(SYS_gettid)); }

cudnnHandle_t __get_cudnn_handle(cudnnHandle_t handle) {
    if (__internal_allContextsEnabled()) {
        GList* el = g_queue_find_custom(used_cudnn_handles, handle, HandleSet<cudnnHandle_t>::finder);
        if (el == NULL) {
            std::cerr << " ### Tried to get a non-existent cudnn handle, I guess." << std::endl;
            return 1;
        }
        HandleSet<cudnnHandle_t>* set = el->data;
        return set->getCurrentGPUHandle();
    } else {
        return handle;
    }
}

cublasHandle_t __get_cublas_handle(cublasHandle_t handle) {
    if (__internal_allContextsEnabled()) {
        GList* el = g_queue_find_custom(used_cublas_handles, handle, HandleSet<cublasHandle_t>::finder);
        if (el == NULL) {
            std::cerr << " ### Tried to get a non-existent cublas handle, I guess." << std::endl;
            return 1;
        }
        HandleSet<cublasHandle_t>* set = el->data;
        return set->getCurrentGPUHandle();
    } else {
        return handle;
    }
}

cudnnHandle_t __get_all_cudnn_handles(cudnnHandle_t handle) {

}

cublasHandle_t __get_all_cublas_handles(cublasHandle_t handle) {

}

/*
 *  cleanup functions
 */
void gq_delete(gpointer key, gpointer value, gpointer userdata) { delete value; }

void worker_cudnn_opt_cleanup(void) {
    {
        absl::MutexLock lk(&cudnn_handles_mu);
        g_queue_foreach(used_cudnn_handles, gq_delete, NULL);
        g_queue_free(used_cudnn_handles);

        g_queue_foreach(idle_cudnn_handles, gq_delete, NULL);
        g_queue_free(idle_cudnn_handles);
    }
    {
        absl::MutexLock lk(&cublas_handles_mu);
        g_queue_foreach(used_cublas_handles, gq_delete, NULL);
        g_queue_free(used_cublas_handles);

        g_queue_foreach(idle_cublas_handles, gq_delete, NULL);
        g_queue_free(idle_cublas_handles);
    }
}

/*
 *  pre creation function
 */

void worker_cudnn_opt_init(uint32_t n_handles) {
    used_cudnn_handles = g_queue_new();
    idle_cudnn_handles = g_queue_new();
    used_cublas_handles = g_queue_new();
    idle_cublas_handles = g_queue_new();

    // create all cudnn
    {
        absl::MutexLock lk(&cudnn_handles_mu);
        for (int i = 0; i < n_handles; i++) {
            auto handleset = new HandleSet<cudnnHandle_t>();
            g_queue_push_tail(idle_cudnn_handles, (gpointer)handleset);
        }
    }
    // create all cublas
    {
        absl::MutexLock lk(&cublas_handles_mu);
        for (int i = 0; i < n_handles; i++) {
            auto handleset = new HandleSet<cublasHandle_t>();
            g_queue_push_tail(idle_cublas_handles, (gpointer)handleset);
        }
    }
}

cudnnStatus_t __cudnnCreate(cudnnHandle_t *handle) {
#ifndef NDEBUG
    auto tid = __gettid();
    std::cerr << fmt::format("<thread={:x}> {} = {}\n", tid, __FUNCTION__, CUDNN_STATUS_SUCCESS);
#endif
  
    absl::MutexLock lk(&cudnn_handles_mu);

    if (g_queue_is_empty(idle_cudnn_handles)) {
        auto handleset = new HandleSet<cudnnHandle_t>();
        *handle = handleset->getCurrentGPUHandle();
        g_queue_push_tail(used_cudnn_handles, (gpointer) handleset);
        return CUDNN_STATUS_SUCCESS;
    }

    HandleSet<cudnnHandle_t> *handleset = g_queue_pop_head(idle_cudnn_handles);
    *handle = handleset->getCurrentGPUHandle();
    g_queue_push_tail(used_cudnn_handles, (gpointer) handleset);
    return CUDNN_STATUS_SUCCESS;
}

cublasStatus_t __cublasCreate(cublasHandle_t *handle) {
#ifndef NDEBUG
    auto tid = __gettid();
    std::cerr << fmt::format("<thread={:x}> {} = {}\n", tid, __FUNCTION__, CUBLAS_STATUS_SUCCESS);
#endif
  
    absl::MutexLock lk(&cublas_handles_mu);

    if (g_queue_is_empty(idle_cublas_handles)) {
        auto handleset = new HandleSet<cublasHandle_t>();
        *handle = handleset->getCurrentGPUHandle();
        g_queue_push_tail(used_cublas_handles, (gpointer) handleset);
        return CUBLAS_STATUS_SUCCESS;
    }

    HandleSet<cublasHandle_t> *handleset = g_queue_pop_head(idle_cublas_handles);
    *handle = handleset->getCurrentGPUHandle();
    g_queue_push_tail(used_cublas_handles, (gpointer) handleset);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t __helper_cublasDestroy(cublasHandle_t handle) {
    absl::MutexLock lk(&cublas_handles_mu);

    GList* el = g_queue_find_custom(used_cublas_handles, handle, HandleSet<cublasHandle_t>::finder);
    if (el == NULL) {
        std::cerr << " ### Tried to destroy a non-existent cublas handle, I guess." << std::endl;
        return 1;
    }
    HandleSet<cublasHandle_t>* set = el->data;
    //AFAIK this does not delete the pointer that is in the list, just the list element itself
    g_queue_delete_link(used_cublas_handles, el);
    g_queue_push_tail(idle_cublas_handles, (gpointer)set);
    return 0;
}

cudnnStatus_t __helper_cudnnDestroy(cudnnHandle_t handle) {
    absl::MutexLock lk(&cudnn_handles_mu);

    GList* el = g_queue_find_custom(used_cudnn_handles, handle, HandleSet<cudnnHandle_t>::finder);
    if (el == NULL) {
        std::cerr << " ### Tried to destroy a non-existent cudnn handle, I guess." << std::endl;
        return 1;
    }
    HandleSet<cudnnHandle_t>* set = el->data;
    //AFAIK this does not delete the pointer that is in the list, just the list element itself
    g_queue_delete_link(used_cudnn_handles, el);
    g_queue_push_tail(idle_cudnn_handles, (gpointer)set);
    return 0;
}

/*
 *      Pool cudnn descriptors below
 */ 
 

cudnnStatus_t __pool_cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t *convDesc, size_t count) {
  size_t i;
  cudnnConvolutionDescriptor_t *desc;
  cudnnStatus_t res = CUDNN_STATUS_SUCCESS;

  for (i = 0; i < count; i++) {
    desc = &convDesc[i];
    res = cudnnCreateConvolutionDescriptor(desc);
    if (res != CUDNN_STATUS_SUCCESS) return res;
  }

  return res;
}

cudnnStatus_t __pool_cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t *convDesc, size_t count) {
  size_t i;
  cudnnStatus_t res;

  for (i = 0; i < count; i++) {
    res = cudnnDestroyConvolutionDescriptor(convDesc[i]);
    if (res != CUDNN_STATUS_SUCCESS) return res;
  }

  return res;
}

cudnnStatus_t __pool_cudnnCreatePoolingDescriptor(cudnnPoolingDescriptor_t *poolingDesc, size_t count) {
  size_t i;
  cudnnPoolingDescriptor_t *desc;
  cudnnStatus_t res = CUDNN_STATUS_SUCCESS;

  for (i = 0; i < count; i++) {
    desc = &poolingDesc[i];
    res = cudnnCreatePoolingDescriptor(desc);
    if (res != CUDNN_STATUS_SUCCESS) return res;
  }

  return res;
}

cudnnStatus_t __pool_cudnnDestroyPoolingDescriptor(cudnnPoolingDescriptor_t *poolingDesc, size_t count) {
  size_t i;
  cudnnStatus_t res;

  for (i = 0; i < count; i++) {
    res = cudnnDestroyPoolingDescriptor(poolingDesc[i]);
    if (res != CUDNN_STATUS_SUCCESS) return res;
  }

  return res;
}

cudnnStatus_t __pool_cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t *tensorDesc, size_t count) {
  size_t i;
  cudnnTensorDescriptor_t *desc;
  cudnnStatus_t res = CUDNN_STATUS_SUCCESS;

  for (i = 0; i < count; i++) {
    desc = &tensorDesc[i];
    res = cudnnCreateTensorDescriptor(desc);
    if (res != CUDNN_STATUS_SUCCESS) return res;
  }

  return res;
}

cudnnStatus_t __pool_cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t *tensorDesc, size_t count) {
  size_t i;
  cudnnStatus_t res;

  for (i = 0; i < count; i++) {
    res = cudnnDestroyTensorDescriptor(tensorDesc[i]);
    if (res != CUDNN_STATUS_SUCCESS) return res;
  }

  return res;
}

cudnnStatus_t __pool_cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t *filterDesc, size_t count) {
  size_t i;
  cudnnFilterDescriptor_t *desc;
  cudnnStatus_t res = CUDNN_STATUS_SUCCESS;

  for (i = 0; i < count; i++) {
    desc = &filterDesc[i];
    res = cudnnCreateFilterDescriptor(desc);
    if (res != CUDNN_STATUS_SUCCESS) return res;
  }

  return res;
}

cudnnStatus_t __pool_cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t *filterDesc, size_t count) {
  size_t i;
  cudnnStatus_t res;

  for (i = 0; i < count; i++) {
    res = cudnnDestroyFilterDescriptor(filterDesc[i]);
    if (res != CUDNN_STATUS_SUCCESS) return res;
  }

  return res;
}

cudnnStatus_t __pool_cudnnCreateReduceTensorDescriptor(cudnnReduceTensorDescriptor_t *reduceTensorDesc, size_t count) {
  size_t i;
  cudnnReduceTensorDescriptor_t *desc;
  cudnnStatus_t res = CUDNN_STATUS_SUCCESS;

  for (i = 0; i < count; i++) {
    desc = &reduceTensorDesc[i];
    res = cudnnCreateReduceTensorDescriptor(desc);
    if (res != CUDNN_STATUS_SUCCESS) {
      return res;
    }
  }

  return res;
}

cudnnStatus_t __pool_cudnnDestroyReduceTensorDescriptor(cudnnReduceTensorDescriptor_t *reduceTensorDesc, size_t count) {
  size_t i;
  cudnnStatus_t res;

  for (i = 0; i < count; i++) {
    res = cudnnDestroyReduceTensorDescriptor(reduceTensorDesc[i]);
    if (res != CUDNN_STATUS_SUCCESS) {
      return res;
    }
  }

  return res;
}
