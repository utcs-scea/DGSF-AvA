#include "cudart_10.1_utilities.hpp"
#include <cuda_runtime_api.h>
#include <fatbinary.h>
#include <plog/Log.h>
#include <stdlib.h>
#include <stdexcept>
#include <deque>
#include "worker/extensions/memory_server/client.hpp"
#include <iostream>
#include <memory>
#include <fmt/core.h>
#include "common/logging.h"
#include <absl/synchronization/mutex.h>

absl::Mutex module_path_mu;
absl::flat_hash_map<CUmodule, std::string> module_path_map ABSL_GUARDED_BY(module_path_mu);

struct tensor_desc {
  cudnnDataType_t data_type;
  tensor_desc(cudnnDataType_t data_type): data_type(data_type) {}
};

absl::Mutex tensor_desc_map_mu;
absl::flat_hash_map<cudnnTensorDescriptor_t, std::unique_ptr<tensor_desc>> tensor_desc_map ABSL_GUARDED_BY(tensor_desc_map_mu);

void __helper_record_tensor_desc(cudnnTensorDescriptor_t desc, cudnnDataType_t data_type) {
  absl::MutexLock lk(&tensor_desc_map_mu);
  tensor_desc_map[desc] = std::make_unique<tensor_desc>(data_type);
}

bool __helper_get_tensor_type(cudnnTensorDescriptor_t desc, cudnnDataType_t *data_type) {
  absl::MutexLock lk(&tensor_desc_map_mu);
  auto search = tensor_desc_map.find(desc);
  if (search != tensor_desc_map.end()) {
    *data_type = search->second->data_type;
    return true;
  } else {
    return false;
  }
}

int deference_int_pointer(int *p) {
  if (p)
    return *p;
  else
    return 0;
}

size_t __helper_fatbin_size(const void *cubin) {
  struct fatBinaryHeader *fbh = (struct fatBinaryHeader *)cubin;
  return fbh->fatSize + fbh->headerSize;
}

int __helper_cubin_num(void **cubin_handle) {
  int num = 0;
  while (cubin_handle[num] != NULL) num++;
  return num;
}

void __helper_print_fatcubin_info(void *fatCubin, void **ret) {
  struct fatbin_wrapper *wp = (struct fatbin_wrapper *)fatCubin;
  printf("fatCubin_wrapper=%p, []={.magic=0x%X, .seq=%d, ptr=0x%lx, data_ptr=0x%lx}\n", fatCubin, wp->magic, wp->seq,
         wp->ptr, wp->data_ptr);
  struct fatBinaryHeader *fbh = (struct fatBinaryHeader *)wp->ptr;
  printf("fatBinaryHeader={.magic=0x%X, version=%d, headerSize=0x%x, fatSize=0x%llx}\n", fbh->magic, fbh->version,
         fbh->headerSize, fbh->fatSize);
  char *fatBinaryEnd = (char *)(wp->ptr + fbh->headerSize + fbh->fatSize);
  printf("fatBin=0x%lx--0x%lx\n", wp->ptr, (int64_t)fatBinaryEnd);

  fatBinaryEnd = (char *)(wp->ptr);
  int i, j;
  for (i = 0; i < 100; i++)
    if (fatBinaryEnd[i] == 0x7F && fatBinaryEnd[i + 1] == 'E' && fatBinaryEnd[i + 2] == 'L') {
      printf("ELF header appears at 0x%d (%lx): \n", i, (uintptr_t)wp->ptr + i);
      break;
    }
  for (j = i; j < i + 32; j++) printf("%.2X ", fatBinaryEnd[j] & 0xFF);
  printf("\n");

  printf("ret=%p\n", (void *)ret);
  printf("fatCubin=%p, *ret=%p\n", (void *)fatCubin, *ret);
}

void __helper_unregister_fatbin(void **fatCubinHandle) {
  // free(fatCubinHandle);
  return;
}

void __helper_parse_function_args(const char *name, struct kernel_arg *args) {
  unsigned i = 0, skip = 0;

  unsigned int argc = 0;
  if (strncmp(name, "_Z", 2)) abort();
  LOG_DEBUG << "Parse CUDA kernel " << name;

  i = 2;
  while (i < strlen(name) && isdigit(name[i])) {
    skip = skip * 10 + name[i] - '0';
    i++;
  }

  i += skip;
  while (i < strlen(name)) {
    switch (name[i]) {
    case 'P':
      args[argc++].is_handle = 1;

      /* skip qualifiers */
      if (strchr("rVK", name[i + 1]) != NULL) i++;

      if (i + 1 < strlen(name) && (strchr("fijl", name[i + 1]) != NULL))
        i++;
      else if (i + 1 < strlen(name) && isdigit(name[i + 1])) {
        skip = 0;
        while (i + 1 < strlen(name) && isdigit(name[i + 1])) {
          skip = skip * 10 + name[i + 1] - '0';
          i++;
        }
        i += skip;
      } else
        abort();
      break;

    case 'f': /* float */
    case 'i': /* int */
    case 'j': /* unsigned int */
    case 'l': /* long */
      args[argc++].is_handle = 0;
      break;

    case 'S':
      args[argc++].is_handle = 1;
      while (i < strlen(name) && name[i] != '_') i++;
      break;

    case 'v':
      i = strlen(name);
      break;

    case 'r': /* restrict (C99) */
    case 'V': /* volatile */
    case 'K': /* const */
      break;

    default:
      abort();
    }
    i++;
  }

  for (i = 0; i < argc; i++) {
    LOG_DEBUG << "function arg[" << i << "] is " << (args[i].is_handle == 1 ? "" : "not ") << "a handle";
  }
}

void __helper_record_module_path(CUmodule module, const char* fname) {
  absl::MutexLock lk(&module_path_mu);
  module_path_map[module] = std::string(fname);
}

void __helper_parse_module_function_args(CUmodule module, const char *name, struct fatbin_function **func) {
  FILE *fp_pipe = nullptr;
  char line[2048];
  unsigned int i = 0;
  int ordinal = 0;
  size_t size = 0;
  char kern_name[MAX_KERNEL_NAME_LEN]; /* mangled name */
  {
    absl::MutexLock lk(&module_path_mu);
    auto fname = module_path_map[module];
    auto pip_cmd = fmt::format("/usr/local/cuda/bin/cuobjdump -elf {}", fname);
    fp_pipe = popen(pip_cmd.c_str(), "r");
    if (fp_pipe == nullptr) {
      SYSCALL_FAILURE_PRINT("popen");
    }
  }
  while (fgets(line, sizeof(line), fp_pipe) != nullptr) {
    if (strncmp(line, ".nv.info.", 9) == 0) {
      sprintf(kern_name, line + 9, strlen(line) - 10);
      assert(strlen(line) - 10 < MAX_KERNEL_NAME_LEN);
      kern_name[strlen(line) - 10] = '\0';
      auto kern_name_length = strlen(kern_name);
      if (strncmp(kern_name, name, kern_name_length) == 0) {
        ava_debug("%s@", name);
        if (*func == nullptr) {
          *func = (struct fatbin_function *)g_malloc(sizeof(struct fatbin_function));
          memset(*func, 0, sizeof(struct fatbin_function));
        }

        /* Search parameters */
        (*func)->argc = 0;
        char* fgets_ret;
        while (fgets(line, sizeof(line), fp_pipe) != nullptr) {
          i = 0;
          while (i < strlen(line) && isspace(line[i])) i++;
          /* Empty line means reaching the end of the function info */
          if (i == strlen(line)) break;

          if (strncmp(&line[i], "Attribute:", 10) == 0) {
            i += 10;
            while (i < strlen(line) && isspace(line[i])) i++;
            if (strncmp(&line[i], "EIATTR_KPARAM_INFO", 18) == 0) {
              /* Skip the format line */
              fgets_ret = fgets(line, sizeof(line), fp_pipe);
              if (fgets_ret == NULL) {
                if (feof(fp_pipe)) {
                  fprintf(stderr, "End of file");
                } else if (ferror(fp_pipe)) {
                  SYSCALL_FAILURE_PRINT("fgets");
                }
              }
              fgets_ret = fgets(line, sizeof(line), fp_pipe);
              if (fgets_ret == NULL) {
                if (feof(fp_pipe)) {
                  fprintf(stderr, "End of file");
                } else if (ferror(fp_pipe)) {
                  SYSCALL_FAILURE_PRINT("fgets");
                }
              }
              /* Get ordinal and size */
              i = 0;
              while (i < strlen(line) && line[i] != 'O') i++;
              sscanf(&line[i], "Ordinal\t: 0x%x", (unsigned int *)&ordinal);
              while (i < strlen(line) && line[i] != 'S') i++;
              sscanf(&line[i], "Size\t: 0x%lx", &size);

              i = (*func)->argc;
              AVA_DEBUG << "ordinal=" << ordinal << ", size=" << size;
              assert(ordinal < MAX_KERNEL_ARG);
              (*func)->args[ordinal].size = size;
              (*func)->argc += 1;
            }
          }
        }
      } else {
        continue;
      }
    }
  }
}

size_t __helper_launch_extra_size(void **extra) {
  if (extra == NULL) return 0;
  size_t size = 1;
  while (extra[size - 1] != CU_LAUNCH_PARAM_END) size++;
  return size;
}

void *__helper_cu_mem_host_alloc_portable(size_t size) {
  void *p = aligned_alloc(64, size);
  assert(p && "p should not be null");
  return p;
}

void __helper_cu_mem_host_free(void *ptr) { free(ptr); }

void __helper_assosiate_function_dump(GHashTable *funcs, struct fatbin_function **func, void *local,
                                      const char *deviceName) {
  if (*func != NULL) {
    LOG_DEBUG << "Function (" << deviceName << ") metadata (" << local << ") already exists";
    return;
  }

  *func = (struct fatbin_function *)g_hash_table_lookup(funcs, deviceName);
  assert(*func && "device function not found!");
}

/**
 * Saves the async buffer information into the list inside the stream's
 * metadata.
 */
void __helper_register_async_buffer(struct async_buffer_list *buffers, void *buffer, size_t size) {
  assert(buffers->num_buffers < MAX_ASYNC_BUFFER_NUM);
  int idx = (buffers->num_buffers)++;
  LOG_VERBOSE << "Register async buffer [" << idx << "] address = " << buffer << ", size = " << size;
  buffers->buffers[idx] = buffer;
  buffers->buffer_sizes[idx] = size;
}

struct async_buffer_list *__helper_load_async_buffer_list(struct async_buffer_list *buffers) {
  if (buffers->num_buffers == 0) return NULL;

  LOG_DEBUG << "Load " << buffers->num_buffers << " async buffers";
  struct async_buffer_list *new_copy = (struct async_buffer_list *)malloc(sizeof(struct async_buffer_list));
  if (new_copy == NULL) {
    throw std::runtime_error("failed to malloc async_buffer_list");
  }
  memcpy(new_copy, buffers, sizeof(struct async_buffer_list));
  memset(buffers, 0, sizeof(struct async_buffer_list));

  return new_copy;
}

int __helper_a_last_dim_size(cublasOperation_t transa, int k, int m) {
  if (transa == CUBLAS_OP_N) {
    return k;
  } else {
    return m;
  }
}

int __helper_b_last_dim_size(cublasOperation_t transb, int k, int n) {
  if (transb == CUBLAS_OP_N) {
    return n;
  } else {
    return k;
  }
}

int __helper_type_size(cudaDataType dataType) {
  switch (dataType) {
  case CUDA_R_16F:
    return 2;
  case CUDA_C_16F:
    return 4;
  case CUDA_R_32F:
    return 4;
  case CUDA_C_32F:
    return sizeof(float _Complex);
  case CUDA_R_64F:
    return 8;
  case CUDA_C_64F:
    return sizeof(double _Complex);
  case CUDA_R_8I:
    return 1;
  case CUDA_C_8I:
    return 2;
  case CUDA_R_8U:
    return 1;
  case CUDA_C_8U:
    return 2;
  case CUDA_R_32I:
    return 4;
  case CUDA_C_32I:
    return 8;
  case CUDA_R_32U:
    return 4;
  case CUDA_C_32U:
    return 8;
  default:
    LOG_ERROR << "invalid data type: " << dataType;
    abort();
  }
}

void __helper_print_pointer_attributes(const struct cudaPointerAttributes *attributes, const void *ptr) {
  LOG_DEBUG << "Pointer " << std::hex << (uintptr_t)ptr << " attributes = {" << std::endl
#if (CUDART_VERSION >= 1000)
            << "    memoryType = " << std::dec << attributes->type << "," << std::endl
#else
            << "    memoryType = " << std::dec << attributes->memoryType << "," << std::endl
#endif
            << "    type = " << attributes->type << "," << std::endl
            << "    device = " << attributes->device << "," << std::endl
            << "    devicePointer = " << std::hex << attributes->devicePointer << "," << std::endl
            << "    hostPointer = " << attributes->hostPointer << "," << std::endl
#if (CUDART_VERSION >= 1000)
            << "    isManaged = " << std::dec << (attributes->type == cudaMemoryTypeManaged) << "," << std::endl
#else
            << "    isManaged = " << std::dec << attributes->isManaged << "," << std::endl
#endif
            << "}";
}
