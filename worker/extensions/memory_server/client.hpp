#ifndef __MEMORYSERVER_CLIENT_HPP__
#define __MEMORYSERVER_CLIENT_HPP__

#include <stdint.h>
#include <string>
#include <zmq.h>
#include <map>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <memory>
//#include <absl/containers/flat_hash_map.h>
#include <unordered_map>
#include <mutex>
#include "common.hpp"

#ifdef __cplusplus
extern "C" {
#endif

CUresult __internal_cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                   unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                                   unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra);
cudaError_t __internal_cudaMalloc(void **devPtr, size_t size);
CUresult __internal_cuMemAlloc(CUdeviceptr *dptr, size_t bytesize);
CUresult __internal_cuMemFree(CUdeviceptr dptr);
cudaError_t __internal_cudaFree(void *devPtr);
void __internal_kernelIn();
void __internal_kernelOut();

#ifdef __cplusplus
}
#endif

bool __internal_allContextsEnabled(); 
bool __internal_setAllContextsEnabled(bool f);
uint32_t __internal_getCurrentDevice(); 
int32_t __internal_getDeviceCount();
void __internal_setDeviceCount(uint32_t dc);
cudaStream_t __translate_stream(cudaStream_t key);
cudaEvent_t __translate_event(cudaEvent_t key);

void __cmd_handle_in();
void __cmd_handle_out();

namespace GPUMemoryServer {

    class Client {
        public:
        //zmq and comms stuff
        void* context;
        void* central_socket;
        std::mutex sockmtx;
        bool enable_reporting;

        //device management
        int current_device, og_device;
        int device_count;
        Migration migrated_type;
        uint32_t listen_port;

        std::string uuid;

        //local mallocs
        struct LocalAlloc {
            uint64_t devptr;
            size_t size;
            uint32_t device_id;
            CUmemAccessDesc accessDesc;
            CUmemGenericAllocationHandle phys_mem_handle;
            
            LocalAlloc(uint32_t device);
            ~LocalAlloc();
            int cudaMalloc(size_t size);
            int physAlloc(size_t size);
            int reserveVaddr();
            int map_at(uint64_t va_ptr);
            int unmap(uint64_t va_ptr);
            int release_phys_handle();
            int moveTo(LocalAlloc* dest);

            // assume there are 8 address spaces (3 bits), each with 32GB (double what's required for 
            //P100s, which has 16GB)
            inline uint64_t vasBitmask() {
                //actually, I don't think we need to split the VA space
                //and instead can just put everything together since its UVA
                return 0;
                //return 0x00007fc000000000;
                /*
                uint64_t mask = va_id;
                mask = mask << 38-1-4;
                uint64_t base_ptr = 0x00007fc000000000;
                return base_ptr | mask;
                */
            }
        };

        std::vector<std::unique_ptr<LocalAlloc>> local_allocs;

        //stream translation
        std::map<cudaStream_t, std::map<uint32_t,cudaStream_t>> streams_map;
        //event translation
        std::map<cudaEvent_t, std::map<uint32_t,cudaEvent_t>> events_map;

        //migration
        void migrateToGPU(uint32_t new_gpuid, Migration migration_type);

        void setListenPort(uint32_t port) {
            listen_port = port;
        }
        void setUuid(std::string id) {
            uuid = id;
        }
        cudaError_t localMalloc(void** devPtr, size_t size);
        cudaError_t localFree(void* devPtr);
        void notifyReady();
        void fullCleanup();
        void kernelIn();
        void kernelOut();
        void resetCurrentGPU();
        void setCurrentGPU(int id);
        void connectToCentral();
        void reportMalloc(uint64_t size);
        void reportFree(uint64_t size);
        void reportCleanup(uint32_t gpuid);
        void reportMemoryRequested(uint64_t mem_mb);

        static Client& getInstance() {
            static Client instance;
            return instance;
        }

        private:
        void sendRequest(Request &req);
        void handleReply(Reply& reply);

        Client();
        ~Client();

        Client(Client const&)         = delete;
        void operator=(Client const&) = delete;
    };
}

#endif
