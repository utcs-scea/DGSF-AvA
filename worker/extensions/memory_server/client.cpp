#include <stdint.h>
#include <string>
#include <sstream>
#include <iostream>
#include <map>
#include <string.h>
#include <stdlib.h>
#include <zmq.h>
#include <errno.h>
#include <memory>
#include <chrono>
#include <thread>
#include <condition_variable>
#include <mutex>
#include "common/extensions/cudart_10.1_utilities.hpp"
#include "common/common_context.h"
#include "extensions/memory_server/client.hpp"
#include "extensions/memory_server/common.hpp"
#include <thread>

/**************************************
 * 
 *    Global functions, used by spec and helper functions. Mostly translate a global call to a client call
 * 
 * ************************************/
std::atomic<uint32_t> pending_cmds(0);
std::atomic<bool> thread_block(false);
std::mutex thread_block_mutex;
std::condition_variable thread_block_cv;

void wait_for_cmd_drain() {
    thread_block = true;
    //we count as a cmd, so wait for 1 pending
    while (pending_cmds != 1) ;
}

void release_cmd_handlers() {
  std::unique_lock<std::mutex> lk(thread_block_mutex);
  thread_block = false;
  thread_block_cv.notify_all();
}

void __cmd_handle_in() {
    if (thread_block) {
        std::unique_lock<std::mutex> lk(thread_block_mutex);
        while (thread_block)
            thread_block_cv.wait(lk);
    }
    pending_cmds++;
#ifndef NDEBUG
    //std::cerr << "in: " << pending_cmds << " pending cmds.." << std::endl;
#endif
}

void __cmd_handle_out() {
    pending_cmds--;
#ifndef NDEBUG
    //std::cerr << "out: " << pending_cmds << " pending cmds.." << std::endl;
#endif
}

bool allContextsEnabled;
bool __internal_allContextsEnabled() {
    return allContextsEnabled;
}
bool __internal_setAllContextsEnabled(bool f) {
    allContextsEnabled = f;
}

int32_t __internal_getDeviceCount() {
    return GPUMemoryServer::Client::getInstance().device_count;
}

void __internal_setDeviceCount(uint32_t dc) {
    GPUMemoryServer::Client::getInstance().device_count = dc;
}

uint32_t __internal_getCurrentDevice() {
    return GPUMemoryServer::Client::getInstance().current_device;
}

void __internal_kernelIn() {
    GPUMemoryServer::Client::getInstance().kernelIn();
}

void __internal_kernelOut() {
    GPUMemoryServer::Client::getInstance().kernelOut();
}

// this is only used for tensorflow
//TODO: to migrate we need to refactor Zhitings modification in a culaunch branch in here
CUresult __internal_cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                                unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra) {
    
    //__internal_kernelIn();
    return cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                sharedMemBytes, hStream, kernelParams, extra);
}

CUresult __internal_cuMemFree(CUdeviceptr dptr) {
    cudaError_t err = GPUMemoryServer::Client::getInstance().localFree((void*)dptr);
    return (CUresult) err;
}

cudaError_t __internal_cudaMalloc(void **devPtr, size_t size) {
    return GPUMemoryServer::Client::getInstance().localMalloc(devPtr, size);
}

cudaError_t __internal_cudaFree(void *devPtr) {
    return GPUMemoryServer::Client::getInstance().localFree(devPtr);
}

cudaStream_t __translate_stream(cudaStream_t key) {
    if (__internal_allContextsEnabled()) {
        if (key == 0) return 0;
        uint32_t cur_dvc = __internal_getCurrentDevice();
        auto v = GPUMemoryServer::Client::getInstance().streams_map[key];
        return v[cur_dvc];
    }
    else {
        return key;
    }
}

cudaEvent_t __translate_event(cudaEvent_t key) {
    if (__internal_allContextsEnabled()) {
        uint32_t cur_dvc = __internal_getCurrentDevice();
        auto v = GPUMemoryServer::Client::getInstance().events_map[key];
        return v[cur_dvc];
    }
    else {
        return key;
    }
}

/**************************************
 * 
 *    Functions for GPUMemoryServer namespace, Client class
 * 
 * ************************************/

namespace GPUMemoryServer {

    //uint32_t kc = 0;
    Client::Client() {
        og_device = -1;
        context = zmq_ctx_new();
        migrated_type = Migration::NOPE;

        connectToCentral();
     }

     Client::~Client() {
        zmq_close(central_socket);
    }

    void Client::connectToCentral() {
        central_socket = zmq_socket(context, ZMQ_REQ);
        while (1) { 
            int ret = zmq_connect(central_socket, GPUMemoryServer::get_central_socket_path().c_str());
            if (ret == 0) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            fprintf( stderr, " !!! GPU Client couldn't connect to central server [%d]\n");
        }
        //fprintf( stderr, "GPU Client succesfully connected to server [%d]\n", gpuid);
    }

    cudaError_t Client::localMalloc(void** devPtr, size_t size) {
        if (size == 0) return 0;

        //test with default
        //cudaError_t err = cudaMalloc(devPtr, size);
        //fprintf( stderr,"### localMalloc request of %zx  returned %d\n", size, err);
        //return err;

        auto al = std::make_unique<Client::LocalAlloc>(current_device);
        int ret = al->cudaMalloc(size);
        //fprintf( stderr,"### localMalloc request of %zx  returned %d\n", size, ret);

        if (size == 0) fprintf( stderr,"how is this thing 0?\n");
        if (ret) return cudaErrorMemoryAllocation;

        *devPtr = (void*)al->devptr;
        local_allocs.push_back(std::move(al));
        //reportMalloc(size); 
        return cudaSuccess;

    }

    cudaError_t Client::localFree(void* devPtr) {
        // cudaFree with nullptr does no operation
        if (devPtr == nullptr) {
            return cudaSuccess;
        }

        //return cudaFree(devPtr);

        for (auto it = local_allocs.begin(); it != local_allocs.end(); ++it) {
            if ((*it)->devptr == devPtr) {
                local_allocs.erase(it);
                //GPUMemoryServer::Client::getInstance().reportFree(it->second->size); 
                return cudaSuccess;
            }
        }
        
        fprintf(stderr, "### ILLEGAL cudaFree call on devPtr %x\n", (uint64_t)devPtr);
        return cudaErrorInvalidValue;   

    }

    void Client::sendRequest(Request &req) {
        //just quit if we are not reporting to gpu server, except READY, which must go through
        if (enable_reporting == false && req.type != READY) {
            return;
        }

        req.gpu = current_device;
        req.data.ready.port = listen_port;
        strncpy(req.worker_id, uuid.c_str(), MAX_UUID_LEN);
        if (strlen(uuid.c_str()) > MAX_UUID_LEN) {
            fprintf( stderr, " ### uuid %S IS LONGER THAN %d, THIS IS AN ISSUE\n", uuid.c_str(), MAX_UUID_LEN);
        }

        sockmtx.lock();
        if (zmq_send(central_socket, &req, sizeof(Request), 0) == -1) {
            fprintf( stderr, " ### zmq_send errno %d\n", errno);
        }
        Reply rep;
        zmq_recv(central_socket, &rep, sizeof(Reply), 0);
        sockmtx.unlock();

        //unlock and handle it
        handleReply(rep);
    }

    void Client::handleReply(Reply& reply) {
        //only migrate if server told us so and we haven't migrated before
        if (reply.code == ReplyCode::MIGRATE && current_device == og_device)
            migrateToGPU(reply.data.migration.target_device, reply.data.migration.type);
    }

    void Client::setCurrentGPU(int id) {
        cudaError_t err = cudaSetDevice(id);
        if (err) {
            fprintf( stderr, "### CUDA err on cudaSetDevice(%d): %d\n", id, err);
            exit(1);
        }
        //fprintf( stderr, "Worker [%s] setting current device to %d\n", uuid.c_str(), id);
        //this if triggers once, when worker is setting up
        if (og_device == -1) 
            og_device = id;
        current_device = id;

        //set on context so shadow threads also change context
        auto ccontext = ava::CommonContext::instance();
        ccontext->current_device = id;
    }

    void Client::resetCurrentGPU() {
        if (current_device != og_device) 
            fprintf( stderr, " ### Worker reseting GPU from (%d) to (%d)\n", current_device, og_device);
        
        current_device = og_device;
        setCurrentGPU(current_device);
        auto ccontext = ava::CommonContext::instance();
        ccontext->current_device = current_device;
        migrated_type = Migration::NOPE;
    }

    void Client::migrateToGPU(uint32_t new_gpuid, Migration migration_type) {
        std::cerr << "[[[MIGRATION]]]\n" << std::endl;
        
        //wait for all cmd handlers to stop
        wait_for_cmd_drain();
        cudaDeviceSynchronize();
        
        //this_thread::sleep_for(chrono::milliseconds(2000) );
        cudaDeviceSynchronize();

        cudaError_t ret2 = cudaGetLastError();
        if(ret2) 
            std::cerr << "\n ### migrateToGPU error after sync " << ret2 << "\n";

        std::cerr << "All cmd handlers are stopped.. migrating\n" << std::endl;
        //mark that we migrated
        migrated_type = migration_type;

        setCurrentGPU(new_gpuid);
        cudaDeviceSynchronize();

        if (migration_type == Migration::KERNEL) {
            fprintf( stderr, "Migration by kernel mode, changing device to [%d]\n", new_gpuid);
            //cudaDeviceEnablePeerAccess(new_gpuid, 0);
            setCurrentGPU(new_gpuid);
            fprintf( stderr, "cudaDeviceEnablePeerAccess: new device [%d] can access memory on old [%d]\n", new_gpuid, og_device);
            cudaDeviceEnablePeerAccess(og_device, 0);
        }
        else if (migration_type == Migration::TOTAL) {
            auto start = chrono::steady_clock::now();
            std::cerr << "Full migration, changing device to" << new_gpuid << std::endl;
            //all we gotta do is change all allocations to new device
            std::vector<LocalAlloc*> new_allocs;

            // size_t free, total;
            // cudaMemGetInfo(&free, &total); 
            // printf("Free memory in new GPU before: %lu\n", free/(1024*1024));
            // setCurrentGPU(og_device);
            // cudaMemGetInfo(&free, &total); 
            // setCurrentGPU(new_gpuid);
            // printf("Free memory in OLD GPU before: %lu\n", free/(1024*1024));

            auto it = local_allocs.begin();
            while (it != local_allocs.end()) {
                //fprintf( stderr, "moving allc at %p\n", (*it)->devptr);
                auto new_alloc = new Client::LocalAlloc(new_gpuid);
                (*it)->moveTo(new_alloc);
                new_allocs.push_back(new_alloc);
                //fprintf( stderr, "erasing%p\n");
                it = local_allocs.erase(it);
            }

            // cudaMemGetInfo(&free, &total); 
            // printf("Free memory in new GPU after: %lu\n", free/(1024*1024));

            local_allocs.clear();
            for (auto const& al : new_allocs) {
                std::unique_ptr<LocalAlloc> ual(al);
                local_allocs.push_back(std::move(ual));
            }

            // setCurrentGPU(og_device);
            // cudaMemGetInfo(&free, &total); 
            // setCurrentGPU(new_gpuid);
            // printf("Free memory in OLD GPU AFTER: %lu\n", free/(1024*1024));

            // for (auto const& al : local_allocs) {
            //     fprintf( stderr, "testing access at %p after new mapping\n", al->devptr);
            //     char a[8];
            //     cudaMemcpy(a, (void*)al->devptr, 8, cudaMemcpyDeviceToHost );
            //     fprintf( stderr, "first 8 bytes of array\n");
            //     for (int j = 0 ; j < 8 ; j++)
            //         fprintf(stderr, "%x ", a[j]);
            //     fprintf( stderr, "\n\n");
            // }

            fprintf( stderr, "migration done, syncing\n");
            cudaDeviceSynchronize();
            auto end = chrono::steady_clock::now();

            fprintf(stderr, " !!! Migration took %d  us\n", chrono::duration_cast<chrono::microseconds>(end - start).count());
        }

        //release handlers
        release_cmd_handlers();
    }

    void Client::fullCleanup() {
        reportCleanup(0);  //TODO

        //clear all allocated memory
        local_allocs.clear();

        //iterate over map of maps, destroying streams
        while (!streams_map.empty())
            __helper_destroy_stream((streams_map.begin())->first);
        streams_map.clear();

        //clear leftover events
        while (!events_map.empty()) 
            __helper_cudaEventDestroy((events_map.begin())->first);
        events_map.clear();
    }

    void Client::notifyReady() {
        Request req;
        req.type = RequestType::READY;
        //std::cerr << "  worker on port " << listen_port << " sending ready message" << std::endl;
        sendRequest(req);
    }

    void Client::reportCleanup(uint32_t gpuid) {
        Request req;
        req.type = RequestType::FINISHED;
        //TODO: report cleanup only device
        //sendRequest(req);
    }

    void Client::kernelIn() {
        //fprintf( stderr, "%d kernels\n", ++kc);
        if (migrated_type != Migration::NOPE) {
            //if we have migrated, lets just skip..
            return;
        }
        Request req;
        req.type = RequestType::KERNEL_IN;
        sendRequest(req);
    }
    
    void Client::kernelOut() {
        Request req;
        req.type = RequestType::KERNEL_OUT;
        //sendRequest(req);
    }

        void Client::reportFree(uint64_t size) {
        Request req;
        req.type = RequestType::FREE;
        req.data.size = size;
        sendRequest(req);
    }

    void Client::reportMemoryRequested(uint64_t mem_mb) {
        Request req;
        req.type = RequestType::MEMREQUESTED;
        req.data.size = mem_mb;
        //sendRequest(req);
    }

    void Client::reportMalloc(uint64_t size) {
        Request req;
        req.type = RequestType::ALLOC;
        req.data.size = size;
        sendRequest(req);
    }
    
    Client::LocalAlloc::LocalAlloc(uint32_t device) {
        devptr = 0;
        device_id = device;
        accessDesc = {};
    }

    Client::LocalAlloc::~LocalAlloc() {
        cuMemUnmap(devptr, size);
        cuMemAddressFree(devptr, size);
    }

    int Client::LocalAlloc::cudaMalloc(size_t size) {
        int ret;
        ret = physAlloc(size);
        if (ret) return ret;
        ret = reserveVaddr();
        //if reserve failed we must free phys allocation
        if (ret) {
            release_phys_handle();
            return ret;
        }
        ret = map_at(devptr);
        if (ret) return ret;
        release_phys_handle();
        //TODO: cleanup if something fails
        return 0;
    }

    int Client::LocalAlloc::physAlloc(size_t req_size) {
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device_id;
        accessDesc.location = prop.location;
        accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

        size_t aligned_sz;
        CUresult err = cuMemGetAllocationGranularity(&aligned_sz, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
        if (err) { fprintf( stderr, "error at cuMemGetAllocationGranularity\n"); return 2; }
        size = ((req_size + aligned_sz - 1) / aligned_sz) * aligned_sz;
        //fprintf( stderr, "cuMemCreate of %zx -> %zx on gpu %d\n", req_size, size, device_id);

        err = cuMemCreate(&phys_mem_handle, size, &prop, 0);
        //if (err) { fprintf( stderr, "error at cuMemCreate %d\n", err); return err; }
        if (err) return err;
        return 0;
    }

    int Client::LocalAlloc::reserveVaddr() {
        uint64_t addr = vasBitmask();
        CUresult err = cuMemAddressReserve(&devptr, size, 0ULL, addr, 0ULL);
        if (err) { fprintf( stderr, "error at cuMemAddressReserve %d\n", err); return err; }
        //fprintf( stderr, "VA reserve at %p, got back %p\n", addr, devptr);
        return 0;
    }

    int Client::LocalAlloc::map_at(uint64_t va_ptr) {
        CUresult err = cuMemMap(va_ptr, size, 0ULL, phys_mem_handle, 0ULL);
        if (err) { fprintf( stderr, "error at cuMemMap\n"); return 2; }
        err = cuMemSetAccess(va_ptr, size, &accessDesc, 1ULL);
        if (err) { fprintf( stderr, "error at cuMemSetAccess\n"); return 2; }
        return 0;
    }

    int Client::LocalAlloc::unmap(uint64_t va_ptr) {
        cuMemUnmap(va_ptr, size);
    }

    int Client::LocalAlloc::release_phys_handle() {
        //https://github.com/NVIDIA/cuda-samples/blob/master/Samples/vectorAddMMAP/multidevicealloc_memmap.cpp#L127
        //we can release here bc it will stay alive until it is unmapped, then it will be really freed
        cuMemRelease(phys_mem_handle);
    }

    //dest has the correct device set
    int Client::LocalAlloc::moveTo(LocalAlloc* dest) {
        //fprintf(stderr, "Copying LocalAlloc from %d to %d  pointer [%x]\n", this->device_id, dest->device_id, this->devptr);
        int err;

        cudaSetDevice(this->device_id);
        
        fprintf( stderr, "testing access before it all\n");
        char a[8];
        cudaMemcpy(a, (void*)devptr, 8, cudaMemcpyDeviceToHost );
        fprintf( stderr, "first 8 bytes of array\n");
        for (int j = 0 ; j < 8 ; j++)
            fprintf(stderr, "%x ", a[j]);
        fprintf( stderr, "\n");
        

        // allocate on new gpu
        err = dest->physAlloc(this->size);
        if (err) { fprintf( stderr, "error at map_at %d\n", err); return err; }

        dest->devptr = this->devptr;

        //get new mapping for our memory
        this->reserveVaddr();
        err = this->map_at(this->devptr);
        if (err) { fprintf( stderr, "error at map_at %d\n", err); return err; }

        err = this->unmap(dest->devptr);
        if (err) { fprintf( stderr, "error at unmap %d\n", err); return err; }

        
        fprintf( stderr, "testing access at %p after new mapping\n", devptr);
        cudaMemcpy(a, (void*)devptr, 8, cudaMemcpyDeviceToHost );
        //fprintf( stderr, "first 8 bytes of array\n");
        //for (int j = 0 ; j < 8 ; j++)
        //    fprintf(stderr, "%x ", a[j]);
        //fprintf( stderr, "\n");
        
        
        err = dest->map_at(dest->devptr);
        if (err) { fprintf( stderr, "error at map_at %d\n", err); return err; }

        fprintf(stderr, "new ptr is at %x\n", dest->devptr);

        err = dest->release_phys_handle();
        if (err) { fprintf( stderr, "error at release_phys_handle %d\n", err); return err; }

        cudaError_t err3 = cudaMemcpyPeer((void*) dest->devptr, dest->device_id, (void*) this->devptr, this->device_id, size);
        if (err3) { fprintf( stderr, "error at cudaMemcpyPeer  %d\n", err3); return 1; }

        cudaSetDevice(dest->device_id);
        
        fprintf( stderr, "testing access at %p after memcpy\n", dest->devptr);
        cudaMemcpy(a, (void*)dest->devptr, 8, cudaMemcpyDeviceToHost );
        //fprintf( stderr, "first 8 bytes of array\n");
        //for (int j = 0 ; j < 8 ; j++)
        //    fprintf(stderr, "%x ", a[j]);
        //fprintf( stderr, "\n");
        
    }

//end of GPUMemoryServer nameserver
}