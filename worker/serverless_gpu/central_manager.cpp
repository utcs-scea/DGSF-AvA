#include "svgpu_manager.hpp"
#include <zmq.h>
#include <random>
#include "extensions/memory_server/common.hpp"

#include <mutex>
/*************************************
 *
 *    Central Manager
 *
 *************************************/

using namespace GPUMemoryServer;

static uint32_t debug_kernel_count = 0;
//std::random_device rdd;
//std::mt19937 rgen(rdd());
std::mt19937 rgen{0};
std::uniform_real_distribution<float> dis01(0, 1);
//std::uniform_int_distribution<int> intdist(0,1);

uint32_t t_scheduled = 0;
uint32_t t_migrated = 0;

void SVGPUManager::centralManagerLoop() {
    void *context = zmq_ctx_new();
    void *responder = zmq_socket(context, ZMQ_REP);
    //make sure it doesn't exist
    unlink(GPUMemoryServer::get_central_socket_path().c_str());
    int rc = zmq_bind(responder, GPUMemoryServer::get_central_socket_path().c_str());
    if (rc == -1)
        printf("BIND FAILED with code %d\n", rc);

    std::cerr << "[SVLESS-MNGR]: central manager up and running" << std::endl;

    Reply rep;
    Request req;
    while(1) {
        zmq_recv(responder, &req, sizeof(Request), 0);
        //std::cerr << "[SVLESS-MNGR]: recvd" << std::endl;
        handleRequest(req, rep);
        zmq_send(responder, &rep, sizeof(Reply), 0);
        //std::cerr << "[SVLESS-MNGR]: sent" << std::endl;
    }
}

void SVGPUManager::handleRequest(Request& req, Reply& rep) {
    //reset reply
    rep.data.migration.type = Migration::NOPE;
    rep.code = ReplyCode::OK;

    /*************************
     *   schedule request
     *************************/
    if (req.type == RequestType::SCHEDULE) {
        //std::cerr << "[SVLESS-MNGR]: handling SCHEDULE message" << std::endl;
        handleSchedule(req, rep);
    }
    /*************************
     *  worker is ready
     *************************/
    else if (req.type == RequestType::READY) {
        //std::cerr << "[SVLESS-MNGR]: handling READY message" << std::endl;
        handleReady(req, rep);
    }
    /*************************
     *  cudaMalloc
     *************************/
    else if (req.type == RequestType::ALLOC) {
        handleMalloc(req, rep);
    }
    /*************************
     *       cudaFree
     *************************/
    else if (req.type == RequestType::FREE) {
        handleFree(req, rep);
    }
    /*************************
     * worker finished, cleanup
     *************************/
    else if (req.type == RequestType::FINISHED) {
        handleFinish(req, rep);
    }
    /*************************
     * store requested amount of memory
     *************************/
    else if (req.type == RequestType::MEMREQUESTED) {
        handleRequestedMemory(req, rep);
    }
    /*************************
     * kernel asking to run
     *************************/
    else if (req.type == RequestType::KERNEL_IN) {
        handleKernelIn(req, rep);
    }
    /*************************
     * kernel done
     *************************/
    else if (req.type == RequestType::KERNEL_OUT) {
        handleKernelOut(req, rep);
    }
    else {
        std::cerr << " Server got an unknown request type!\n";
    }
}

void SVGPUManager::handleSchedule(Request& req, Reply& rep) {
    int32_t port = scheduler->getGPU(req.data.size);
    if (port > 0) {
        rep.data.ready.port = port;
        rep.code = ReplyCode::OK;
        t_scheduled++;
        //std::cerr << " Scheduled succesfully: " << t_scheduled << std::endl;
    }
    else {
        rep.code = ReplyCode::RETRY;
    }
}

void SVGPUManager::handleReady(Request& req, Reply& rep) {
    auto port = req.data.ready.port;
    auto gpu = req.gpu;

    //bad code, just need to find if the worker is reusing, then we update
    for (auto& kv : gpu_workers) {
        if (kv.second.count(port) != 0){
            //update current GPU worker is in
            gpu_states[gpu].busy_workers -= 1;
            gpu_states[gpu].estimated_free_memory += kv.second[port].memory_requested;
            //it migrated, so set gpu to be the original one
            gpu = kv.first;
            fprintf(stderr, " worker ready: GPU %d now with %d memory\n", gpu, gpu_states[gpu].estimated_free_memory);
            break;
        }
    }

    //update this worker which might have been from a different GPU
    gpu_workers[gpu][port].busy = false;

    //notify backend, if there is one, that we can handle another function
    if (!resmngr_address.empty())
        resmngr_client->addGPUWorker(uuid);
}

void SVGPUManager::handleMalloc(Request& req, Reply& rep) {
    uint64_t requested = req.data.size;
    std::string worker_id(req.worker_id);

    //workers_info[worker_id].mem_used += requested;
    (void)rep;
    
    //if (requested > gpu_memory_total-gpu_memory_used) {
        //TODO: maybe check not good citizens and move them.
        //TODO: request migration by memory
    //}
}

void SVGPUManager::handleFree(Request& req, Reply& rep) {
    std::string worker_id(req.worker_id);
    //workers_info[worker_id].mem_used -= req.data.size;
    (void)rep;
}

void SVGPUManager::handleRequestedMemory(Request& req, Reply& rep) {
    std::string worker_id(req.worker_id);
    //workers_info[worker_id].requested_memory = req.data.size;
    (void)rep;
}

void SVGPUManager::handleFinish(Request& req, Reply& rep) {
    std::string worker_id(req.worker_id);
    //workers_info.erase(worker_id);
    (void)rep;
}

void SVGPUManager::handleKernelIn(Request& req, Reply& rep) {
    //check if we are just debugging
    char* dbg_mig = std::getenv("SG_DEBUG_MIGRATION");
    if (dbg_mig) {
        debug_kernel_count += 1;
        uint32_t dbgi = atoi(std::getenv("SG_DEBUG_MIGRATION"));

        //if debug type 1 or 2 and multiple of 2
        if ((dbgi == 1 || dbgi == 2)  &&  debug_kernel_count % 2 == 0) {
            printf("SG_DEBUG_MIGRATION:  kernel #%d, setting migration on\n", debug_kernel_count);

            if (!strcmp(dbg_mig, "1")) {
                printf("SG_DEBUG_MIGRATION:  setting EXECUTION migration on\n");
                rep.data.migration.type = Migration::KERNEL;
                rep.code = ReplyCode::MIGRATE;
                rep.data.migration.target_device = req.gpu == 1 ? 0 : 1;
            }
            else if (!strcmp(dbg_mig, "2")) {
                printf("SG_DEBUG_MIGRATION:  setting MEMORY migration on\n");
                rep.data.migration.type = Migration::TOTAL;
                rep.code = ReplyCode::MIGRATE;
                rep.data.migration.target_device = req.gpu == 1 ? 0 : 1;
            }
        }
        else if (dbgi == 3) {
            if (req.gpu == 0 && t_migrated == 0 && t_scheduled == 4) {
                rep.code = ReplyCode::MIGRATE;
                std::cerr << "\n\n\nSG_DEBUG_MIGRATION: ITS GOING DOWN\n\n";
                rep.data.migration.type = Migration::TOTAL;
                rep.data.migration.target_device = 1;
                t_migrated = 1;
            }
        }
        //if a multiple of 10, divide 1 by it and that's the prob
        else if (dbgi >= 10 && dbgi % 10 == 0) {
            float prob = 1.0 / dbgi;
            if (dis01(rgen) <= prob) { 
            //if (1) {
                rep.data.migration.type = Migration::TOTAL;
                rep.code = ReplyCode::MIGRATE;
                uint32_t dg = req.gpu == 0 ? 2 : 0;
                rep.data.migration.target_device = dg;
                std::cerr << " SG_DEBUG_MIGRATION: TOTAL random migration triggered:  " << req.gpu  << " -> " << dg << " with prob " << prob << std::endl;
            }

            gpu_states[req.gpu].busy_workers -= 1;
            gpu_states[rep.data.migration.target_device].busy_workers += 1;
        }
        /*
        //if a multiple of 10 after -1, divide 1 by it and that's the prob, use kernel migration
        else if (dbgi >= 11 && (dbgi-1) % 10 == 0) {
            float prob = 1.0 / dbgi;
            if (dis01(rgen) <= prob) { 
                rep.data.migration.type = Migration::KERNEL;
                rep.code = ReplyCode::MIGRATE;
                uint32_t dg = req.gpu == 0 ? 1 : 0;
                rep.data.migration.target_device = dg;
                std::cerr << " SG_DEBUG_MIGRATION: KERNEL random migration triggered:  " << req.gpu  << " -> " << dg << " with prob " << prob << std::endl;
            }
        }*/
    }

    //fast quits
    if (dbg_mig && rep.code != ReplyCode::MIGRATE ) return;
    if (!imbalance || on_cooldown()) return;

    {
        std::lock_guard<std::mutex> lg(scheduler->lock);
        if (migration_strategy != 0 && imbalance && !on_cooldown()) {
            if (req.gpu == overwhelmed_gpu
                    &&  gpu_states[underwhelmed_gpu].busy_workers == 0 ) {  //this last condition isnt really good, but whatever
                rep.data.migration.type = Migration::TOTAL;
                rep.code = ReplyCode::MIGRATE;
                rep.data.migration.target_device = underwhelmed_gpu;
                fprintf(stderr, " !!! Migrating from %d to %d\n", uint32_t(overwhelmed_gpu), uint32_t(underwhelmed_gpu));
                set_cooldown();
            }
        }

        //we are migrating, update worker counts
        if (rep.code != ReplyCode::OK) {
            gpu_states[req.gpu].busy_workers -= 1;
            gpu_states[rep.data.migration.target_device].busy_workers += 1;
            auto port = req.data.ready.port;
            gpu_states[req.gpu].estimated_free_memory +=  gpu_workers[req.gpu][port].memory_requested;
            gpu_states[rep.data.migration.target_device].estimated_free_memory -=  gpu_workers[req.gpu][port].memory_requested;
        }
    }

    (void)req;
}

void SVGPUManager::handleKernelOut(Request& req, Reply& rep) {
    (void)req; (void)rep;
    //kernels_queued--;
    //printf(" <<out: there are now %d kernels queued\n", kernels_queued);
}
