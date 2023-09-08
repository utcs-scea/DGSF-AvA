#ifndef __SVGPU_MANAGER_HPP__
#define __SVGPU_MANAGER_HPP__

#include <grpcpp/grpcpp.h>

#include <map>
#include <vector>
#include <mutex>

#include "manager_service.hpp"
#include "manager_service.proto.h"
#include "resmngr.grpc.pb.h"
#include "resmngr.pb.h"
#include "extensions/memory_server/common.hpp"

#include "scheduling/common.hpp"

using ava_manager::ManagerServiceServerBase;
using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using resmngr::ResMngrService;
using namespace GPUMemoryServer;

struct SVGPUManager : public ManagerServiceServerBase {
  /**************************************
   *        INTERNAL CLASSES
   **************************************/
    struct ResMngrClient {
        ResMngrClient(std::shared_ptr<Channel> channel) : stub_(ResMngrService::NewStub(channel)) {}
        std::string registerSelf();
        void addGPUWorker(std::string uuid);
        std::unique_ptr<ResMngrService::Stub> stub_;
    };


    /**************************************
     *        FIELDS
     **************************************/
    // gRPC related fields
    std::string resmngr_address;
    ResMngrClient *resmngr_client;
    std::string uuid;

    // internal state
    uint32_t n_gpus, gpu_offset;
    uint32_t uuid_counter;
    uint32_t precreated_workers;
    uint32_t device_count;
    uint32_t real_device_count;

    // GPU and worker information
    BaseScheduler *scheduler;
    std::thread central_server_thread;
    void* zmq_context;
    void* zmq_central_socket;

    //nvml monitor
    std::thread nvml_monitor_thread;

    const uint32_t timestep_msec = 250;
    const uint32_t print_every = 5;

    std::mutex gpu_states_lock;
    std::map<uint32_t, GPUState> gpu_states;
    std::map<uint32_t, std::map<uint32_t, GPUWorkerState>> gpu_workers;

    //migration stuff
    uint32_t migration_strategy;
    std::atomic<uint32_t> migration_cooldown;
    //cooldown length is multipled by timestep_msec
    const uint32_t cooldown_length = 20; //20 = 5s
    std::atomic<uint8_t> imbalance;
    std::atomic<uint32_t> overwhelmed_gpu, underwhelmed_gpu;

    void set_cooldown();
    bool on_cooldown();
    void check_for_imbalance_strat1();
    void check_for_imbalance_strat2();
    void print_stats();
    
    /**************************************
     *        METHODS
     **************************************/
    SVGPUManager(uint32_t port, uint32_t worker_port_base, std::string worker_path, 
            std::vector<std::string> &worker_argv, std::vector<std::string> &worker_env, 
            uint16_t ngpus, uint16_t gpu_offset, std::string resmngr_address, 
            std::string scheduler_name, uint32_t precreated_workers, std::string nvmlmonitor,
            uint32_t migration_strategy);

    void setRealGPUOffsetCount();
    void registerSelf();
    void launchReportServers();
    void centralManagerLoop();
    void nvmlMonitorLoop();
    uint32_t launchWorker(uint32_t gpu_id);
    void createScheduler(std::string name);
    //ava override
    ava_proto::WorkerAssignReply HandleRequest(const ava_proto::WorkerAssignRequest &request) override;

    void handleRequest(Request& req, Reply& rep);
    void handleMalloc(Request& req, Reply& rep);
    void handleFree(Request& req, Reply& rep);
    void handleRequestedMemory(Request& req, Reply& rep);
    void handleFinish(Request& req, Reply& rep);
    void handleKernelIn(Request& req, Reply& rep);
    void handleKernelOut(Request& req, Reply& rep);
    void handleReady(Request& req, Reply& rep);
    void handleSchedule(Request& req, Reply& rep);
};

#endif
