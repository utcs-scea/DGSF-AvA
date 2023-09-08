#include "svgpu_manager.hpp"
#include <nvml.h>
#include <sys/wait.h>
#include <boost/algorithm/string/join.hpp>
#include "declaration.h"
#include <string>
#include <atomic>
#include <memory>
#include <mutex>
#include <zmq.h>
#include "extensions/memory_server/common.hpp"
#include "resmngr.grpc.pb.h"
#include "resmngr.pb.h"

#include "scheduling/first_fit.hpp"
#include "scheduling/best_fit.hpp"
#include "scheduling/worst_fit.hpp"

/*************************************
 *
 *    gRPC methods
 *
 *************************************/

std::string SVGPUManager::ResMngrClient::registerSelf() {
    resmngr::RegisterGPUNodeRequest request;
    std::cout << "[SVLESS-MNGR]: registerSelf to resource manager.. " << std::endl;
    ClientContext context;
    resmngr::RegisterGPUNodeResponse reply;
    Status status = stub_->RegisterGPUNode(&context, request, &reply);
    if (!status.ok()) {
        std::cerr << "Error registering self with resmngr:" << status.error_code() << ": " << status.error_message()
                << std::endl;
        std::exit(1);
    }

    return reply.uuid();
}

void SVGPUManager::ResMngrClient::addGPUWorker(std::string uuid) {
    resmngr::AddGPUWorkerRequest request;
    request.set_uuid(uuid);
    request.set_workers(1);

    std::cout << "[SVLESS-MNGR]: adding gpu worker to resource manager.. " << std::endl;
    ClientContext context;
    resmngr::AddGPUWorkerResponse reply;
    Status status = stub_->AddGPUWorker(&context, request, &reply);
    if (!status.ok()) {
        std::cerr << "Error registering self with resmngr:" << status.error_code() << ": " << status.error_message()
                << std::endl;
        std::exit(1);
    }
}

void SVGPUManager::registerSelf() {
    resmngr_client = new ResMngrClient(grpc::CreateChannel(resmngr_address, grpc::InsecureChannelCredentials()));
    uuid = resmngr_client->registerSelf();
}

/*************************************
 *
 *    AvA HandleRequest override
 *
 *************************************/

std::atomic<uint32_t> current_turn(0);
uint32_t current_id(0);
std::mutex sched_lock;

ava_proto::WorkerAssignReply SVGPUManager::HandleRequest(const ava_proto::WorkerAssignRequest &request) {
    ava_proto::WorkerAssignReply reply;

    if (request.gpu_count() > 1) {
        std::cerr << "ERR: someone requested more than 1 GPU, no bueno" << std::endl;
        return reply;
    }

    sched_lock.lock();
    uint32_t q_id = current_id;
    current_id += 1;
    sched_lock.unlock();

    uint32_t gpu_mem = request.gpu_mem()[0];

    //std::cerr << "[SVLESS-MNGR]: API server request arrived, asking for schedule with memory " << gpu_mem << std::endl;
    while (true) {
        //if not our turn, wait
        if (current_turn != current_id) {
            printf("not my turn:  me %u  turn %u\n", current_id, current_turn);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        GPUMemoryServer::Request req;
        req.type = GPUMemoryServer::RequestType::SCHEDULE;
        req.data.size = gpu_mem;
        if (zmq_send(zmq_central_socket, &req, sizeof(GPUMemoryServer::Request), 0) == -1) {
            printf(" ### zmq_send errno %d\n", errno);
        }

        GPUMemoryServer::Reply rep;
        zmq_recv(zmq_central_socket, &rep, sizeof(GPUMemoryServer::Reply), 0);
        
        if (rep.code == GPUMemoryServer::ReplyCode::OK) {
            //allow next to go
            current_turn += 1;

            std::string ip = std::getenv("SELF_IP");
            ip += ":";

            //if (std::getenv("RESMNGR_ADDR")) {
            //    ip = std::string(std::getenv("RESMNGR_ADDR"));
            //    ip += ":";
            //}

            std::cerr << "[SVLESS-MNGR]: scheduled at " << ip << std::endl;
            reply.worker_address().push_back(ip + std::to_string(rep.data.ready.port));
            return reply;
        }
        else if (rep.code == GPUMemoryServer::ReplyCode::RETRY) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
}

/*************************************
 *
 *    General stuff
 *
 *************************************/

void SVGPUManager::launchReportServers() {
    // launch central
    central_server_thread = std::thread(&SVGPUManager::centralManagerLoop, this);
    std::cerr << "[SVLESS-MNGR]: Launched central server, sleeping to give them time to spin up" << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    //lets connect ourself to the central server, so that when we need to schedule we can talk to it
    zmq_context = zmq_ctx_new();
    zmq_central_socket = zmq_socket(zmq_context, ZMQ_REQ);
    while (1) { 
        int ret = zmq_connect(zmq_central_socket, GPUMemoryServer::get_central_socket_path().c_str());
        if (ret == 0) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        printf(" !!! Manager couldn't connect to central server\n");
    }
}

SVGPUManager::SVGPUManager(uint32_t port, uint32_t worker_port_base, std::string worker_path, std::vector<std::string> &worker_argv,
            std::vector<std::string> &worker_env, uint16_t ngpus, uint16_t gpu_offset, std::string resmngr_address,
            std::string scheduler_name, uint32_t precreated_workers, std::string nvmlmonitor, uint32_t migration_strategy)
    : ManagerServiceServerBase(port, worker_port_base, worker_path, worker_argv, worker_env) {

    this->n_gpus = ngpus;
    this->gpu_offset = gpu_offset;
    this->uuid_counter = 0;
    this->resmngr_address = resmngr_address;
    this->migration_strategy = migration_strategy;
    this->migration_cooldown = 0;
    this->imbalance = false;
    //update to real values using nvml
    setRealGPUOffsetCount();

    createScheduler(scheduler_name);

    launchReportServers();
    
    if (!resmngr_address.empty())
        registerSelf();

    //now that everything is up, we can launch workers
    this->precreated_workers = precreated_workers;
    for (unsigned int gpu = gpu_offset; gpu < gpu_offset+n_gpus; gpu++) {
        for (int i = 0 ; i < precreated_workers ; i++) {
            launchWorker(gpu);
        }
    }

    if (nvmlmonitor != "no") {
        std::cerr << "[SVLESS-MNGR]: Launching NVML monitor" << std::endl;
        nvml_monitor_thread = std::thread(&SVGPUManager::nvmlMonitorLoop, this);
    }
};

uint32_t SVGPUManager::launchWorker(uint32_t gpu_id) {
    // Start from input environment variables
    std::vector<std::string> environments(worker_env_);

    std::string visible_devices = "GPU_DEVICE=" + std::to_string(gpu_id);
    environments.push_back(visible_devices);

    // Let API server use TCP channel
    environments.push_back("AVA_CHANNEL=TCP");

    std::string worker_uuid = "AVA_WORKER_UUID=" + std::to_string(uuid_counter);
    environments.push_back(worker_uuid);
    uuid_counter++;

    // Pass port to API server
    auto port = worker_port_base_ + worker_id_.fetch_add(1, std::memory_order_relaxed);
    std::vector<std::string> parameters;
    parameters.push_back(std::to_string(port));

    // Append custom API server arguments
    for (const auto &argv : worker_argv_) {
        parameters.push_back(argv);
    }

    //for (auto& element : environments) {
    //    printf("  > %s\n", element.c_str());
    //}

    std::cerr << "Spawn API server at 0.0.0.0:" << port << " (cmdline=\"" << boost::algorithm::join(environments, " ")
                << " " << boost::algorithm::join(parameters, " ") << "\")" << std::endl;

    auto child_pid = SpawnWorker(environments, parameters);

    auto child_monitor = std::make_shared<std::thread>(
        [](pid_t child_pid, uint32_t port, std::map<pid_t, std::shared_ptr<std::thread>> *worker_monitor_map) {
            pid_t ret = waitpid(child_pid, NULL, 0);
            std::cerr << "[pid=" << child_pid << "] API server at ::" << port << " has exit (waitpid=" << ret << ")"
                    << std::endl;
            worker_monitor_map->erase(port);
        },
        child_pid, port, &worker_monitor_map_);
    child_monitor->detach();
    worker_monitor_map_.insert({port, child_monitor});

    return port;
}

void SVGPUManager::setRealGPUOffsetCount() {
    nvmlReturn_t result;
    result = nvmlInit();
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to initialize NVML: " << nvmlErrorString(result) << std::endl;
        std::exit(1);
    }

    result = nvmlDeviceGetCount(&device_count);
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to query device count NVML: " << nvmlErrorString(result) << std::endl;
        std::exit(1);
    }

    result = nvmlShutdown();
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to shutdown NVML: " << nvmlErrorString(result) << std::endl;
        std::exit(1);
    }

    // bounds check requested gpus, use all gpus if n_gpus == 0
    if (n_gpus == 0) {
        n_gpus = device_count - gpu_offset;

    } 

    
    //if we want to dedicate a GPU for migration, we can set device count to 4 but only launch workers on the other 3
    
    //test
    device_count = n_gpus;
    
    //for (int i = gpu_offset ; i < gpu_offset+n_gpus ; i++) {
    for (int i = 0 ; i < n_gpus ; i++) {
        nvmlDevice_t dev;
        nvmlDeviceGetHandleByIndex_v2(i, &dev);
        nvmlMemory_t utmem;
        nvmlDeviceGetMemoryInfo(dev, &utmem);
        gpu_states[i].total_memory = utmem.free; //free so we can ignore overhead of contexts
        gpu_states[i].free_memory = utmem.free;
        gpu_states[i].estimated_free_memory = 12000; //utmem.free - (3*1024*1024*1024);
        gpu_states[i].proc_utilization = 0;
        gpu_states[i].busy_workers = 0;
    }
    
    std::cout << "[SVLESS-MNGR]: set GPU offset to " << gpu_offset << " and GPU count to " << n_gpus << std::endl;
}

void SVGPUManager::createScheduler(std::string name) {
    /*if (name == "firstfit") {
        this->scheduler = new FirstFit(&gpu_workers, &gpu_states);
        std::cout << "[SVLESS-MNGR]: Using First Fit scheduler\n";
    }
    else*/ 
    if (name == "bestfit") {
        this->scheduler = new BestFit(&gpu_workers, &gpu_states);
        std::cout << "[SVLESS-MNGR]: Using Best Fit scheduler\n";
    }
    else if (name == "worstfit") {
        this->scheduler = new WorstFit(&gpu_workers, &gpu_states);
        std::cout << "[SVLESS-MNGR]: Using Worst Fit scheduler\n";
    }
    //default
    else {
        this->scheduler = new BestFit(&gpu_workers, &gpu_states);
        std::cout << "[SVLESS-MNGR]: Using Best Fit scheduler\n";
    }
}

void SVGPUManager::nvmlMonitorLoop() {
    nvmlReturn_t result;
    result = nvmlInit();
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to initialize NVML: " << nvmlErrorString(result) << std::endl;
        std::exit(1);
    }

    nvmlDeviceGetCount(&real_device_count);

    nvmlDevice_t devs[real_device_count];
    for (int i = 0 ; i < real_device_count ; i++) {
        nvmlDeviceGetHandleByIndex_v2(i, devs+i);
    }
    std::cout << "[SVLESS-MNGR]: NVML monitor found " << real_device_count << " devices, looping..\n";

    nvmlUtilization_t ut;
    float utils[real_device_count];
    nvmlMemory_t utmem;
    uint64_t gpu_free_mem[real_device_count];
    int32_t SMA_sample_size = 5;
    float SMA_samples[real_device_count][SMA_sample_size];
    float SMA[real_device_count];
    float timestamp_sec = 0;
    uint64_t count = 0;

    while (1) {
        for (int i = 0 ; i < real_device_count ; i++) {
            nvmlDeviceGetUtilizationRates(devs[i], &ut);
            utils[i] = ut.gpu;
            SMA_samples[i][count%SMA_sample_size] = ut.gpu;

            nvmlDeviceGetMemoryInfo(devs[i], &utmem);
            gpu_free_mem[i] = utmem.free;
        }

        //probably not a performance bottleneck, but not optimun
        if (count >= SMA_sample_size) {
            for (int i = 0 ; i < device_count ; i++) {
                float sum = 0;
                for (int j = 0 ; j < SMA_sample_size ; j++) {
                    sum += SMA_samples[i][j];
                }
                SMA[i] = sum/SMA_sample_size;
                //printf("SMA of GPU [%d] at %f:  %.2f\n", i, timestamp_sec, SMA[i]);
            }
        }

        //update structs of each GPU
        gpu_states_lock.lock();
        for (int i = 0 ; i < device_count ; i++) {
            gpu_states[i].free_memory = gpu_free_mem[i]/(1024*1024); //transform from B to MB
            gpu_states[i].proc_utilization = SMA[i];
        }
        gpu_states_lock.unlock();

        if (count % print_every == 0) 
            print_stats();

        if (on_cooldown()) {
            migration_cooldown -= 1;
        } else if (migration_strategy == 1)
            check_for_imbalance_strat1();
        else if (migration_strategy == 2)
            check_for_imbalance_strat2();

        timestamp_sec += (timestep_msec/1000);
        count++;
        std::this_thread::sleep_for(std::chrono::milliseconds(timestep_msec));
    }
}

void SVGPUManager::print_stats() {
    printf("estm\t\t");
    for (int i = 0 ; i < real_device_count ; i++)
        printf("%luM\t", gpu_states[i].estimated_free_memory);
    printf("\nmem\t\t");
    for (int i = 0 ; i < real_device_count ; i++)
        printf("%luM\t", gpu_states[i].free_memory);
    printf("\nproc\t\t");
    for (int i = 0 ; i < real_device_count ; i++)
        printf("%.1f%\t", gpu_states[i].proc_utilization);
    printf("\nwrks\t\t");
    for (int i = 0 ; i < real_device_count ; i++)
        printf("%lu\t", uint32_t(gpu_states[i].busy_workers));
    printf("\n\n");
}

void SVGPUManager::check_for_imbalance_strat1() {
    int32_t over_candidate = -1;
    int32_t under_candidate = -1;
    float proc_util_candidate = 0;

    gpu_states_lock.lock();    
    for (int i = 0 ; i < device_count ; i++) {
        if (uint32_t(gpu_states[i].busy_workers) > 1) {
            //if we already have a candidate, and i is more utilized, then update
            if (over_candidate != -1 && gpu_states[i].proc_utilization > proc_util_candidate) {
                over_candidate = i;
                proc_util_candidate = gpu_states[i].proc_utilization;
            } else {
                over_candidate = i;
                proc_util_candidate = gpu_states[i].proc_utilization;
            }
        }
    }
    gpu_states_lock.unlock();
    imbalance = false;

    if (over_candidate != -1) 
        printf(" 1/2 - Found a candidate for migration: GPU [%d] w/ [%d] workers and util [%.2f]\n", 
                over_candidate, uint32_t(gpu_states[over_candidate].busy_workers), proc_util_candidate);
    //if we didnt find it, dont waste time
    else return;

    for (int i = 0 ; i < device_count ; i++) {
        if (uint32_t(gpu_states[i].busy_workers) == 0) {
            under_candidate = i;
            break;
        }
    }

    if (under_candidate != -1) 
        printf("2/2 - Found a candidate to migration to: GPU [%d] w/ [%d]\n", 
                under_candidate, uint32_t(gpu_states[under_candidate].busy_workers));
    //if we didnt find it, dont waste time
    else return;
    
    imbalance = true;
    overwhelmed_gpu = over_candidate;
    underwhelmed_gpu = under_candidate;
}

void SVGPUManager::check_for_imbalance_strat2() {
    //TBD
}

void SVGPUManager::set_cooldown() {
    migration_cooldown = cooldown_length;
    imbalance = false;
}

bool SVGPUManager::on_cooldown() {
    return migration_cooldown != 0;
}
