#ifndef __SCHEDULING_COMMON_HPP__
#define __SCHEDULING_COMMON_HPP__

#include <stdint.h>
#include <mutex>

struct GPUWorkerState {
  //uint32_t port;
  uint32_t gpu;
  bool busy;
  uint64_t memory_requested;
};

struct GPUState {
  std::vector<GPUWorkerState> workers;
  uint64_t total_memory;
  uint64_t free_memory; 
  uint64_t estimated_free_memory; 
  float proc_utilization;
  std::atomic<uint32_t> busy_workers; //
};

struct BaseScheduler {
  std::map<uint32_t, std::map<uint32_t, GPUWorkerState>> *workers;
  std::map<uint32_t, GPUState> *gpu_states;

  std::mutex lock;

  BaseScheduler(
      std::map<uint32_t, std::map<uint32_t, GPUWorkerState>> *workers,
      std::map<uint32_t, GPUState> *gpu_states) :
    workers(workers), gpu_states(gpu_states)
  {}

  virtual int32_t getGPU(uint64_t requested_memory) = 0;
};


#endif