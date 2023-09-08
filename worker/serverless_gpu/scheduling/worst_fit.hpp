#ifndef __WORSTFIT_SCHEDULER_HPP__
#define __WORSTFIT_SCHEDULER_HPP__

#include "common.hpp"
#include <map>
#include <mutex>

struct WorstFit : public BaseScheduler {

  WorstFit(
      std::map<uint32_t, std::map<uint32_t, GPUWorkerState>> *workers,
      std::map<uint32_t, GPUState> *gpu_states) :
    BaseScheduler(workers, gpu_states)
  {}

  int32_t getGPU(uint64_t requested_memory) {
    int32_t best_gpu = -1;
    uint64_t best_mem = -1;

    std::lock_guard<std::mutex> lg(lock);

    for (auto& gpus : *gpu_states) {
      //uint64_t free_memory = gpus.second.free_memory;
      uint64_t free_memory = gpus.second.estimated_free_memory;
      //std::cerr << "gpu " << gpus.first << " has free memory: " << free_memory <<std::endl;
      //if there is enough memory
      if (requested_memory <= free_memory) {
        //and if there is an idle worker on this gpu
        bool free_worker_exists = false;
        for (auto& gpu_wks : (*workers)[gpus.first]) {
          if (gpu_wks.second.busy == false) {
            free_worker_exists = true;
          }
        }
        
        if (free_worker_exists) {
          if (best_gpu == -1 || best_mem < free_memory ) {
            best_gpu = gpus.first;
            best_mem = free_memory;
            fprintf(stderr, " worst: GPU %d with %d of requested %d\n", best_gpu, free_memory, requested_memory);
          }
        }
      }
    }

    //we didnt find one, so we need to reply retry
    if (best_gpu == -1) {
      std::cerr << "no gpus available for the request, retry. " << requested_memory << " MB" << std::endl;
      return -1;
    }

    //at this point we will be able to schedule
    for (auto& gpu_wks : (*workers)[best_gpu]) {
      if (gpu_wks.second.busy == false) {
        gpu_wks.second.busy = true;
        gpu_wks.second.memory_requested = requested_memory;
        gpu_wks.second.gpu = best_gpu;
        //std::cerr << "scheduling worker at " << gpu_wks.first << std::endl;

        //update gpu stats, ugly, but all schedulers have to have this
        (*gpu_states)[best_gpu].busy_workers += 1;
        (*gpu_states)[best_gpu].estimated_free_memory -= requested_memory;

        return gpu_wks.first;
      }
    }

  }
};

#endif