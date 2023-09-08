#ifndef __FIRSTFIT_SCHEDULER_HPP__
#define __FIRSTFIT_SCHEDULER_HPP__

#include "common.hpp"

/*
struct FirstFit : public BaseScheduler {

  FirstFit(
      std::map<uint32_t, std::map<uint32_t, GPUWorkerState>> *workers,
      std::map<uint32_t, GPUState> *gpu_states) :
    BaseScheduler(workers, gpu_states)
  {}

  int32_t getGPU(uint64_t gpu_memory) {
    for (auto& gpu_wks : *workers) {
        std::cerr << "checking gpu " << gpu_wks.first << std::endl;
        for (auto& port_wk : gpu_wks.second) {
            std::cerr << "checking port " << port_wk.first << std::endl;
            if (port_wk.second.busy == false) {
                port_wk.second.busy = true;

                //update gpu stats, ugly, but all schedulers have to have this
                (*gpu_states)[gpu_wks.first].busy_workers += 1;

                return port_wk.first;
            }
        }
    }
    return -1;
  }
};

*/
#endif