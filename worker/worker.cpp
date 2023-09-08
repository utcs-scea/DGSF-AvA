#include "worker.h"

#include <absl/debugging/failure_signal_handler.h>
#include <absl/debugging/symbolize.h>
#include <errno.h>
#include <execinfo.h>
#include <fcntl.h>
#include <fmt/core.h>
#include <nvml.h>
#include <poll.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#include <zmq.h>

#include <cstdio>
#include <gsl/gsl>
#include <iostream>
#include <string>

#include "common/cmd_channel_impl.hpp"
#include "common/cmd_handler.hpp"
#include "common/common_context.h"
#include "common/linkage.h"
#include "common/socket.hpp"
#include "common/support/env_variables.h"
#include "extensions/memory_server/client.hpp"
#include "plog/Initializers/RollingFileInitializer.h"
#include "worker_context.h"

#include "common/extensions/cudnn_optimization.h"
#include "common/extensions/tcp_timeline_client.hpp"

struct command_channel *chan;
struct command_channel *chan_hv = NULL;
extern int nw_global_vm_id;

__sighandler_t original_sigint_handler = SIG_DFL;
__sighandler_t original_sigsegv_handler = SIG_DFL;
__sighandler_t original_sigchld_handler = SIG_DFL;

void sigint_handler(int signo) {
  void *array[10];
  size_t size;
  size = backtrace(array, 10);
  fprintf(stderr, "===== backtrace =====\n");
  fprintf(stderr, "receive signal %d:\n", signo);
  backtrace_symbols_fd(array, size, STDERR_FILENO);

  if (chan) {
    command_channel_free(chan);
    chan = NULL;
  }
  signal(signo, original_sigint_handler);
  raise(signo);
}

void sigsegv_handler(int signo) {
  void *array[10];
  size_t size;
  size = backtrace(array, 10);
  fprintf(stderr, "===== backtrace =====\n");
  fprintf(stderr, "receive signal %d:\n", signo);
  backtrace_symbols_fd(array, size, STDERR_FILENO);

  if (chan) {
    command_channel_free(chan);
    chan = NULL;
  }
  signal(signo, original_sigsegv_handler);
  raise(signo);
}

void nw_report_storage_resource_allocation(const char *const name, ssize_t amount) {
  if (chan_hv) command_channel_hv_report_storage_resource_allocation(chan_hv, name, amount);
}

void nw_report_throughput_resource_consumption(const char *const name, ssize_t amount) {
  if (chan_hv) command_channel_hv_report_throughput_resource_consumption(chan_hv, name, amount);
}

static struct command_channel *channel_create() { return chan; }

EXPORTED_WEAKLY std::string worker_init_log() {
  std::ios_base::Init();
  // Initialize logger
  std::string log_file = std::tmpnam(nullptr);
  auto logger_severity = plog::debug;
  auto env_log = ava::support::GetEnvVariable("AVA_LOG_LEVEL");
  if (env_log != "") {
    if ((env_log == "verbose") || (env_log == "VERBOSE")) {
      logger_severity = plog::verbose;
    } else if ((env_log == "debug") || (env_log == "DEBUG")) {
      logger_severity = plog::debug;
    } else if ((env_log == "info") || (env_log == "INFO")) {
      logger_severity = plog::info;
    } else if ((env_log == "warning") || (env_log == "WARNING")) {
      logger_severity = plog::warning;
    } else if ((env_log == "error") || (env_log == "ERROR")) {
      logger_severity = plog::error;
    } else if ((env_log == "fatal") || (env_log == "FATAL")) {
      logger_severity = plog::fatal;
    }
  }
  plog::init(logger_severity, log_file.c_str());
  //std::cerr << "To check the state of AvA remoting progress, use `tail -f " << log_file << "`" << std::endl;
  return log_file;
}

static void nvml_setDeviceCount() {
  nvmlReturn_t result;
  uint32_t device_count;
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

  printf("Setting internal device count to %d\n", device_count);
  __internal_setDeviceCount(device_count);
}

static void create_cuda_contexts() {
  for (int i = 0; i < __internal_getDeviceCount(); i++) {
    cudaSetDevice(i);
    // this forcibly creates a primary context, which is lazily-created
    cudaFree(0);
    std::cerr << "Created CUDA context on device [" << i << "]" << std::endl;
  }
}

int main(int argc, char *argv[]) {
  if (!(argc == 3 && !strcmp(argv[1], "migrate")) && (argc != 2)) {
    printf(
        "Usage: %s <listen_port>\n"
        "or     %s <mode> <listen_port> \n",
        argv[0], argv[0]);
    return 0;
  }
  absl::InitializeSymbolizer(argv[0]);

  //std::cerr << "Worker spinning up.." << std::endl;
  //ttc.notify(0);

  char const *gpu_device_str = getenv("GPU_DEVICE");
  std::string gpu_device = std::string(gpu_device_str);
  /* set current device*/
  GPUMemoryServer::Client::getInstance().setCurrentGPU(std::stoi(gpu_device));
  //read device count from nvml and set it internally
  nvml_setDeviceCount();

  //ttc.notify(1);

  // AVA_WORKER_UUID is a unique, starting at 0, id we can use
  char const *cworker_uuid = getenv("AVA_WORKER_UUID");
  std::string worker_uuid = std::string(cworker_uuid);

  // check if we are enabling all device contexts
  std::string enable_all_ctx = std::string(getenv("AVA_ENABLE_ALL_CTX"));
  if (enable_all_ctx == "yes") {
    // preemptively create context on all GPUs
    create_cuda_contexts();
    __internal_setAllContextsEnabled(true);
  } else {
    cudaSetDevice(std::stoi(gpu_device));
    // this forcibly creates a primary context, which is lazily-created
    cudaFree(0);
    __internal_setAllContextsEnabled(false);
  }
  //ttc.notify(2);

  //now that we have possibly all contexts created, init stuff
  static auto worker_context = ava::WorkerContext::instance();

#ifdef AVA_PRELOAD_CUBIN
  worker_cudnn_opt_init(1);
#endif

  //ttc.notify(3);
  std::string enable_reporting = std::string(getenv("AVA_ENABLE_REPORTING"));
  if (enable_reporting == "yes") {
    GPUMemoryServer::Client::getInstance().enable_reporting = true;
  } else {
    GPUMemoryServer::Client::getInstance().enable_reporting = false;
  }

  /* setup signal handler */
  if ((original_sigint_handler = signal(SIGINT, sigint_handler)) == SIG_ERR) printf("failed to catch SIGINT\n");
  if ((original_sigsegv_handler = signal(SIGSEGV, sigsegv_handler)) == SIG_ERR) printf("failed to catch SIGSEGV\n");

  absl::FailureSignalHandlerOptions options;
  options.call_previous_handler = true;
  absl::InstallFailureSignalHandler(options);

  /* define arguments */
  auto wctx = ava::WorkerContext::instance();
  nw_worker_id = 0;
  /* parse arguments */
  uint32_t listen_port = (uint32_t)atoi(argv[1]);
  GPUMemoryServer::Client::getInstance().setListenPort(listen_port);
  wctx->set_api_server_listen_port(listen_port);
  //std::cerr << "[worker#" << listen_port << "] To check the state of AvA remoting progress, use `tail -f "
  //          << wctx->log_file << "`" << std::endl;
  //ttc.notify(4);

  if (!getenv("AVA_CHANNEL") || !strcmp(getenv("AVA_CHANNEL"), "TCP")) {
    chan_hv = NULL;
    chan = command_channel_socket_tcp_worker_new(listen_port);
    nw_record_command_channel = command_channel_log_new(listen_port);
    //ttc.notify(5);

    // this sets API id and other stuff
    init_internal_command_handler();
    //ttc.notify(6);

    // only loop if we are in serverless mode
    do {
      //tell manager we are ready
      GPUMemoryServer::Client::getInstance().notifyReady();

      // go back to original GPU
      GPUMemoryServer::Client::getInstance().resetCurrentGPU();

      // get a guestlib connection
      //std::cerr << "[worker#" << listen_port << "] waiting for connection" << std::endl;
      //ttc.notify(7);
      chan = command_channel_listen(chan);
      //ttc.notify(8);
      // this launches the thread that listens for commands
      init_command_handler(channel_create);
      //ttc.notify(9);
      /*
       *  sync with worker until we get vmid and memory requested
       */
      wait_for_worker_setup();
      //printf("CV: worker was notified vmid was received..\n");
      // if this is serverless, we need to update our id
      if (svless_vmid == "NO_VMID" || svless_vmid == "") {
        //printf("svless_vmid is default, using %s\n", worker_uuid.c_str());
        GPUMemoryServer::Client::getInstance().setUuid(worker_uuid);
      } else {
        //printf("got vmid from cmd channel: %s\n", svless_vmid.c_str());
        GPUMemoryServer::Client::getInstance().setUuid(svless_vmid);
      }

      // now wait for cubin flag to be set
      wait_for_cubin_loaded();
      
      //printf("CV: worker was notified cubin was loaded..\n");
      // report our max memory requested
      GPUMemoryServer::Client::getInstance().reportMemoryRequested(requested_gpu_mem);

      //std::cerr << "[worker#" << listen_port << "] is free to work now" << std::endl;
      // and now all threads can work
            
      release_shadow_threads();
      //ttc.notify(10);

      wait_for_command_handler();
      //ttc.notify(11);
      destroy_command_handler(false);
      std::cerr << "[worker#" << listen_port << "] worker is done, looping." << std::endl;

      // clean up allocations, local and remote
      GPUMemoryServer::Client::getInstance().fullCleanup();
      //ttc.notify(12);
    } while (std::getenv("REUSE_WORKER"));

    std::cerr << "[worker#" << listen_port << "] freeing channel and quiting." << std::endl;
    command_channel_free(chan);
    command_channel_free((struct command_channel *)nw_record_command_channel);
    return 0;
  }

  printf("Unsupported AVA_CHANNEL type (export AVA_CHANNEL=[TCP]\n");
  return 1;
}
