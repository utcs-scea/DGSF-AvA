#include <stdlib.h>
#include <absl/debugging/symbolize.h>
#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <chrono>
#include <thread>
#include "svgpu_manager.hpp"

// arguments to this manager
ABSL_FLAG(uint32_t, manager_port, 3333, "(OPTIONAL) Specify manager port number");
ABSL_FLAG(uint32_t, worker_port_base, 4000, "(OPTIONAL) Specify base port number of API servers");
ABSL_FLAG(std::vector<std::string>, worker_argv, {}, "(OPTIONAL) Specify process arguments passed to API servers");
ABSL_FLAG(std::string, worker_path, "", "(REQUIRED) Specify API server binary path");
ABSL_FLAG(std::vector<std::string>, worker_env, {},
          "(OPTIONAL) Specify environment variables, e.g. HOME=/home/ubuntu, passed to API servers");
ABSL_FLAG(uint16_t, ngpus, 0, "(OPTIONAL) Number of GPUs the manager should use");
ABSL_FLAG(uint16_t, gpuoffset, 0, "(OPTIONAL)GPU id offset");
ABSL_FLAG(std::string, scheduler, "bestfit", "(OPTIONAL) choose GPU scheduler, default is bestfit. Options are: bestfit, worstfit");
ABSL_FLAG(uint32_t, precreated_workers, 1, "(OPTIONAL) How many workers per GPU we pre create. Default is 1.");

ABSL_FLAG(std::string, keepworkeralive, "no", "(OPTIONAL) (debug) forcefully make the worker not die when not in serverless mode");
ABSL_FLAG(std::string, allctx, "no", "(OPTIONAL) turn on setting up all device ctx on workers (required for migration)");
ABSL_FLAG(std::string, reporting, "no", "(OPTIONAL) turn on client reports to gpu server (required for migration)");
ABSL_FLAG(std::string, debug_migration, "no", "(OPTIONAL) turn on debug migration (1 for execution, 2 for memory, 3 for random)");
ABSL_FLAG(std::string, ttc_addr, "0", "(OPTIONAL) address of ttc server for timeline creation)");
ABSL_FLAG(std::string, nvmlmonitor, "yes", "(OPTIONAL) disable NVML monitor");
ABSL_FLAG(uint32_t, migration_strat, 0, "(OPTIONAL) migration strategy. 0 is disabled.");

int main(int argc, const char *argv[]) {
  if (!std::getenv("SELF_IP")) {
    printf("env var SELF_IP does not exist, export it to be the ip of this GPU server and rerun\n");
    exit(1);
  }

  absl::ParseCommandLine(argc, const_cast<char **>(argv));
  absl::InitializeSymbolizer(argv[0]);
  ava_manager::setupSignalHandlers();
  auto worker_argv = absl::GetFlag(FLAGS_worker_argv);
  auto worker_env = absl::GetFlag(FLAGS_worker_env);

  uint32_t port;
  // let's give env priority
  if (const char *env_port = std::getenv("AVAMNGR_PORT")) {
    port = static_cast<uint32_t>(std::stoul(env_port));
  } else {
    port = absl::GetFlag(FLAGS_manager_port);
  }
  std::cerr << "[SVLESS-MNGR]: Using port " << port << std::endl;

  //check for debug flag
  if (absl::GetFlag(FLAGS_debug_migration) != "no") {
    printf(">>> Setting SG_DEBUG_MIGRATION \n");
    std::string kmd = "SG_DEBUG_MIGRATION=";
    kmd += absl::GetFlag(FLAGS_debug_migration);
    worker_env.push_back(kmd);
    setenv("SG_DEBUG_MIGRATION", absl::GetFlag(FLAGS_debug_migration).c_str(), 1);
  }

  // Check all contexts option
  if (absl::GetFlag(FLAGS_allctx) == "yes") {
    std::cerr << "[SVLESS-MNGR]: All context init is enabled." << std::endl;
    worker_env.push_back("AVA_ENABLE_ALL_CTX=yes");
  }
  else {
    std::cerr << "[SVLESS-MNGR]: All context init is DISABLED." << std::endl;
    worker_env.push_back("AVA_ENABLE_ALL_CTX=no");
  }

  // Check reporting option
  if (absl::GetFlag(FLAGS_reporting) == "yes") {
    std::cerr << "[SVLESS-MNGR]: Worker usage reporting is enabled.." << std::endl;
    worker_env.push_back("AVA_ENABLE_REPORTING=yes");
  }
  else {
    std::cerr << "[SVLESS-MNGR]: Reporting is disabled, no migration or reporting will be done" << std::endl;
    worker_env.push_back("AVA_ENABLE_REPORTING=no");
  }

  //check ttc addr argument
  if (absl::GetFlag(FLAGS_ttc_addr) != "0") {
    std::cerr << "[SVLESS-MNGR]: Setting ttc address to " << absl::GetFlag(FLAGS_ttc_addr) << std::endl;
    std::string kmd = "TTC_ADDR=";
    kmd += absl::GetFlag(FLAGS_ttc_addr);
    worker_env.push_back(kmd);
  }

  // if serverless mode or force reuse
  char *rm_addr = std::getenv("RESMNGR_ADDR");
  if (rm_addr || absl::GetFlag(FLAGS_keepworkeralive) == "yes") {
    setenv("REUSE_WORKER", "1", 1);
    std::string kmd = "REUSE_WORKER=1";
    worker_env.push_back(kmd);
  }

  std::string resmngr_addr = "";
  if (rm_addr) {
    resmngr_addr = std::string(rm_addr);
    resmngr_addr += ":";
    resmngr_addr += std::getenv("RESMNGR_PORT");
    std::cerr << "[SVLESS-MNGR]: Running manager on serverless mode, rm at " << resmngr_addr << std::endl;   
  }
  
  /*
   *  Create the manager 
   */
  std::cerr << "[SVLESS-MNGR]: Using port " << port << " for AvA" << std::endl;
  SVGPUManager *manager =
      new SVGPUManager(port, absl::GetFlag(FLAGS_worker_port_base), absl::GetFlag(FLAGS_worker_path), worker_argv,
                       worker_env, absl::GetFlag(FLAGS_ngpus), absl::GetFlag(FLAGS_gpuoffset), 
                       resmngr_addr, absl::GetFlag(FLAGS_scheduler), absl::GetFlag(FLAGS_precreated_workers),
                       absl::GetFlag(FLAGS_nvmlmonitor), absl::GetFlag(FLAGS_migration_strat) );

  //loop forever
  manager->RunServer();
  return 0;
}
