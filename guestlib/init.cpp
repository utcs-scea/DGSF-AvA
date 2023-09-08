#include <plog/Log.h>
#include <stdlib.h>
#include <unistd.h>

#include <cstdio>
#include <iostream>
#include <vector>

#include "common/cmd_channel.hpp"
#include "common/cmd_handler.hpp"
#include "common/endpoint_lib.hpp"
#include "common/linkage.h"
#include "common/support/env_variables.h"
#include "guest_config.h"
#include "guestlib.h"
#include "guestlib/guest_thread.h"
#include <plog/Initializers/RollingFileInitializer.h>
#include <plog/Appenders/ConsoleAppender.h>
#include <plog/Formatters/TxtFormatter.h>
struct command_channel *chan;

struct param_block_info nw_global_pb_info = {0, 0};
extern int nw_global_vm_id;

static struct command_channel *channel_create() { return chan; }

EXPORTED_WEAKLY void nw_init_log() {
  std::ios_base::Init();
  guestconfig::config = guestconfig::readGuestConfig();
  if (guestconfig::config == nullptr) exit(EXIT_FAILURE);
#ifdef DEBUG
  guestconfig::config->print();
#endif
  auto logger_severity = guestconfig::config->logger_severity_;
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
  // Initialize logger
  int use_console = ava::support::GetEnvVariableAsInt("AVA_LOG_CONSOLE", 0);
  if (use_console == 1) {
    static plog::ConsoleAppender<plog::TxtFormatter> consoleAppender;
    plog::init(logger_severity, &consoleAppender);
    std::cerr << "AvA log to console" << std::endl;
  } else {
    std::string log_file = std::tmpnam(nullptr);
    plog::init(logger_severity, log_file.c_str());
    std::cerr << "To check the state of AvA remoting progress, use `tail -f " << log_file << "`" << std::endl;
  }
}

EXPORTED_WEAKLY void nw_init_guestlib(intptr_t api_id) {
  ava::register_guestlib_main_thread(ava::GuestThread::kGuestStatsPath, ava::GuestThread::kGuestStatsPrefix);

#ifdef AVA_PRINT_TIMESTAMP
  struct timeval ts;
  gettimeofday(&ts, NULL);
#endif

  std::cerr << ">>>>>>>>>> nw_init_guestlib\n ";

  /* Create connection to worker and start command handler thread */
  if (guestconfig::config->channel_ == "TCP") {
    std::vector<struct command_channel *> channels = command_channel_socket_tcp_guest_new();
    chan = channels[0];
    // } else if (guestconfig::config->channel_ == "SHM") {
    //   chan = command_channel_shm_guest_new();
    // } else if (guestconfig::config->channel_ == "VSOCK") {
    //   chan = command_channel_socket_new();
  } else {
    std::cerr << "Unsupported channel specified in " << guestconfig::getConfigFilePath()
              << ", expect channel = [\"TCP\"]" << std::endl;
    exit(0);
  }
  if (!chan) {
    std::cerr << "Failed to create command channel" << std::endl;
    exit(1);
  }
  init_command_handler(channel_create);
  init_internal_command_handler();

  /* Send initialize API command to the worker */
  struct command_handler_initialize_api_command *api_init_command =
      (struct command_handler_initialize_api_command *)command_channel_new_command(
          nw_global_command_channel, sizeof(struct command_handler_initialize_api_command), 0);
  api_init_command->base.api_id = COMMAND_HANDLER_API;
  api_init_command->base.command_id = COMMAND_HANDLER_INITIALIZE_API;
  api_init_command->base.vm_id = nw_global_vm_id;
  api_init_command->new_api_id = api_id;
  api_init_command->pb_info = nw_global_pb_info;
  command_channel_send_command(chan, (struct command_base *)api_init_command);
 
  /* Send dump directory and requested GPU memory to worker*/
  char *dump_dir = std::getenv("AVA_WORKER_DUMP_DIR");

  uint32_t gpu_mem = 0;
  if (std::getenv("AVA_REQUESTED_GPU_MEMORY"))
      gpu_mem = std::stoul(std::getenv("AVA_REQUESTED_GPU_MEMORY"));

  struct command_base *dump_dir_command = command_channel_new_command(
        nw_global_command_channel, sizeof(struct command_base), 0);
  dump_dir_command->api_id = COMMAND_HANDLER_API;
  dump_dir_command->command_id = COMMAND_HANDLER_REGISTER_DUMP_DIR;
  dump_dir_command->vm_id = nw_global_vm_id;

  if (dump_dir != NULL) {
    if (strlen(dump_dir) > RESERVED_AREA_SIZE - sizeof(uint32_t) - 1) {
      std::cerr << "Dump directory name length is greater than reserved_area size" << std::endl;
      exit(1);
    }
    sprintf(&dump_dir_command->reserved_area[sizeof(uint32_t)/sizeof(char)], "%s", dump_dir);
  }

  memcpy(dump_dir_command->reserved_area, &gpu_mem, sizeof(uint32_t));
  command_channel_send_command(chan, (struct command_base *)dump_dir_command);

  /* Send VMID*/
  char default_vmid[] = "NO_VMID";
  char* vmid = default_vmid;
  //this will always be true when running serverless
  if (std::getenv("VMID")) {
      vmid = std::getenv("VMID");
  }

  struct command_base *set_vmid_command = command_channel_new_command(
        nw_global_command_channel, sizeof(struct command_base), 0);
  set_vmid_command->api_id = COMMAND_HANDLER_API;
  set_vmid_command->command_id = COMMAND_HANDLER_REGISTER_VMID;
  set_vmid_command->vm_id = nw_global_vm_id;
  strcpy(set_vmid_command->reserved_area, vmid);
  command_channel_send_command(chan, (struct command_base *)set_vmid_command);


#ifdef AVA_PRINT_TIMESTAMP
  struct timeval ts_end;
  gettimeofday(&ts_end, NULL);
  printf("loading_time: %f\n", ((ts_end.tv_sec - ts.tv_sec) * 1000.0 + (float)(ts_end.tv_usec - ts.tv_usec) / 1000.0));
#endif
}

EXPORTED_WEAKLY void nw_destroy_guestlib(void) {
  /* Send shutdown command to the worker */
  /*
  struct command_base* api_shutdown_command =
  command_channel_new_command(nw_global_command_channel, sizeof(struct
        command_base), 0); api_shutdown_command->api_id = COMMAND_HANDLER_API;
  api_shutdown_command->command_id = COMMAND_HANDLER_SHUTDOWN_API;
  api_shutdown_command->vm_id = nw_global_vm_id;
  command_channel_send_command(chan, api_shutdown_command);
  api_shutdown_command = command_channel_receive_command(chan);
  */

  // TODO: This is called by the guestlib so destructor for each API. This is
  // safe, but will make the handler shutdown when the FIRST API unloads when
  // having it shutdown with the last would be better.
  //printf("destroy_command_handler IN");
  destroy_command_handler(true);
  //printf("destroy_command_handler OUT");
  exit(0);
}
