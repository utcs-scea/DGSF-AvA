#include "common/cmd_handler.hpp"

#include <plog/Log.h>

#include "common/common_context.h"
#include "common/endpoint_lib.hpp"
#include "common/linkage.h"
#include "common/shadow_thread_pool.hpp"
#include <cuda_runtime_api.h>
#ifdef AVA_WORKER
#include "extensions/memory_server/client.hpp"
#include "worker/worker_context.h"
#include "worker/worker.h"
uint32_t requested_gpu_mem;
std::string svless_vmid;
#endif

#ifdef __cplusplus
#include <atomic>
using namespace std;
#else
#include <stdatomic.h>
#endif

#include <assert.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>

#include <chrono>
#include <thread>
#include <condition_variable>
#include <mutex>
#include <iostream>

/*
 *   two condition variables to make thread and worker synchronize
 */
std::atomic<bool> received_vmid(false);
std::mutex received_vmid_mutex;
std::condition_variable received_vmid_cv;

//stuff to block until load
std::mutex cubin_loaded_mutex;
std::atomic<bool> cubin_loaded(false);
std::condition_variable cubin_loaded_cv;

//stuff to block until load
std::mutex threads_ready_mutex;
std::atomic<bool> threads_ready(false);
std::condition_variable threads_ready_cv;

//helpers to synchronize threads
void wait_for_worker_setup() {
  //wait for worker to set up and let us go
  std::unique_lock<std::mutex> lk(received_vmid_mutex);
  while (!received_vmid)
      received_vmid_cv.wait(lk);
}

void worker_setup_done() {
  std::unique_lock<std::mutex> lk(received_vmid_mutex);
  received_vmid = true;
  //printf("CV: cmd_handler notifying vmid was received..\n");
  received_vmid_cv.notify_all();
}

void wait_for_cubin_loaded() {
  //wait for worker to set up and let us go
  std::unique_lock<std::mutex> lk(cubin_loaded_mutex);
  while (!cubin_loaded)
      cubin_loaded_cv.wait(lk);
}

void set_cubin_loaded() {
  std::unique_lock<std::mutex> lk(cubin_loaded_mutex);
  cubin_loaded = true;
  cubin_loaded_cv.notify_all();
}

void release_shadow_threads() {
  std::unique_lock<std::mutex> lk(threads_ready_mutex);
  threads_ready = true;
  threads_ready_cv.notify_all();
}

void wait_for_shadow_threads_release() {
  std::unique_lock<std::mutex> lk(threads_ready_mutex);
  while (!threads_ready)
      threads_ready_cv.wait(lk);
}

// Internal flag set by the first call to init_command_handler
EXPORTED_WEAKLY volatile int init_command_handler_executed;

EXPORTED_WEAKLY struct command_channel *nw_global_command_channel;
EXPORTED_WEAKLY pthread_mutex_t nw_handler_lock = PTHREAD_MUTEX_INITIALIZER;

struct command_handler_t {
  void (*replay)(struct command_channel *__chan, struct nw_handle_pool *handle_pool, struct command_channel *__log,
                 const struct command_base *__call_cmd, const struct command_base *__ret_cmd);
  void (*handle)(struct command_channel *__chan, struct nw_handle_pool *handle_pool, struct command_channel *__log,
                 const struct command_base *__cmd);
  void (*print)(FILE *file, const struct command_channel *__chan, const struct command_base *__cmd);
};

EXPORTED_WEAKLY struct command_handler_t nw_apis[MAX_API_ID];
EXPORTED_WEAKLY pthread_t nw_handler_thread;

static int handle_command(struct command_channel *chan, struct nw_handle_pool *handle_pool, struct command_channel *log,
                          struct command_base *cmd);

EXPORTED_WEAKLY void register_command_handler(int api_id,
                                              void (*handle)(struct command_channel *, struct nw_handle_pool *,
                                                             struct command_channel *, const struct command_base *),
                                              void (*print)(FILE *, const struct command_channel *,
                                                            const struct command_base *),
                                              void (*replay)(struct command_channel *, struct nw_handle_pool *,
                                                             struct command_channel *, const struct command_base *,
                                                             const struct command_base *)) {
  assert(api_id < MAX_API_ID);
  LOG_INFO << "Registering API command handler for API id " << api_id << ": handler at 0x" << std::hex
           << (uintptr_t)handle;
  struct command_handler_t *api = &nw_apis[api_id];
  //assert(api->handle == NULL && "Only one handler can be registered for each API id");
  if (api->handle != NULL) {
    LOG_INFO << "Overwriting API id " << api_id << std::endl;
  }
  api->handle = handle;
  api->print = print;
  api->replay = replay;
}

EXPORTED_WEAKLY void print_command(FILE *file, const struct command_channel *chan, const struct command_base *cmd) {
  const intptr_t api_id = cmd->api_id;
  // Lock the file to prevent commands from getting mixed in the print out
  flockfile(file);
  if (nw_apis[api_id].print) nw_apis[api_id].print(file, chan, cmd);
  funlockfile(file);
}

static int handle_command(struct command_channel *chan, struct nw_handle_pool *handle_pool, struct command_channel *log,
                          struct command_base *cmd) {
  const intptr_t api_id = cmd->api_id;
  assert(nw_apis[api_id].handle != NULL);
  nw_apis[api_id].handle(chan, handle_pool, log, cmd);
  command_channel_free_command(chan, cmd);
  return 0;
}

static void _handle_commands_loop(struct command_channel *chan) {
  while (1) {
    struct command_base *cmd = command_channel_receive_command(chan);

    if (cmd == NULL) {
      LOG_INFO << "received NULL cmd, ending thread" << std::endl;
      return;
    }

#ifdef AVA_PRINT_TIMESTAMP
    if (cmd->api_id != 0) {
      struct timeval ts;
      gettimeofday(&ts, NULL);
      printf("Handler: command_%ld receive_command at : %ld s, %ld us\n", cmd->command_id, ts.tv_sec, ts.tv_usec);
    }
#endif

    // TODO: checks MSG_SHUTDOWN messages/channel close from the other side.
    auto context = ava::CommonContext::instance();
    shadow_thread_pool_dispatch(context->nw_shadow_thread_pool, chan, cmd);
  }
}

void handle_command_and_notify(struct command_channel *chan, struct command_base *cmd) {
  thread_local int32_t tl_current_device = -1;
  auto context = ava::CommonContext::instance();

#ifdef AVA_WORKER
  __cmd_handle_in();

  //printf("before lock.  %d   %d  %d\n", int(cubin_loaded), cmd->command_id == COMMAND_HANDLER_REGISTER_VMID, cmd->command_id == COMMAND_HANDLER_REGISTER_DUMP_DIR);
  //if cubin is not loaded we gotta wait, but let the load cmds go through
  if (!threads_ready && cmd->command_id != COMMAND_HANDLER_REGISTER_VMID && cmd->command_id != COMMAND_HANDLER_REGISTER_DUMP_DIR) {
    //wait for worker to set up and let us go
    wait_for_shadow_threads_release();
    //printf("  #### shadow thread unlocked for handling!\n");
  }

  if (tl_current_device == -1) {
    cudaSetDevice(context->current_device);
    tl_current_device = context->current_device;
    //printf(">>> shadow thread setting default device  [%d] \n", context->current_device);
  }
  
  if (tl_current_device != context->current_device) {
    std::cerr << ">>> shadow thread detected change of device, changing from" << tl_current_device << " to " << context->current_device << std::endl;
    cudaSetDevice(context->current_device);
    tl_current_device = context->current_device;
  }

#endif

  handle_command(chan, context->nw_global_handle_pool, (struct command_channel *)nw_record_command_channel, cmd);

#ifdef AVA_WORKER
  __cmd_handle_out();
#endif

}

static void *dispatch_thread_impl(void *userdata) {
  struct command_channel *chan = (struct command_channel *)userdata;

  // set cancellation state
  if (pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL)) {
    perror("pthread_setcancelstate failed\n");
    exit(0);
  }

  // PTHREAD_CANCEL_DEFERRED means that it will wait the pthread_join
  if (pthread_setcanceltype(PTHREAD_CANCEL_DEFERRED, NULL)) {
    perror("pthread_setcanceltype failed\n");
    exit(0);
  }

  _handle_commands_loop(chan);
  return NULL;
}
// TODO: This will not correctly handle running callbacks in the initially
// calling thread.

EXPORTED_WEAKLY void init_command_handler(struct command_channel *(*channel_create)()) {
  //reset the loading part
  cubin_loaded = false;
  received_vmid = false;
  threads_ready = false;
  //kill_all_shadow_threads();

  pthread_mutex_lock(&nw_handler_lock);
  if (!init_command_handler_executed) {
    nw_global_command_channel = channel_create();
    pthread_create(&nw_handler_thread, NULL, dispatch_thread_impl, (void *)nw_global_command_channel);
    atomic_thread_fence(memory_order_release);
    init_command_handler_executed = 1;
  }
  pthread_mutex_unlock(&nw_handler_lock);
}

EXPORTED_WEAKLY void destroy_command_handler(bool destroy_channel) {
  pthread_mutex_lock(&nw_handler_lock);
  if (init_command_handler_executed) {
    pthread_cancel(nw_handler_thread);
    pthread_join(nw_handler_thread, NULL);
    if (destroy_channel) {
      command_channel_free(nw_global_command_channel);
      //kill_all_shadow_threads();
    }
    atomic_thread_fence(memory_order_release);
    init_command_handler_executed = 0;
  }
  pthread_mutex_unlock(&nw_handler_lock);

}

EXPORTED_WEAKLY void wait_for_command_handler() { pthread_join(nw_handler_thread, NULL); }

//! Feel free to move these functions around

EXPORTED_WEAKLY struct command_channel_log *nw_record_command_channel;

static void replay_command(struct command_channel *chan, struct nw_handle_pool *handle_pool,
                           struct command_channel *log, const struct command_base *call_cmd,
                           const struct command_base *ret_cmd) {
  const intptr_t api_id = call_cmd->api_id;
  assert(call_cmd->api_id == ret_cmd->api_id);
  assert(nw_apis[api_id].replay != NULL);
  nw_apis[api_id].replay(chan, handle_pool, log, call_cmd, ret_cmd);
}

// TODO: Should be removed. See COMMAND_END_MIGRATION below.
// Flag used to communicate with the guestlib. It is extern declared again at
// guestlib/src/init.c:102
EXPORTED_WEAKLY int nw_end_migration_flag = 0;

void internal_api_handler(struct command_channel *chan, struct nw_handle_pool *handle_pool,
                          struct command_channel *AVA_UNUSED(log), const struct command_base *cmd) {
  assert(cmd->api_id == COMMAND_HANDLER_API);

  struct command_channel *transfer_chan;

  // TODO(migration,refactor): define handlers in separate files, and register
  // them
  // when guestlib and API server start.
  switch (cmd->command_id) {
  case COMMAND_HANDLER_SHUTDOWN_API: {
    exit(0);
    break;
  }

#ifdef AVA_WORKER
  /**
   * For testing, guestlib initiates the migration and worker
   * replays the logs that recorded by itself.
   */
  case COMMAND_START_MIGRATION: {
    //! Complete steps
    // Create a log channel for sending;
    // (Spawn a new worker which) Creates a log channel for receiving;
    // Transfer logs from nw_record_command_channel to sending channel by
    // ava_extract_objects(sending_chan, nw_record_command_channel,
    //         nw_handle_pool_get_live_handles(nw_global_handle_pool));
    // New worker executes received logs.

    auto ccontext = ava::CommonContext::instance();
    //! Simplified steps
    // Create a log channel for sending and receiving
    transfer_chan = (struct command_channel *)command_channel_log_new(nw_worker_id + 1000);

    // Transfer logs from nw_record_command_channel to new log channel
    ava_extract_objects(transfer_chan, nw_record_command_channel,
                        nw_handle_pool_get_live_handles(ccontext->nw_global_handle_pool));
    {
      struct command_base *log_end = command_channel_new_command(transfer_chan, sizeof(struct command_base), 0);
      log_end->api_id = COMMAND_HANDLER_API;
      log_end->command_id = COMMAND_END_MIGRATION;
      command_channel_send_command(transfer_chan, log_end);
    }

    command_channel_free((struct command_channel *)nw_record_command_channel);

    struct nw_handle_pool *replay_handle_pool = nw_handle_pool_new();
    struct command_channel_log *replay_log = command_channel_log_new(nw_worker_id);
    nw_record_command_channel = replay_log;

    printf("\n//! starts to read recorded commands\n\n");
    while (1) {
      // Read logged commands by command_channel_recieve_command
      struct command_base *call_cmd = command_channel_receive_command(transfer_chan);
      if (call_cmd->api_id == COMMAND_HANDLER_API) {
        if (call_cmd->command_id == COMMAND_END_MIGRATION) {
          break;
        }

        /* replace explicit state */
        handle_command(transfer_chan, replay_handle_pool, (struct command_channel *)nw_record_command_channel,
                       call_cmd);
      } else {
        command_channel_print_command(transfer_chan, call_cmd);
        struct command_base *ret_cmd = command_channel_receive_command(transfer_chan);
        command_channel_print_command(transfer_chan, ret_cmd);

        // Replay the commands.
        replay_command(transfer_chan, replay_handle_pool, (struct command_channel *)replay_log, call_cmd, ret_cmd);

        command_channel_free_command(transfer_chan, ret_cmd);
      }
      command_channel_free_command(transfer_chan, call_cmd);
    }
    printf("\n//! finishes read of recorded commands\n\n");

    // TODO: For swapping we will need to selectively copy values back into
    // the nw_global_handle_pool
    //  and then destroy the reply_handle_pool.
    nw_handle_pool_free(ccontext->nw_global_handle_pool);
    ccontext->nw_global_handle_pool = replay_handle_pool;

    {
      struct command_base *log_end = command_channel_new_command(chan, sizeof(struct command_base), 0);
      log_end->api_id = COMMAND_HANDLER_API;
      log_end->thread_id = cmd->thread_id;
      log_end->command_id = COMMAND_END_MIGRATION;
      /* notify guestlib of completion */
      command_channel_send_command(chan, log_end);
    }
    command_channel_free(transfer_chan);
    break;
  }
#endif

  case COMMAND_END_MIGRATION: {
    // TODO: Move this command into a handler guestlib/src/init.c
    nw_end_migration_flag = 1;
    break;
  }

#ifdef AVA_WORKER
  case COMMAND_HANDLER_REPLACE_EXPLICIT_STATE: {
    ava_handle_replace_explicit_state(chan, handle_pool, (struct ava_replay_command_t *)cmd);
    break;
  }

    // TODO(migration): Move to a separate file.
  case COMMAND_START_LIVE_MIGRATION: {
    auto cctx = ava::CommonContext::instance();
    auto wctx = ava::WorkerContext::instance();
    transfer_chan =
        (struct command_channel *)command_channel_socket_tcp_migration_new(wctx->get_api_server_listen_port(), 1);
    struct timeval start, end;

    FILE *fd;
    fd = fopen("migration.log", "a");
    gettimeofday(&start, NULL);
    // Initiate the live migration
    {
      struct command_base *log_init = command_channel_new_command(transfer_chan, sizeof(struct command_base), 0);
      log_init->api_id = COMMAND_HANDLER_API;
      log_init->command_id = COMMAND_ACCEPT_LIVE_MIGRATION;
      // TODO: send more worker information
      command_channel_send_command(transfer_chan, log_init);
      LOG_DEBUG << "sent init migration message to target";
    }

    // Extract recorded commands and exlicit objects
    ava_extract_objects_in_pair(transfer_chan, nw_record_command_channel,
                                nw_handle_pool_get_live_handles(cctx->nw_global_handle_pool));
    LOG_DEBUG << "sent recorded commands to target";

    {
      struct command_base *log_end = command_channel_new_command(transfer_chan, sizeof(struct command_base), 0);
      log_end->api_id = COMMAND_HANDLER_API;
      log_end->command_id = COMMAND_END_LIVE_MIGRATION;
      command_channel_send_command(transfer_chan, log_end);
      LOG_DEBUG << "sent end migration message to target";
    }

    /* notify guestlib of completion */
    {
      struct command_base *log_end = command_channel_new_command(chan, sizeof(struct command_base), 0);
      log_end->api_id = COMMAND_HANDLER_API;
      log_end->thread_id = cmd->thread_id;
      log_end->command_id = COMMAND_END_MIGRATION;
      command_channel_send_command(chan, log_end);
      // TODO: guestlib reconnect to new worker
    }
    gettimeofday(&end, NULL);
    printf("migration takes %lf\n", ((end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) / 1000.0));
    fprintf(fd, "[%d] migration takes %lf\n", nw_worker_id,
            ((end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) / 1000.0));
    fclose(fd);

    break;
  }
#endif

  case COMMAND_END_LIVE_MIGRATION: {
    printf("\n//! finishes live migration\n\n");
    // TODO: target worker connects to guestlib
    usleep(1000000);  // enough time for source worker to print out migration
                      // time
    exit(0);
    break;
  }

#ifdef AVA_WORKER
  case COMMAND_ACCEPT_LIVE_MIGRATION: {
    printf("\n//! starts to accept incoming commands\n\n");
    break;
  }

  case COMMAND_HANDLER_RECORDED_PAIR: {
    auto common_context = ava::CommonContext::instance();
    struct ava_replay_command_pair_t *combine = (struct ava_replay_command_pair_t *)cmd;
    struct command_base *call_cmd =
        (struct command_base *)command_channel_get_buffer(chan, (struct command_base *)combine, combine->call_cmd);
    struct command_base *ret_cmd =
        (struct command_base *)command_channel_get_buffer(chan, (struct command_base *)combine, combine->ret_cmd);
    command_channel_print_command(chan, call_cmd);
    command_channel_print_command(chan, ret_cmd);

    printf("replay command <%ld, %lx>\n", call_cmd->command_id, call_cmd->region_size);

    // Replay the commands.
    replay_command(chan, common_context->nw_global_handle_pool, (struct command_channel *)nw_record_command_channel,
                   call_cmd, ret_cmd);
    break;
  }

  case COMMAND_HANDLER_REGISTER_VMID: {
    svless_vmid = std::string(cmd->reserved_area);
    //notify worker we set up the vmid
    worker_setup_done();
    break;
  }

  case COMMAND_HANDLER_REGISTER_DUMP_DIR: {
    size_t offset = sizeof(uint32_t) / sizeof(char);
    const char* dump_dir = (const char*)&cmd->reserved_area[offset];
    memcpy(&requested_gpu_mem, cmd->reserved_area, sizeof(uint32_t));
    //printf("\n//! Requested GPU memory: %u\n\n", requested_gpu_mem);

    //if it wasnt specified, use default
    if (strcmp(dump_dir, "") == 0) {
      //printf("dump_dir not specified in COMMAND_HANDLER_REGISTER_DUMP_DIR, using default /cuda_dumps\n");
      dump_dir = "/cuda_dumps";
    }

//if we are in opt spec, we need to load cubin (dump will dump instead)
#ifdef AVA_PRELOAD_CUBIN
    ava_load_cubin_worker(dump_dir);
#endif

    //printf("  !!! cubin loading done, signaling\n");
    //loading is done
    set_cubin_loaded();

    break;
  }

#endif

  default:
    LOG_ERROR << "Unknown internal command: " << cmd->command_id;
    exit(0);
  }
}

EXPORTED_WEAKLY void init_internal_command_handler() {
  // TODO: currently guestlib initiates the migration. Should let hypervisor
  // or manager initiate it.
  register_command_handler(COMMAND_HANDLER_API, internal_api_handler, NULL, NULL);
}
