#ifndef AVA_COMMON_CMD_HANDLER_HPP_
#define AVA_COMMON_CMD_HANDLER_HPP_

#ifndef __KERNEL__

#include <pthread.h>
#include <stdio.h>

#include "common/cmd_channel.hpp"
#include "common/guest_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

struct nw_handle_pool;

/**
 * Register a function to handle commands with the specified API id.
 *
 * This call will abort the program if init_command_handler has
 * already been called.
 */
void register_command_handler(int api_id,
                              void (*handle)(struct command_channel *__chan, struct nw_handle_pool *handle_pool,
                                             struct command_channel *__log, const struct command_base *__cmd),
                              void (*print)(FILE *file, const struct command_channel *__chan,
                                            const struct command_base *__cmd),
                              void (*replay)(struct command_channel *__chan, struct nw_handle_pool *handle_pool,
                                             struct command_channel *__log, const struct command_base *__call_cmd,
                                             const struct command_base *__ret_cmd));

/**
 * Print the given command (which much be from chan) to the stream file.
 */
void print_command(FILE *file, const struct command_channel *chan, const struct command_base *cmd);

#ifndef NDEBUG
#define DEBUG_PRINT_COMMAND(chan, cmd) print_command(stderr, chan, cmd)
#else
#define DEBUG_PRINT_COMMAND(chan, cmd)
#endif

void wait_for_worker_setup();
void worker_setup_done();
void wait_for_cubin_loaded();
void set_cubin_loaded();
void release_shadow_threads();
void wait_for_shadow_threads_release();

void handle_command_and_notify(struct command_channel *chan, struct command_base *cmd);

/**
 * Initialize and start the command handler thread.
 *
 * This call is always very slow.
 */
void init_command_handler(struct command_channel *(*channel_create)());

/**
 * Terminate the handler and close the channel and release other
 * resources.
 */
void destroy_command_handler(bool destroy_channel = true);

/**
 * Block until the command handler thread exits. This may never
 * happen.
 */
void wait_for_command_handler();

/**
 * Initialize internal API handler.
 */
void init_internal_command_handler();

/**
 * The global channel used by this process (either the guestlib or the
 * worker).
 */
extern struct command_channel *nw_global_command_channel;

/**
 * The record channel used by the original worker for migration.
 */
extern struct command_channel_log *nw_record_command_channel;
// TODO: Rename nw_record_command_channel to nw_global_command_log?

/**
 * The associated worker id.
 */
extern int nw_worker_id;

#define MAX_API_ID 256

///// Commands

/**
 * Commands in the internal API.
 */
enum command_handler_command_id {
  COMMAND_HANDLER_INITIALIZE_API,
  COMMAND_HANDLER_SHUTDOWN_API,
  COMMAND_HANDLER_REPLACE_EXPLICIT_STATE,
  COMMAND_HANDLER_THREAD_EXIT,
  COMMAND_HANDLER_RECORDED_PAIR,
  COMMAND_START_MIGRATION,
  COMMAND_START_LIVE_MIGRATION,
  COMMAND_END_MIGRATION,
  COMMAND_ACCEPT_LIVE_MIGRATION,
  COMMAND_END_LIVE_MIGRATION,
  COMMAND_HANDLER_REGISTER_DUMP_DIR,
  COMMAND_HANDLER_REGISTER_VMID
};

struct command_handler_initialize_api_command {
  struct command_base base;
  intptr_t new_api_id;
  struct param_block_info pb_info;
};

struct ava_replay_command_pair_t {
  struct command_base base;
  void *call_cmd;
  void *ret_cmd;
};

#endif

/**
 * We use an internal API to communicate between components. This is
 * its ID.
 */
#define COMMAND_HANDLER_API 0
#define INTERNAL_API COMMAND_HANDLER_API

#ifdef __cplusplus
}
#endif

#endif  // AVA_COMMON_CMD_HANDLER_HPP_
