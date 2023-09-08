//
// Created by amp on 4/2/19.
//

#include "common/shadow_thread_pool.hpp"

#include <assert.h>
#include <plog/Log.h>
#include <stdio.h>

#include <fmt/core.h>
#include <gsl/gsl>
#include <unistd.h>
#include <sys/syscall.h>

#include "common/cmd_handler.hpp"
#include "common/endpoint_lib.hpp"
#include "common/linkage.h"
#include "common/logging.h"
#include "common/support/fs.h"
#include "common/support/io.h"

#include <mutex>

static constexpr const char* kWorkerStatsPath = "/tmp";
static constexpr const char* kWorkerStatsPrefix = "worker_stats";

struct shadow_thread_pool_t {
  GHashTable *threads; /* Keys are ava IDs, values are shadow_thread_t* */
  pthread_mutex_t lock;
  pthread_key_t key;
};

struct shadow_thread_t {
  uintptr_t ava_id;
  GAsyncQueue *queue;
  pthread_t thread;
  struct shadow_thread_pool_t *pool;
  int cmd_stat_fd;
};

struct shadow_thread_command_t {
  struct command_channel *chan;
  struct command_base *cmd;
};

static void *shadow_thread_loop(void *arg);

struct shadow_thread_t *shadow_thread_new(struct shadow_thread_pool_t *pool, uintptr_t ava_id) {
  assert(g_hash_table_lookup(pool->threads, reinterpret_cast<gpointer>(ava_id)) == NULL);
  AVA_DEBUG << "Creating shadow thread id = " << ava_id;
  struct shadow_thread_t *t = static_cast<struct shadow_thread_t *>(malloc(sizeof(struct shadow_thread_t)));
  t->ava_id = ava_id;
  t->queue = g_async_queue_new_full(NULL);
  t->pool = pool;
  t->cmd_stat_fd = -1;
  int r = pthread_create(&t->thread, NULL, shadow_thread_loop, t);
  assert(r == 0);
  assert(t->thread != ava_id);  // TODO: This may spuriously fail.
  r = g_hash_table_insert(pool->threads, reinterpret_cast<gpointer>(ava_id), t);
  assert(r);
  (void)r;
  return t;
}

struct shadow_thread_t *shadow_thread_self(struct shadow_thread_pool_t *pool) {
  struct shadow_thread_t *t = (struct shadow_thread_t *)pthread_getspecific(pool->key);
  if (t == NULL) {
    t = (struct shadow_thread_t *)malloc(sizeof(struct shadow_thread_t));
    uintptr_t ava_id = static_cast<uintptr_t>(pthread_self());  // TODO: This may not work correctly on non-Linux
    assert(g_hash_table_lookup(pool->threads, (gpointer)ava_id) == NULL);
    t->ava_id = ava_id;
    t->queue = g_async_queue_new_full(NULL);
    t->pool = pool;
    t->thread = pthread_self();
    auto stat_fname = fmt::format("{}/{}_{}", kWorkerStatsPath, kWorkerStatsPrefix, 
                                  gsl::narrow_cast<int>(syscall(SYS_gettid)));
    auto stat_path = ava::support::GetRealPath(stat_fname);
    if (auto fd = ava::support::Create(stat_path)) {
      t->cmd_stat_fd = *fd;
    } else {
      t->cmd_stat_fd = STDOUT_FILENO;
    }
    gboolean r = g_hash_table_insert(pool->threads, (gpointer)ava_id, t);
    assert(r);
    (void)r;
    pthread_setspecific(pool->key, t);
  }
  return t;
}

void shadow_thread_free_from_thread(struct shadow_thread_t *t) {
  pthread_mutex_lock(&t->pool->lock);

  // If our ID is the same as the local thread reference then we must be a solid
  // (instead of shadow) thread. If we are solid, send a command to exit the
  // shadow.
  if (t->ava_id == t->thread) {
    struct command_base *cmd = command_channel_new_command(nw_global_command_channel, sizeof(struct command_base), 0);
    cmd->api_id = COMMAND_HANDLER_API;
    cmd->command_id = COMMAND_HANDLER_THREAD_EXIT;
    cmd->thread_id = t->ava_id;
    command_channel_send_command(nw_global_command_channel, cmd);
  }

  // Drop this thread from the pool.
  g_hash_table_remove(t->pool->threads, (gpointer)t->ava_id);
  pthread_mutex_unlock(&t->pool->lock);

  g_async_queue_unref(t->queue);
  t->queue = NULL;
  free(t);
}

struct shadow_thread_pool_t *shadow_thread_pool_new() {
  struct shadow_thread_pool_t *pool = (struct shadow_thread_pool_t *)malloc(sizeof(struct shadow_thread_pool_t));
  pool->threads = g_hash_table_new_full(nw_hash_pointer, g_direct_equal, NULL, NULL);
  pthread_key_create(&pool->key, (void (*)(void *))shadow_thread_free_from_thread);
  pthread_mutex_init(&pool->lock, NULL);
  return pool;
}

intptr_t shadow_thread_id(struct shadow_thread_pool_t *pool) {
  struct shadow_thread_t *t = shadow_thread_self(pool);
  return t->ava_id;
}

int shadow_thread_handle_single_command(struct shadow_thread_pool_t *pool) {
  struct shadow_thread_t *t = shadow_thread_self(pool);
  struct shadow_thread_command_t *scmd = (struct shadow_thread_command_t *)g_async_queue_pop(t->queue);

  struct command_channel *chan = scmd->chan;
  struct command_base *cmd = scmd->cmd;
  free(scmd);

  if (cmd->api_id == COMMAND_HANDLER_API && cmd->command_id == COMMAND_HANDLER_THREAD_EXIT) {
    command_channel_free_command(chan, cmd);
    return 1;
  }

  assert(gsl::narrow_cast<uintptr_t>(cmd->thread_id) == t->ava_id);
  // TODO: checks MSG_SHUTDOWN messages/channel close from the other side.

  handle_command_and_notify(chan, cmd);
  return 0;
}

static void *shadow_thread_loop(void *arg) {
  struct shadow_thread_t *t = (struct shadow_thread_t *)arg;
  if (t->cmd_stat_fd == -1) {
    auto stat_fname = fmt::format("{}/{}_{}", kWorkerStatsPath, kWorkerStatsPrefix, 
                                  gsl::narrow_cast<int>(syscall(SYS_gettid)));
    auto stat_path = ava::support::GetRealPath(stat_fname);
    if (auto fd = ava::support::Create(stat_path)) {
      t->cmd_stat_fd = *fd;
    } else {
      t->cmd_stat_fd = STDOUT_FILENO;
    }
  }
  pthread_setspecific(t->pool->key, t);
  
  int exit_thread_flag;
  do {
    exit_thread_flag = shadow_thread_handle_single_command(t->pool);
  } while (!exit_thread_flag);

  //printf(" !!!!  shadow thread exited\n");
  if (t->cmd_stat_fd != STDOUT_FILENO) {
    close(t->cmd_stat_fd);
  }
  return NULL;
}

void shadow_thread_pool_free(struct shadow_thread_pool_t *pool) {
  pthread_mutex_lock(&pool->lock);
  g_hash_table_destroy(pool->threads);
  free(pool);
}

void shadow_thread_pool_dispatch(struct shadow_thread_pool_t *pool, struct command_channel *chan,
                                 struct command_base *cmd) {
  pthread_mutex_lock(&pool->lock);
  struct shadow_thread_t *t = (struct shadow_thread_t *)g_hash_table_lookup(pool->threads, (gpointer)cmd->thread_id);
  if (t == NULL) {
    if (cmd->api_id == INTERNAL_API && cmd->command_id == COMMAND_HANDLER_THREAD_EXIT) {
      // If a thread for which we have no shadow is exiting, just drop the
      // message.
      pthread_mutex_unlock(&pool->lock);
      return;
    }
    t = shadow_thread_new(pool, cmd->thread_id);
  }
  struct shadow_thread_command_t *scmd =
      (struct shadow_thread_command_t *)malloc(sizeof(struct shadow_thread_command_t));
  scmd->chan = chan;
  scmd->cmd = cmd;
  g_async_queue_push(t->queue, scmd);
  pthread_mutex_unlock(&pool->lock);
}


void delete_shadow_thread(gpointer key, gpointer value, gpointer userdata) {
  printf("canceling shadow thread\n");
  struct shadow_thread_t *t = (struct shadow_thread_t *) value;
  pthread_cancel(t->thread);
}

void kill_all_shadow_threads() {
  auto context = ava::CommonContext::instance();
  g_hash_table_foreach(context->nw_shadow_thread_pool->threads, delete_shadow_thread, NULL);
}

void worker_write_stats(struct shadow_thread_pool_t *pool, const char *data, size_t size) {
  struct shadow_thread_t *t = shadow_thread_self(pool);
  ava::support::WriteData(t->cmd_stat_fd, data, size);
}