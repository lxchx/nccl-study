/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Binary call-graph trace implementation.
 *************************************************************************/

#include "callgraph_trace.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <pthread.h>
#include <stdint.h>

/* ====== Record header (28 bytes) ======
 * Packed layout, written as raw bytes to avoid struct padding issues.
 */
#pragma pack(push, 1)
typedef struct {
  uint8_t  func_id;
  uint8_t  dir;
  uint8_t  rank;
  uint8_t  nargs;
  uint32_t call_id;
  uint32_t parent_call_id;
  uint64_t tsc;
  uint64_t nrec;   // total bytes including header
} trace_hdr_t;
#pragma pack(pop)

#define HDR_SIZE 28

/* ====== Global trace state (per-process) ======
 * For now we assume the topo search runs on a single thread per rank,
 * so a simple mutex-guarded buffer is fine. If multiple ranks share a
 * process (ncclCommInitAll), each rank thread will have its own
 * thread-local state via __thread.
 */
static int g_enabled = 0;
static int g_rank = -1;
static int g_rank_filter = 0;   /* -1 means trace all ranks */

/* Thread-local per-rank state */
static __thread int       t_initialized = 0;
static __thread uint8_t   t_rank = 0;
static __thread uint32_t  t_next_call_id = 1;
static __thread uint32_t  t_call_stack[256];
static __thread int       t_call_stack_top = 0;

/* Per-thread output buffer (mmap'd). We write to a per-thread file
 * to avoid cross-thread locking. File name = base + .rank + .tid.
 */
static __thread int       t_fd = -1;
static __thread uint8_t*  t_buf = NULL;
static __thread size_t    t_buf_size = 0;
static __thread size_t    t_buf_used = 0;
static char               g_file_base[512];

/* pthread key to auto-flush when each thread exits. */
static pthread_key_t      g_thread_key;
static int                g_key_created = 0;

#define TRACE_BUF_SIZE (256 * 1024 * 1024)  /* 256MB per thread */

/* Thread destructor: flush this thread's buffer. */
static void thread_destructor(void* arg) {
  (void)arg;
  ncclCallgraphTraceFlush();
}

static inline uint64_t rdtsc(void) {
#if defined(__x86_64__)
  unsigned lo, hi;
  __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
  return ((uint64_t)hi << 32) | lo;
#else
  return 0;
#endif
}

/* Build the per-thread file name. */
static void build_thread_filename(char* out, size_t outsz) {
  char* slash = strrchr(g_file_base, '/');
  const char* dir = ".";
  const char* base = g_file_base;
  if (slash) {
    size_t dlen = slash - g_file_base;
    if (dlen >= outsz) dlen = outsz - 1;
    memcpy(out, g_file_base, dlen);
    out[dlen] = '\0';
    dir = out;
    base = slash + 1;
  }
  char tmp[600];
  snprintf(tmp, sizeof(tmp), "%s/%s.r%d.t%lu", dir, base, (int)t_rank, (unsigned long)pthread_self());
  strncpy(out, tmp, outsz - 1);
  out[outsz - 1] = '\0';
}

int ncclCallgraphTraceInit(void) {
  if (g_enabled) return 0;
  const char* f = getenv("NCCL_CALLGRAPH_TRACE_FILE");
  if (!f) f = "/tmp/nccl_callgraph.bin";
  strncpy(g_file_base, f, sizeof(g_file_base) - 1);
  g_file_base[sizeof(g_file_base) - 1] = '\0';

  const char* r = getenv("NCCL_CALLGRAPH_TRACE_RANK");
  g_rank_filter = r ? atoi(r) : 0;
  g_enabled = 1;
  /* Create pthread key for per-thread cleanup (only once). */
  if (!g_key_created) {
    pthread_key_create(&g_thread_key, thread_destructor);
    g_key_created = 1;
  }
  /* Register flush at exit (only once for the process). */
  static int registered = 0;
  if (!registered) {
    atexit(ncclCallgraphTraceFlush);
    registered = 1;
  }
  return 0;
}

/* Called lazily on first trace point of a thread to set up its rank
 * and open its output file. */
static void ensure_thread_init(void) {
  if (t_initialized) return;
  /* t_rank was set by ncclCallgraphTraceSetRank() from the comm init path.
   * If not set, default to 0 (shouldn't happen in normal flow). */

  /* Check rank filter */
  if (g_rank_filter >= 0 && (int)t_rank != g_rank_filter) {
    t_initialized = 1;  /* initialized but disabled for this thread */
    return;
  }

  char fname[700];
  build_thread_filename(fname, sizeof(fname));
  t_fd = open(fname, O_RDWR | O_CREAT | O_TRUNC, 0644);
  if (t_fd < 0) {
    t_initialized = 1;
    return;
  }
  /* Pre-size the file to TRACE_BUF_SIZE so we can mmap it. */
  if (ftruncate(t_fd, TRACE_BUF_SIZE) != 0) {
    close(t_fd); t_fd = -1;
    t_initialized = 1;
    return;
  }
  t_buf = (uint8_t*)mmap(NULL, TRACE_BUF_SIZE, PROT_READ | PROT_WRITE,
                         MAP_SHARED, t_fd, 0);
  if (t_buf == MAP_FAILED) {
    t_buf = NULL;
    close(t_fd); t_fd = -1;
    t_initialized = 1;
    return;
  }
  t_buf_size = TRACE_BUF_SIZE;
  t_buf_used = 0;
  t_initialized = 1;
  /* Associate this thread with the key so thread_destructor runs on exit. */
  pthread_setspecific(g_thread_key, (void*)1);
}

int ncclCallgraphTraceEnabled(void) {
  if (!g_enabled) return 0;
  if (!t_initialized) ensure_thread_init();
  return (t_buf != NULL);
}

int ncclCallgraphTraceRank(void) {
  return (int)t_rank;
}

int ncclCallgraphTraceSetRank(int rank) {
  /* If rank changed on this thread (thread pool reuse), discard the old
   * buffer (it was written under the wrong rank) and reset so
   * ensure_thread_init re-evaluates the filter for the new rank. */
  if (t_initialized && t_rank != (uint8_t)rank) {
    /* Close and delete the stale file. */
    if (t_buf) {
      munmap(t_buf, t_buf_size);
      t_buf = NULL;
    }
    if (t_fd >= 0) {
      /* Best-effort: get the filename to unlink. We don't store it, so
       * just close; the file will be empty/tiny. To actually unlink we
       * would need to remember the path. For now, truncate to 0. */
      ftruncate(t_fd, 0);
      close(t_fd);
      t_fd = -1;
    }
    t_buf_size = 0;
    t_buf_used = 0;
    t_initialized = 0;
  }
  t_rank = (uint8_t)rank;
  return 0;
}

/* Write one record into the thread buffer. If buffer is full, flush to
 * the mmap'd file (which grows on disk as we write past the initial
 * size — but we cap usage at TRACE_BUF_SIZE). For NCCL init this is
 * plenty; if we ever exceed it, we silently drop further records. */
static void write_record(uint8_t func_id, uint8_t dir, uint8_t nargs,
                         uint32_t call_id, uint32_t parent_call_id,
                         int64_t* args) {
  if (!t_buf) return;
  size_t rec_size = HDR_SIZE + (size_t)nargs * 8;
  if (t_buf_used + rec_size > t_buf_size) return;  /* drop on overflow */

  trace_hdr_t hdr;
  hdr.func_id        = func_id;
  hdr.dir            = dir;
  hdr.rank           = t_rank;
  hdr.nargs          = nargs;
  hdr.call_id        = call_id;
  hdr.parent_call_id = parent_call_id;
  hdr.tsc            = rdtsc();
  hdr.nrec           = rec_size;

  uint8_t* p = t_buf + t_buf_used;
  memcpy(p, &hdr, HDR_SIZE);
  if (nargs > 0) memcpy(p + HDR_SIZE, args, (size_t)nargs * 8);
  t_buf_used += rec_size;
}

/* Write a JSON-payload record. nargs is set to 0xFF to mark "JSON variant";
 * the payload is a UTF-8 JSON string (NOT null-terminated in the buffer —
 * length is derived from nrec). The parser/exporter checks nargs==0xFF and
 * reads (nrec - HDR_SIZE) bytes as JSON. */
static void write_record_json(uint8_t func_id, uint8_t dir,
                              uint32_t call_id, uint32_t parent_call_id,
                              const char* json, size_t json_len) {
  if (!t_buf) return;
  size_t rec_size = HDR_SIZE + json_len;
  if (t_buf_used + rec_size > t_buf_size) return;  /* drop on overflow */

  trace_hdr_t hdr;
  hdr.func_id        = func_id;
  hdr.dir            = dir;
  hdr.rank           = t_rank;
  hdr.nargs          = 0xFF;  /* JSON marker */
  hdr.call_id        = call_id;
  hdr.parent_call_id = parent_call_id;
  hdr.tsc            = rdtsc();
  hdr.nrec           = rec_size;

  uint8_t* p = t_buf + t_buf_used;
  memcpy(p, &hdr, HDR_SIZE);
  if (json_len > 0) memcpy(p + HDR_SIZE, json, json_len);
  t_buf_used += rec_size;
}

uint32_t ncclCallgraphTraceEnter(uint8_t func_id, uint8_t nargs,
                                 int64_t a0, int64_t a1, int64_t a2, int64_t a3,
                                 int64_t a4, int64_t a5, int64_t a6, int64_t a7) {
  if (!ncclCallgraphTraceEnabled()) return 0;
  int64_t args[8] = {a0, a1, a2, a3, a4, a5, a6, a7};
  if (nargs > 8) nargs = 8;
  uint32_t cid = t_next_call_id++;
  uint32_t parent = (t_call_stack_top > 0) ? t_call_stack[t_call_stack_top - 1] : 0;
  write_record(func_id, NCCL_TRACE_DIR_ENTER, nargs, cid, parent, args);
  /* Push this call_id onto the stack so nested calls see it as parent. */
  if (t_call_stack_top < (int)(sizeof(t_call_stack) / sizeof(t_call_stack[0]))) {
    t_call_stack[t_call_stack_top++] = cid;
  }
  return cid;
}

void ncclCallgraphTraceExit(uint32_t call_id, uint8_t func_id, uint8_t nargs,
                            int64_t a0, int64_t a1, int64_t a2, int64_t a3,
                            int64_t a4, int64_t a5, int64_t a6, int64_t a7) {
  if (!ncclCallgraphTraceEnabled()) return;
  int64_t args[8] = {a0, a1, a2, a3, a4, a5, a6, a7};
  if (nargs > 8) nargs = 8;
  /* parent_call_id is the current top (which should be call_id itself on EXIT).
   * We record the parent of the caller instead, which is what's left after pop. */
  if (t_call_stack_top > 0) t_call_stack_top--;
  uint32_t parent = (t_call_stack_top > 0) ? t_call_stack[t_call_stack_top - 1] : 0;
  write_record(func_id, NCCL_TRACE_DIR_EXIT, nargs, call_id, parent, args);
}

/* ====== JSON-variant API (preferred for new trace points) ======
 * Stores a JSON string payload instead of fixed int64 args. Preserves
 * original types (float stays float). Use nlohmann::json on the caller side
 * to build the string, then pass json.dump().c_str() here. */
uint32_t ncclCallgraphTraceEnterJson(uint8_t func_id, const char* json, size_t json_len) {
  if (!ncclCallgraphTraceEnabled()) return 0;
  uint32_t cid = t_next_call_id++;
  uint32_t parent = (t_call_stack_top > 0) ? t_call_stack[t_call_stack_top - 1] : 0;
  write_record_json(func_id, NCCL_TRACE_DIR_ENTER, cid, parent, json, json_len);
  if (t_call_stack_top < (int)(sizeof(t_call_stack) / sizeof(t_call_stack[0]))) {
    t_call_stack[t_call_stack_top++] = cid;
  }
  return cid;
}

void ncclCallgraphTraceExitJson(uint32_t call_id, uint8_t func_id, const char* json, size_t json_len) {
  if (!ncclCallgraphTraceEnabled()) return;
  if (t_call_stack_top > 0) t_call_stack_top--;
  uint32_t parent = (t_call_stack_top > 0) ? t_call_stack[t_call_stack_top - 1] : 0;
  write_record_json(func_id, NCCL_TRACE_DIR_EXIT, call_id, parent, json, json_len);
}

void ncclCallgraphTracePush(uint32_t call_id) {
  if (t_call_stack_top < (int)(sizeof(t_call_stack) / sizeof(t_call_stack[0]))) {
    t_call_stack[t_call_stack_top++] = call_id;
  }
}

void ncclCallgraphTracePop(void) {
  if (t_call_stack_top > 0) t_call_stack_top--;
}

uint32_t ncclCallgraphTraceTopId(void) {
  return (t_call_stack_top > 0) ? t_call_stack[t_call_stack_top - 1] : 0;
}

void ncclCallgraphTraceFlush(void) {
  if (!t_buf) return;
  /* Truncate the file to the actual used size. */
  msync(t_buf, t_buf_used, MS_SYNC);
  munmap(t_buf, t_buf_size);
  if (t_fd >= 0) {
    ftruncate(t_fd, t_buf_used);
    close(t_fd);
    t_fd = -1;
  }
  t_buf = NULL;
  t_buf_used = 0;
  t_buf_size = 0;
  /* Mark not-initialized so a subsequent trace point would re-init — but
   * at exit time this won't happen. */
  t_initialized = 0;
}
