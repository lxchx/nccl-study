/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Binary call-graph trace for NCCL topo search functions.
 * Variable-length records written to an mmap'd ring buffer, flushed at exit.
 *
 * Record layout (variable length):
 *   uint8_t  func_id         // function ID (see NCCL_TRACE_FN_* below)
 *   uint8_t  dir             // ENTER(1) / EXIT(0)
 *   uint8_t  rank            // which rank produced this record
 *   uint8_t  nargs           // number of int64 args in this record
 *   uint32_t call_id         // global monotonic id assigned at ENTER
 *   uint32_t parent_call_id  // caller's call_id (for tree reconstruction)
 *   uint64_t tsc             // rdtsc timestamp
 *   uint64_t nrec            // total bytes of this record (header+payload)
 *   int64_t  arg[nargs]      // semantic depends on func_id + dir (see table)
 *
 * Header size = 28 bytes. Payload = nargs * 8 bytes.
 *************************************************************************/

#ifndef NCCL_CALLGRAPH_TRACE_H_
#define NCCL_CALLGRAPH_TRACE_H_

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Function IDs */
#define NCCL_TRACE_FN_Compute           1
#define NCCL_TRACE_FN_SearchInit        2
#define NCCL_TRACE_FN_SearchRec         3
#define NCCL_TRACE_FN_SearchTryGpu      4
#define NCCL_TRACE_FN_SearchRecGpu      5
#define NCCL_TRACE_FN_FollowPath        6
#define NCCL_TRACE_FN_CompareGraphs     7
#define NCCL_TRACE_FN_SearchNextGpuSort 8
#define NCCL_TRACE_FN_ReplayGetGpu      9

#define NCCL_TRACE_DIR_ENTER 1
#define NCCL_TRACE_DIR_EXIT  0

/* Header size = 28 bytes (8-byte aligned). */
#define NCCL_TRACE_HDR_SIZE  28

/* Initialize tracing. Safe to call multiple times.
 * Reads:
 *   NCCL_CALLGRAPH_TRACE_FILE  (default /tmp/nccl_callgraph.bin)
 *   NCCL_CALLGRAPH_TRACE_RANK  (default 0; set to -1 for all ranks)
 * Returns 0 on success. */
int ncclCallgraphTraceInit(void);

/* Returns 1 if tracing is enabled for this rank, 0 otherwise.
 * Call this before every trace point to skip overhead. */
int ncclCallgraphTraceEnabled(void);

/* Returns the rank being traced (or -1 if disabled or not yet initialized). */
int ncclCallgraphTraceRank(void);

/* Set the current thread's rank. Call this from each rank's init thread
 * before any trace point, so that the rank filter works correctly.
 * Returns 0 on success. */
int ncclCallgraphTraceSetRank(int rank);

/* Enter a function with up to 8 int64 args. Returns the call_id.
 * Pass nargs=0..8; extra args are ignored. */
uint32_t ncclCallgraphTraceEnter(uint8_t func_id, uint8_t nargs,
                                 int64_t a0, int64_t a1, int64_t a2, int64_t a3,
                                 int64_t a4, int64_t a5, int64_t a6, int64_t a7);

/* Exit a function. call_id is the one returned by Enter.
 * parent_call_id is read from the thread-local stack automatically. */
void ncclCallgraphTraceExit(uint32_t call_id, uint8_t func_id, uint8_t nargs,
                            int64_t a0, int64_t a1, int64_t a2, int64_t a3,
                            int64_t a4, int64_t a5, int64_t a6, int64_t a7);

/* ====== JSON-variant API (preferred; preserves original types) ======
 * Stores a JSON string payload. nargs field in the record is set to 0xFF
 * as a marker so the parser knows to read (nrec - HDR_SIZE) bytes as JSON
 * instead of int64[]. Use nlohmann::json on the caller side. */
uint32_t ncclCallgraphTraceEnterJson(uint8_t func_id, const char* json, size_t json_len);
void     ncclCallgraphTraceExitJson (uint32_t call_id, uint8_t func_id, const char* json, size_t json_len);

/* Thread-local call_id stack: so nested calls know their parent. */
void   ncclCallgraphTracePush(uint32_t call_id);
void   ncclCallgraphTracePop(void);
uint32_t ncclCallgraphTraceTopId(void);  /* 0 if empty */

/* Flush and close. Register with atexit. */
void ncclCallgraphTraceFlush(void);

#ifdef __cplusplus
}
#endif

/* ====== Convenience macros ====== */

#define NCCL_TRACE_ENTER8(fid, n, a0,a1,a2,a3,a4,a5,a6,a7) \
  ncclCallgraphTraceEnter(fid, n, (int64_t)(a0),(int64_t)(a1),(int64_t)(a2),(int64_t)(a3),\
                          (int64_t)(a4),(int64_t)(a5),(int64_t)(a6),(int64_t)(a7))
#define NCCL_TRACE_ENTER7(fid, a0,a1,a2,a3,a4,a5,a6) \
  NCCL_TRACE_ENTER8(fid, 7, a0,a1,a2,a3,a4,a5,a6, 0)
#define NCCL_TRACE_ENTER6(fid, a0,a1,a2,a3,a4,a5) \
  NCCL_TRACE_ENTER8(fid, 6, a0,a1,a2,a3,a4,a5, 0, 0)
#define NCCL_TRACE_ENTER5(fid, a0,a1,a2,a3,a4) \
  NCCL_TRACE_ENTER8(fid, 5, a0,a1,a2,a3,a4, 0, 0, 0)
#define NCCL_TRACE_ENTER4(fid, a0,a1,a2,a3) \
  NCCL_TRACE_ENTER8(fid, 4, a0,a1,a2,a3, 0, 0, 0, 0)
#define NCCL_TRACE_ENTER3(fid, a0,a1,a2) \
  NCCL_TRACE_ENTER8(fid, 3, a0,a1,a2, 0, 0, 0, 0, 0)
#define NCCL_TRACE_ENTER2(fid, a0,a1) \
  NCCL_TRACE_ENTER8(fid, 2, a0,a1, 0, 0, 0, 0, 0, 0)
#define NCCL_TRACE_ENTER1(fid, a0) \
  NCCL_TRACE_ENTER8(fid, 1, a0, 0, 0, 0, 0, 0, 0, 0)
#define NCCL_TRACE_ENTER0(fid) \
  NCCL_TRACE_ENTER8(fid, 0, 0,0,0,0,0,0,0,0)

/* JSON-variant: pass a std::string (or anything with .c_str() and .size()).
 * The caller builds the JSON with nlohmann::json and passes json.dump(). */
#define NCCL_TRACE_ENTER_JSON(fid, json_str) \
  ncclCallgraphTraceEnterJson(fid, (json_str).c_str(), (json_str).size())
#define NCCL_TRACE_EXIT_JSON(cid, fid, json_str) \
  ncclCallgraphTraceExitJson(cid, fid, (json_str).c_str(), (json_str).size())

#define NCCL_TRACE_EXIT8(cid, fid, n, a0,a1,a2,a3,a4,a5,a6,a7) \
  ncclCallgraphTraceExit(cid, fid, n, (int64_t)(a0),(int64_t)(a1),(int64_t)(a2),(int64_t)(a3),\
                          (int64_t)(a4),(int64_t)(a5),(int64_t)(a6),(int64_t)(a7))
#define NCCL_TRACE_EXIT7(cid, fid, a0,a1,a2,a3,a4,a5,a6) \
  NCCL_TRACE_EXIT8(cid, fid, 7, a0,a1,a2,a3,a4,a5,a6, 0)
#define NCCL_TRACE_EXIT6(cid, fid, a0,a1,a2,a3,a4,a5) \
  NCCL_TRACE_EXIT8(cid, fid, 6, a0,a1,a2,a3,a4,a5, 0, 0)
#define NCCL_TRACE_EXIT5(cid, fid, a0,a1,a2,a3,a4) \
  NCCL_TRACE_EXIT8(cid, fid, 5, a0,a1,a2,a3,a4, 0, 0, 0)
#define NCCL_TRACE_EXIT4(cid, fid, a0,a1,a2,a3) \
  NCCL_TRACE_EXIT8(cid, fid, 4, a0,a1,a2,a3, 0, 0, 0, 0)
#define NCCL_TRACE_EXIT3(cid, fid, a0,a1,a2) \
  NCCL_TRACE_EXIT8(cid, fid, 3, a0,a1,a2, 0, 0, 0, 0)
#define NCCL_TRACE_EXIT2(cid, fid, a0,a1) \
  NCCL_TRACE_EXIT8(cid, fid, 2, a0,a1, 0, 0, 0, 0, 0, 0)
#define NCCL_TRACE_EXIT1(cid, fid, a0) \
  NCCL_TRACE_EXIT8(cid, fid, 1, a0, 0, 0, 0, 0, 0, 0, 0)
#define NCCL_TRACE_EXIT0(cid, fid) \
  NCCL_TRACE_EXIT8(cid, fid, 0, 0,0,0,0,0,0,0,0)

/* ====== RAII helper for EXIT-only (ENTER called manually) ====== */
#ifdef __cplusplus
#include <utility>
#include <string>
struct NcclCallgraphTraceExitGuard {
  uint32_t cid;
  uint8_t  fid;
  bool     active;
  int64_t  exit_args[8];
  uint8_t  exit_nargs;
  std::string exit_json;  /* used when exit_nargs == 0xFF (JSON variant) */
  NcclCallgraphTraceExitGuard(uint32_t c, uint8_t f) : cid(c), fid(f), active(c != 0), exit_nargs(0) {
    for (int i = 0; i < 8; i++) exit_args[i] = 0;
  }
  ~NcclCallgraphTraceExitGuard() {
    if (!active) return;
    if (exit_nargs == 0xFF) {
      ncclCallgraphTraceExitJson(cid, fid, exit_json.c_str(), exit_json.size());
    } else {
      ncclCallgraphTraceExit(cid, fid, exit_nargs,
        exit_args[0], exit_args[1], exit_args[2], exit_args[3],
        exit_args[4], exit_args[5], exit_args[6], exit_args[7]);
    }
  }
  void setExit(uint8_t n) { exit_nargs = n; }
  void setExit(uint8_t n, int64_t a0) { exit_nargs=n; exit_args[0]=a0; }
  void setExit(uint8_t n, int64_t a0, int64_t a1) { exit_nargs=n; exit_args[0]=a0; exit_args[1]=a1; }
  void setExit(uint8_t n, int64_t a0, int64_t a1, int64_t a2) { exit_nargs=n; exit_args[0]=a0; exit_args[1]=a1; exit_args[2]=a2; }
  void setExit(uint8_t n, int64_t a0, int64_t a1, int64_t a2, int64_t a3) { exit_nargs=n; exit_args[0]=a0; exit_args[1]=a1; exit_args[2]=a2; exit_args[3]=a3; }
  void setExit(uint8_t n, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4) { exit_nargs=n; exit_args[0]=a0; exit_args[1]=a1; exit_args[2]=a2; exit_args[3]=a3; exit_args[4]=a4; }
  void setExit(uint8_t n, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5) { exit_nargs=n; exit_args[0]=a0; exit_args[1]=a1; exit_args[2]=a2; exit_args[3]=a3; exit_args[4]=a4; exit_args[5]=a5; }
  /* JSON variant: pass a pre-built JSON string. Sets exit_nargs=0xFF marker. */
  void setExitJson(const std::string& json_str) { exit_nargs = 0xFF; exit_json = json_str; }
  /* JSON variant from initializer list (builds an object): setExitJsonObj({{"k",v},...}). */
  template<typename T>
  void setExitJsonObj(T&& pairs) { exit_nargs = 0xFF; json j(std::forward<T>(pairs)); exit_json = j.dump(); }
};
/* Usage:
 *   uint32_t _cid = NCCL_TRACE_ENTER...(...);
 *   NcclCallgraphTraceExitGuard _trace_exit(_cid, FID);
 *   _trace_exit.setExit(n, args...);  // call before any return
 */
#endif

#endif /* NCCL_CALLGRAPH_TRACE_H_ */
