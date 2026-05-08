/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_DEVICE_GIN_BARRIER_H_
#define _NCCL_DEVICE_GIN_BARRIER_H_
#include "core.h"
#if defined(NCCL_OS_WINDOWS)
#include "gin_win_stub.h"
#else
#include "gin.h"
#endif

struct ncclGinBarrierHandle;

NCCL_EXTERN_C __host__ ncclResult_t ncclGinBarrierCreateRequirement(ncclComm_t, ncclTeam_t, int nBarriers, ncclGinBarrierHandle_t* outHandle, ncclDevResourceRequirements_t* outReq);

#if NCCL_CHECK_CUDACC
// Bit-flag enum: Put and Get (and any future flags) are independent bits that compose via
// bitwise OR. There is no separate combined-everything enumerator -- callers wanting both
// write `Put | Get` explicitly. Underlying type is fixed at uint32_t so the operators below
// are well-defined.
enum class ncclGinFenceLevel : uint32_t {
  None    = 0,        // Pure synchronization. No drain.
  Put     = 1u << 0,  // After barrier returns, all prior puts from this rank on the bound GIN context have settled at their destinations.
  Get     = 1u << 1,  // After barrier returns, all prior gets from this rank on the bound GIN context have landed in local memory.
  Relaxed = None,     // Deprecated alias for None; kept for source-level backward compatibility.
};

// Composition operators so callers can write `ncclGinFenceLevel::Put | ncclGinFenceLevel::Get`
// (without these, enum class disallows the implicit conversions a bitwise-OR would need).
// Marked __host__ __device__ so the operators can be invoked from both host code (e.g.,
// default-argument materialization) and device code (e.g., barrier syncInternal).
NCCL_HOST_DEVICE_INLINE constexpr ncclGinFenceLevel operator|(ncclGinFenceLevel a, ncclGinFenceLevel b) {
  return static_cast<ncclGinFenceLevel>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
NCCL_HOST_DEVICE_INLINE constexpr ncclGinFenceLevel operator&(ncclGinFenceLevel a, ncclGinFenceLevel b) {
  return static_cast<ncclGinFenceLevel>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}

// Pass `ncclGinAllContexts(comm)` to a barrier in place of an `ncclGin` to signal "fence
// drains every GIN context on the comm". The barrier still uses one concrete context
// (context 0) for the signal/wait coordination; only the fence is expanded.
//
// `ncclGin` itself stays a single-context handle -- `gin.put` / `gin.get` / `gin.signal` are
// always unambiguous. Multi-context fence semantics live on the barrier API surface, not on
// the gin object.
struct ncclGinAllContexts {
  ncclDevComm const& comm;
  NCCL_HOST_DEVICE_INLINE constexpr ncclGinAllContexts(ncclDevComm const& comm_): comm(comm_) {}
};

template<typename Coop>
struct ncclGinBarrierSession_internal;

template<typename Coop>
struct ncclGinBarrierSession: ncclGinBarrierSession_internal<Coop> {
  // Single-context constructors: `gin` carries both the signal/wait context and the fence scope.
  NCCL_DEVICE_INLINE ncclGinBarrierSession(Coop, ncclGin, ncclTeam, ncclGinBarrierHandle, uint32_t index);
  NCCL_DEVICE_INLINE ncclGinBarrierSession(Coop, ncclGin, ncclTeamTagRail, uint32_t index);
  NCCL_DEVICE_INLINE ncclGinBarrierSession(Coop, ncclGin, ncclTeamTagWorld, uint32_t index);

  // All-contexts constructors: signal/wait runs on a fixed internal context (0); the fence
  // iterates every GIN context on the comm. Used internally by the free-function
  // `ncclGinBarrier(coop, ncclGinAllContexts(comm), ...)` overloads -- callers usually
  // prefer the free-function form.
  NCCL_DEVICE_INLINE ncclGinBarrierSession(Coop, ncclGinAllContexts, ncclTeam, ncclGinBarrierHandle, uint32_t index);
  NCCL_DEVICE_INLINE ncclGinBarrierSession(Coop, ncclGinAllContexts, ncclTeamTagRail, uint32_t index);
  NCCL_DEVICE_INLINE ncclGinBarrierSession(Coop, ncclGinAllContexts, ncclTeamTagWorld, uint32_t index);

  NCCL_DEVICE_INLINE ~ncclGinBarrierSession();

  ncclGinBarrierSession(ncclGinBarrierSession const&) = delete; // Sessions are not copyable

  NCCL_DEVICE_INLINE void sync(Coop, cuda::memory_order, ncclGinFenceLevel = ncclGinFenceLevel::Put | ncclGinFenceLevel::Get);
  NCCL_DEVICE_INLINE ncclResult_t sync(Coop, cuda::memory_order, ncclGinFenceLevel, uint64_t timeoutCycles);
};

// Free-function GIN barrier. Wraps session construct + sync + destruct so callers don't need
// to manage a session object for one-shot barriers.
//
// `gin_or_allCtx` is either an `ncclGin` (single context for both signal and fence) or
// `ncclGinAllContexts(comm)` (signal on context 0; fence iterates every context on the comm).

template<typename Coop>
NCCL_DEVICE_INLINE void ncclGinBarrier(Coop, ncclGin, ncclTeam, ncclGinBarrierHandle, uint32_t index,
    cuda::memory_order = cuda::memory_order_acq_rel,
    ncclGinFenceLevel = ncclGinFenceLevel::Put | ncclGinFenceLevel::Get);
template<typename Coop>
NCCL_DEVICE_INLINE void ncclGinBarrier(Coop, ncclGin, ncclTeamTagRail, uint32_t index,
    cuda::memory_order = cuda::memory_order_acq_rel,
    ncclGinFenceLevel = ncclGinFenceLevel::Put | ncclGinFenceLevel::Get);
template<typename Coop>
NCCL_DEVICE_INLINE void ncclGinBarrier(Coop, ncclGin, ncclTeamTagWorld, uint32_t index,
    cuda::memory_order = cuda::memory_order_acq_rel,
    ncclGinFenceLevel = ncclGinFenceLevel::Put | ncclGinFenceLevel::Get);

template<typename Coop>
NCCL_DEVICE_INLINE void ncclGinBarrier(Coop, ncclGinAllContexts, ncclTeam, ncclGinBarrierHandle, uint32_t index,
    cuda::memory_order = cuda::memory_order_acq_rel,
    ncclGinFenceLevel = ncclGinFenceLevel::Put | ncclGinFenceLevel::Get);
template<typename Coop>
NCCL_DEVICE_INLINE void ncclGinBarrier(Coop, ncclGinAllContexts, ncclTeamTagRail, uint32_t index,
    cuda::memory_order = cuda::memory_order_acq_rel,
    ncclGinFenceLevel = ncclGinFenceLevel::Put | ncclGinFenceLevel::Get);
template<typename Coop>
NCCL_DEVICE_INLINE void ncclGinBarrier(Coop, ncclGinAllContexts, ncclTeamTagWorld, uint32_t index,
    cuda::memory_order = cuda::memory_order_acq_rel,
    ncclGinFenceLevel = ncclGinFenceLevel::Put | ncclGinFenceLevel::Get);
#endif

#endif // _NCCL_DEVICE_GIN_BARRIER_H_
