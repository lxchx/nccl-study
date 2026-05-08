/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_DEVICE_BARRIER_H_
#define _NCCL_DEVICE_BARRIER_H_
#include "impl/core__types.h"
#include "impl/lsa_barrier__types.h"
#include "impl/gin_barrier__types.h"
#include "gin_barrier.h"  // for ncclGinAllContexts

#if NCCL_CHECK_CUDACC
template<typename Coop>
struct ncclBarrierSession_internal;

template<typename Coop>
struct ncclBarrierSession: ncclBarrierSession_internal<Coop> {
  // Full featured constructor:
  NCCL_DEVICE_INLINE ncclBarrierSession(
    Coop, ncclTeam innerTeam, ncclTeam outerTeam, ncclGin,
    ncclLsaBarrierHandle innerBarHandle,
    ncclGinBarrierHandle outerBarHandle,
    uint32_t index,
    bool multimem=false, ncclMultimemHandle innerMmHandle={}
  );
  // Convenience constructors for baked in teams:
  NCCL_DEVICE_INLINE ncclBarrierSession(
    Coop, ncclTeamTagWorld, ncclGin, uint32_t index, bool multimem=false
  );
  NCCL_DEVICE_INLINE ncclBarrierSession(
    Coop, ncclTeamTagLsa, ncclDevComm const&, uint32_t index, bool multimem=false
  );
  NCCL_DEVICE_INLINE ncclBarrierSession(
    Coop, ncclTeamTagRail, ncclGin, uint32_t index
  );

  // All-contexts constructors: signal/wait runs on a fixed internal context (0); the fence
  // iterates every GIN context on the comm. Used internally by `ncclBarrier(coop, ...,
  // ncclGinAllContexts(comm), ...)` -- callers usually prefer the free-function form.
  NCCL_DEVICE_INLINE ncclBarrierSession(
    Coop, ncclTeam innerTeam, ncclTeam outerTeam, ncclGinAllContexts,
    ncclLsaBarrierHandle innerBarHandle,
    ncclGinBarrierHandle outerBarHandle,
    uint32_t index,
    bool multimem=false, ncclMultimemHandle innerMmHandle={}
  );
  NCCL_DEVICE_INLINE ncclBarrierSession(
    Coop, ncclTeamTagWorld, ncclGinAllContexts, uint32_t index, bool multimem=false
  );
  NCCL_DEVICE_INLINE ncclBarrierSession(
    Coop, ncclTeamTagRail, ncclGinAllContexts, uint32_t index
  );

  ncclBarrierSession(ncclBarrierSession const&) = delete; // Sessions are not copyable

  NCCL_DEVICE_INLINE ncclLsaBarrierSession<Coop>& lsaBarrier();
  NCCL_DEVICE_INLINE ncclGinBarrierSession<Coop>& ginBarrier();

  NCCL_DEVICE_INLINE void sync(Coop, cuda::memory_order, ncclGinFenceLevel = ncclGinFenceLevel::Put | ncclGinFenceLevel::Get);
  NCCL_DEVICE_INLINE ncclResult_t sync(Coop, cuda::memory_order, ncclGinFenceLevel, uint64_t timeoutCycles);
};

// Free-function hybrid barrier (LSA + rail-GIN). Wraps session construct + sync + destruct.
//
// `gin_or_allCtx` is either an `ncclGin` (single context for the rail-GIN inner) or
// `ncclGinAllContexts(comm)` (rail-GIN signal/wait runs on context 0; the fence iterates
// every context).

template<typename Coop>
NCCL_DEVICE_INLINE void ncclBarrier(Coop, ncclTeamTagWorld, ncclGin, uint32_t index,
    cuda::memory_order = cuda::memory_order_acq_rel,
    ncclGinFenceLevel = ncclGinFenceLevel::Put | ncclGinFenceLevel::Get,
    bool multimem = false);
template<typename Coop>
NCCL_DEVICE_INLINE void ncclBarrier(Coop, ncclTeamTagRail, ncclGin, uint32_t index,
    cuda::memory_order = cuda::memory_order_acq_rel,
    ncclGinFenceLevel = ncclGinFenceLevel::Put | ncclGinFenceLevel::Get);

template<typename Coop>
NCCL_DEVICE_INLINE void ncclBarrier(Coop, ncclTeamTagWorld, ncclGinAllContexts, uint32_t index,
    cuda::memory_order = cuda::memory_order_acq_rel,
    ncclGinFenceLevel = ncclGinFenceLevel::Put | ncclGinFenceLevel::Get,
    bool multimem = false);
template<typename Coop>
NCCL_DEVICE_INLINE void ncclBarrier(Coop, ncclTeamTagRail, ncclGinAllContexts, uint32_t index,
    cuda::memory_order = cuda::memory_order_acq_rel,
    ncclGinFenceLevel = ncclGinFenceLevel::Put | ncclGinFenceLevel::Get);
#endif

#endif // _NCCL_DEVICE_BARRIER_H_
