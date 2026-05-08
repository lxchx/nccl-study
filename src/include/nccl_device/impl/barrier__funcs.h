/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_DEVICE_BARRIER__FUNCS_H_
#define _NCCL_DEVICE_BARRIER__FUNCS_H_
#include "barrier__types.h"
#include "lsa_barrier__funcs.h"
#if defined(NCCL_OS_LINUX)
#include "gin_barrier__funcs.h"
#endif
#include "../utility.h"

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclBarrierSession<Coop>::ncclBarrierSession(
    Coop coop, ncclTeam innerTeam, ncclTeam outerTeam, ncclGin gin,
    ncclLsaBarrierHandle innerHandle, ncclGinBarrierHandle outerHandle,
    uint32_t index, bool multimem, ncclMultimemHandle innerMmHandle
  ):
  ncclBarrierSession_internal<Coop>(coop,
    nccl::utility::present(gin),
    nccl::utility::present(coop, gin.comm, innerTeam, innerHandle, index, multimem, innerMmHandle),
    nccl::utility::present(coop, gin, outerTeam, outerHandle, index)
  ) {
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclBarrierSession<Coop>::ncclBarrierSession(
    Coop coop, ncclTeamTagWorld, ncclGin gin, uint32_t index, bool multimem
  ):
  ncclBarrierSession<Coop>(
    coop, ncclTeamLsa(gin.comm), ncclTeamRail(gin.comm), gin,
    gin.comm.hybridLsaBarrier, gin.comm.hybridRailGinBarrier,
    index, multimem, gin.comm.lsaMultimem
  ) {
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclBarrierSession<Coop>::ncclBarrierSession(
    Coop coop, ncclTeamTagLsa, ncclDevComm const& comm, uint32_t index, bool multimem
  ):
  ncclBarrierSession_internal<Coop>(coop,
    nccl::utility::Absent(),
    nccl::utility::present(coop, comm, ncclTeamLsa(comm), comm.hybridLsaBarrier, index, multimem, comm.lsaMultimem),
    nccl::utility::Absent()
  ) {
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclBarrierSession<Coop>::ncclBarrierSession(
    Coop coop, ncclTeamTagRail, ncclGin gin, uint32_t index
  ):
  ncclBarrierSession_internal<Coop>(coop,
    nccl::utility::present(gin),
    nccl::utility::Absent(),
    nccl::utility::present(coop, gin, ncclTeamRail(gin.comm), gin.comm.hybridRailGinBarrier, index)
  ) {
}
#endif

// All-contexts hybrid-barrier constructors. Inner GIN barrier is built with the all-contexts
// tag so its fence iterates every GIN context on the comm. The outer LSA barrier is unchanged
// (LSA has no GIN ops to drain).
#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclBarrierSession<Coop>::ncclBarrierSession(
    Coop coop, ncclTeam innerTeam, ncclTeam outerTeam, ncclGinAllContexts allCtx,
    ncclLsaBarrierHandle innerHandle, ncclGinBarrierHandle outerHandle,
    uint32_t index, bool multimem, ncclMultimemHandle innerMmHandle
  ):
  ncclBarrierSession_internal<Coop>(coop,
    nccl::utility::present(ncclGin(allCtx.comm, 0)),
    nccl::utility::present(coop, allCtx.comm, innerTeam, innerHandle, index, multimem, innerMmHandle),
    nccl::utility::present(coop, allCtx, outerTeam, outerHandle, index)
  ) {
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclBarrierSession<Coop>::ncclBarrierSession(
    Coop coop, ncclTeamTagWorld, ncclGinAllContexts allCtx, uint32_t index, bool multimem
  ):
  ncclBarrierSession<Coop>(
    coop, ncclTeamLsa(allCtx.comm), ncclTeamRail(allCtx.comm), allCtx,
    allCtx.comm.hybridLsaBarrier, allCtx.comm.hybridRailGinBarrier,
    index, multimem, allCtx.comm.lsaMultimem
  ) {
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclBarrierSession<Coop>::ncclBarrierSession(
    Coop coop, ncclTeamTagRail, ncclGinAllContexts allCtx, uint32_t index
  ):
  ncclBarrierSession_internal<Coop>(coop,
    nccl::utility::present(ncclGin(allCtx.comm, 0)),
    nccl::utility::Absent(),
    nccl::utility::present(coop, allCtx, ncclTeamRail(allCtx.comm), allCtx.comm.hybridRailGinBarrier, index)
  ) {
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclLsaBarrierSession<Coop>& ncclBarrierSession<Coop>::lsaBarrier() {
  return this->innerLsaBar.thing;
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclGinBarrierSession<Coop>& ncclBarrierSession<Coop>::ginBarrier() {
  return this->outerGinBar.thing;
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE void ncclBarrierSession<Coop>::sync(Coop, cuda::memory_order ord, ncclGinFenceLevel fence) {
  if (this->innerLsaBar.present) {
    this->innerLsaBar.thing.sync(this->coop, this->outerGinBar.present ? nccl::utility::releaseOrderOf(ord) : ord);
  }
  if (this->outerGinBar.present) {
    this->outerGinBar.thing.sync(this->coop, this->innerLsaBar.present ? nccl::utility::acquireOrderOf(ord) : ord, fence);
  }
  // Two-pass LSA: when both inner LSA and outer rail-GIN are present and fence semantics are
  // requested, run a second intra-node LSA sync to propagate rail-GIN completion across rails.
  // This closes the cross-rail-cross-node knowledge gap so Barrier(World, fence!=None) matches
  // GinBarrier(World, fence!=None) at world scope. Skipped for fence=None (the all-arrived
  // guarantee is already provided by the single-pass causal chain).
  if (this->innerLsaBar.present && this->outerGinBar.present && fence != ncclGinFenceLevel::None) {
    this->innerLsaBar.thing.sync(this->coop, nccl::utility::acquireOrderOf(ord));
  }
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclResult_t ncclBarrierSession<Coop>::sync(
    Coop, cuda::memory_order ord, ncclGinFenceLevel fence, uint64_t timeoutCycles) {
  ncclResult_t lsaResult = ncclSuccess, railResult = ncclSuccess, lsaResult2 = ncclSuccess;

  // Inner LSA barrier (if present) - detects remote CTA/rank issues
  if (this->innerLsaBar.present) {
    uint64_t startCycle = clock64();
    lsaResult = this->innerLsaBar.thing.sync(
      this->coop,
      this->outerGinBar.present ? nccl::utility::releaseOrderOf(ord) : ord,
      timeoutCycles
    );
    uint64_t elapsed = clock64() - startCycle;
    timeoutCycles -= min(elapsed, timeoutCycles);
    // Because threads within a coop don't synchronize about the timeout condition,
    // we need to invoke the second barrier even if the first one times out,
    // to ensure that all the threads arrive at the coop sync.
  }

  // Outer GIN barrier (if present) - detects remote GPU/network issues
  if (this->outerGinBar.present) {
    uint64_t startCycle = clock64();
    railResult = this->outerGinBar.thing.sync(
      this->coop,
      this->innerLsaBar.present ? nccl::utility::acquireOrderOf(ord) : ord,
      fence,
      timeoutCycles
    );
    uint64_t elapsed = clock64() - startCycle;
    timeoutCycles -= min(elapsed, timeoutCycles);
  }

  // Two-pass LSA: second intra-node sync after rail-GIN propagates rail-GIN completion across
  // rails. Required for Barrier(World, fence!=None) to match GinBarrier(World, fence!=None) at
  // world scope. Skipped for fence=None.
  if (this->innerLsaBar.present && this->outerGinBar.present && fence != ncclGinFenceLevel::None) {
    lsaResult2 = this->innerLsaBar.thing.sync(
      this->coop,
      nccl::utility::acquireOrderOf(ord),
      timeoutCycles
    );
  }

  if (lsaResult != ncclSuccess) return lsaResult;
  if (railResult != ncclSuccess) return railResult;
  return lsaResult2;
}
#endif

// Free-function hybrid barrier: thin wrappers around session construct + sync + destruct.
#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE void ncclBarrier(
    Coop coop, ncclTeamTagWorld tag, ncclGin gin, uint32_t index,
    cuda::memory_order ord, ncclGinFenceLevel fence, bool multimem) {
  ncclBarrierSession<Coop> session(coop, tag, gin, index, multimem);
  session.sync(coop, ord, fence);
}

template<typename Coop>
NCCL_DEVICE_INLINE void ncclBarrier(
    Coop coop, ncclTeamTagRail tag, ncclGin gin, uint32_t index,
    cuda::memory_order ord, ncclGinFenceLevel fence) {
  ncclBarrierSession<Coop> session(coop, tag, gin, index);
  session.sync(coop, ord, fence);
}

template<typename Coop>
NCCL_DEVICE_INLINE void ncclBarrier(
    Coop coop, ncclTeamTagWorld tag, ncclGinAllContexts allCtx, uint32_t index,
    cuda::memory_order ord, ncclGinFenceLevel fence, bool multimem) {
  ncclBarrierSession<Coop> session(coop, tag, allCtx, index, multimem);
  session.sync(coop, ord, fence);
}

template<typename Coop>
NCCL_DEVICE_INLINE void ncclBarrier(
    Coop coop, ncclTeamTagRail tag, ncclGinAllContexts allCtx, uint32_t index,
    cuda::memory_order ord, ncclGinFenceLevel fence) {
  ncclBarrierSession<Coop> session(coop, tag, allCtx, index);
  session.sync(coop, ord, fence);
}
#endif

#endif // _NCCL_DEVICE_BARRIER__FUNCS_H_
