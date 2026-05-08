/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_DEVICE_GIN_BARRIER__FUNCS_H_
#define _NCCL_DEVICE_GIN_BARRIER__FUNCS_H_
#include "gin_barrier__types.h"

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclGinBarrierSession<Coop>::ncclGinBarrierSession(
    Coop coop, ncclGin net, ncclTeam team, ncclGinBarrierHandle handle, uint32_t barrierIndex
  ):
  ncclGinBarrierSession_internal<Coop>{coop, net, team, handle, (int)barrierIndex} {
  this->signal = handle.signal0 + barrierIndex * team.nRanks;
  this->fenceAllContexts = false;
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclGinBarrierSession<Coop>::ncclGinBarrierSession(
    Coop coop, ncclGin net, ncclTeamTagRail, uint32_t barrierIndex
  ):
  ncclGinBarrierSession(coop, net, ncclTeamRail(net.comm), net.comm.railGinBarrier, barrierIndex) {
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclGinBarrierSession<Coop>::ncclGinBarrierSession(
    Coop coop, ncclGin net, ncclTeamTagWorld, uint32_t barrierIndex
  ):
  ncclGinBarrierSession(coop, net, ncclTeamWorld(net.comm), net.comm.worldGinBarrier, barrierIndex) {
}
#endif

// All-contexts constructors: build a single-context gin (context 0) for the signal/wait
// path, then flip the `fenceAllContexts` flag so the fence iterates every GIN context on
// the comm.
#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclGinBarrierSession<Coop>::ncclGinBarrierSession(
    Coop coop, ncclGinAllContexts allCtx, ncclTeam team, ncclGinBarrierHandle handle, uint32_t barrierIndex
  ):
  ncclGinBarrierSession_internal<Coop>{coop, ncclGin(allCtx.comm, 0), team, handle, (int)barrierIndex} {
  this->signal = handle.signal0 + barrierIndex * team.nRanks;
  this->fenceAllContexts = true;
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclGinBarrierSession<Coop>::ncclGinBarrierSession(
    Coop coop, ncclGinAllContexts allCtx, ncclTeamTagRail, uint32_t barrierIndex
  ):
  ncclGinBarrierSession(coop, allCtx, ncclTeamRail(allCtx.comm), allCtx.comm.railGinBarrier, barrierIndex) {
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclGinBarrierSession<Coop>::ncclGinBarrierSession(
    Coop coop, ncclGinAllContexts allCtx, ncclTeamTagWorld, uint32_t barrierIndex
  ):
  ncclGinBarrierSession(coop, allCtx, ncclTeamWorld(allCtx.comm), allCtx.comm.worldGinBarrier, barrierIndex) {
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclGinBarrierSession<Coop>::~ncclGinBarrierSession() {
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
template<bool EnableTimeout>
NCCL_DEVICE_INLINE ncclResult_t ncclGinBarrierSession_internal<Coop>::syncInternal(Coop, cuda::memory_order ord,
                                                                      ncclGinFenceLevel fence, uint64_t timeoutCycles) {
  uint64_t startCycle;
  ncclResult_t ret = ncclSuccess;
  this->coop.sync();
  // For fence containing Put: drain prior outgoing puts on this context before signaling so
  // peers, upon observing our signal, see our outgoing data already settled at their
  // destinations. When the session was constructed from `ncclGinAllContexts(comm)` the
  // fence iterates every GIN context; otherwise it's a one-shot flush on the bound context.
  // The release-order argument is passed through to the proxy backend; GDAKI ignores it
  // (DOCA already guarantees memory_order_acquire on flush completion).
  if ((fence & ncclGinFenceLevel::Put) != ncclGinFenceLevel::None) {
    if (this->fenceAllContexts) {
      for (uint32_t i = 0; i < this->net.comm.ginContextCount; ++i) {
        ncclGin scratch(this->net.comm, (int)i, this->net.resourceSharingMode);
        scratch.flush(this->coop, nccl::utility::releaseOrderOf(ord));
      }
    } else {
      this->net.flush(this->coop, nccl::utility::releaseOrderOf(ord));
    }
  }
  if NCCL_IF_CONSTEXPR (EnableTimeout) {
    startCycle = clock64();
  }
  #pragma unroll 1
  for (int i=this->coop.thread_rank(); i < this->team.nRanks-1; i += this->coop.size()) {
    // Use a rotating pattern to avoid hot spots
    int peer = 1 + this->team.rank + i;
    if (this->team.nRanks <= peer) peer -= this->team.nRanks;

    // Initiate signal
    this->net.signal(
      this->team, peer, ncclGin_SignalInc{this->signal + this->team.rank}, ncclCoopThread(), ncclGin_None(),
      nccl::utility::releaseOrderOf(ord) != cuda::memory_order_relaxed
        ? cuda::thread_scope_thread
        : cuda::thread_scope_system
    );

    // Load and update barrier state in memory. The load/store should be covered by the GIN signal latency.
    uint32_t* shadowPtr = (uint32_t*)this->net.getSignalShadowPtr(this->signal + peer);
    int waitVal = ++*shadowPtr;

    if NCCL_IF_CONSTEXPR (EnableTimeout) {
      while (true) {
        uint64_t got = this->net.readSignal(this->signal + peer, 32, nccl::utility::acquireOrderOf(ord));
        if (nccl::utility::rollingLessEq(static_cast<uint64_t>(waitVal), got, 32)) break;
        if (clock64() - startCycle >= timeoutCycles) {
          ret = ncclTimeout;
          goto exit;
        }
      }
    } else {
      this->net.waitSignal(ncclCoopThread(), this->signal + peer, waitVal, 32, nccl::utility::acquireOrderOf(ord));
    }
  }
  // For fence containing Get: drain prior outgoing gets on this context after waiting so that,
  // on successful barrier exit, all RDMA-Read responses targeting this rank's local buffers
  // have been DMA'd into GPU memory and are visible after the trailing coop.sync(). When the
  // session was constructed from `ncclGinAllContexts(comm)` the fence iterates every GIN
  // context; otherwise it's a one-shot flush. Skipped on the timeout path (control jumps
  // directly to exit: with ret = ncclTimeout).
  if ((fence & ncclGinFenceLevel::Get) != ncclGinFenceLevel::None) {
    if (this->fenceAllContexts) {
      for (uint32_t i = 0; i < this->net.comm.ginContextCount; ++i) {
        ncclGin scratch(this->net.comm, (int)i, this->net.resourceSharingMode);
        scratch.flush(this->coop, nccl::utility::acquireOrderOf(ord));
      }
    } else {
      this->net.flush(this->coop, nccl::utility::acquireOrderOf(ord));
    }
  }
  goto exit; // Silence a compiler warning.
exit:
  this->coop.sync();
  return ret;
}
#endif


#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE void ncclGinBarrierSession<Coop>::sync(Coop coop, cuda::memory_order ord, ncclGinFenceLevel fence) {
  (void)(this->template syncInternal</*EnableTimeout=*/false>(coop, ord, fence, 0ULL));
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclResult_t ncclGinBarrierSession<Coop>::sync(
    Coop coop, cuda::memory_order ord, ncclGinFenceLevel fence, uint64_t timeoutCycles) {
  return this->template syncInternal</*EnableTimeout=*/true>(coop, ord, fence, timeoutCycles);
}
#endif

// Free-function GIN barrier: thin wrappers around session construct + sync + destruct.
#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE void ncclGinBarrier(
    Coop coop, ncclGin gin, ncclTeam team, ncclGinBarrierHandle handle, uint32_t index,
    cuda::memory_order ord, ncclGinFenceLevel fence) {
  ncclGinBarrierSession<Coop> session(coop, gin, team, handle, index);
  session.sync(coop, ord, fence);
}

template<typename Coop>
NCCL_DEVICE_INLINE void ncclGinBarrier(
    Coop coop, ncclGin gin, ncclTeamTagRail tag, uint32_t index,
    cuda::memory_order ord, ncclGinFenceLevel fence) {
  ncclGinBarrierSession<Coop> session(coop, gin, tag, index);
  session.sync(coop, ord, fence);
}

template<typename Coop>
NCCL_DEVICE_INLINE void ncclGinBarrier(
    Coop coop, ncclGin gin, ncclTeamTagWorld tag, uint32_t index,
    cuda::memory_order ord, ncclGinFenceLevel fence) {
  ncclGinBarrierSession<Coop> session(coop, gin, tag, index);
  session.sync(coop, ord, fence);
}

template<typename Coop>
NCCL_DEVICE_INLINE void ncclGinBarrier(
    Coop coop, ncclGinAllContexts allCtx, ncclTeam team, ncclGinBarrierHandle handle,
    uint32_t index, cuda::memory_order ord, ncclGinFenceLevel fence) {
  ncclGinBarrierSession<Coop> session(coop, allCtx, team, handle, index);
  session.sync(coop, ord, fence);
}

template<typename Coop>
NCCL_DEVICE_INLINE void ncclGinBarrier(
    Coop coop, ncclGinAllContexts allCtx, ncclTeamTagRail tag, uint32_t index,
    cuda::memory_order ord, ncclGinFenceLevel fence) {
  ncclGinBarrierSession<Coop> session(coop, allCtx, tag, index);
  session.sync(coop, ord, fence);
}

template<typename Coop>
NCCL_DEVICE_INLINE void ncclGinBarrier(
    Coop coop, ncclGinAllContexts allCtx, ncclTeamTagWorld tag, uint32_t index,
    cuda::memory_order ord, ncclGinFenceLevel fence) {
  ncclGinBarrierSession<Coop> session(coop, allCtx, tag, index);
  session.sync(coop, ord, fence);
}
#endif

#endif // _NCCL_DEVICE_GIN_BARRIER__FUNCS_H_
