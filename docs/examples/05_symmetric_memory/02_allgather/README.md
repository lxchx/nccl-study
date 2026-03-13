<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# NCCL Symmetric Memory AllGather Example with Copy Engine

This example demonstrates how to configure NCCL to use the GPU's Copy Engine
instead of SMs for collective operations. By offloading communication to the
Copy Engine, GPU compute resources remain fully available for application
kernels, enabling true overlap of computation and communication.

## Overview

Symmetric memory windows provide a way to register memory buffers that benefit
from optimized collective operations. When using `NCCL_WIN_COLL_SYMMETRIC`, all
ranks must provide symmetric buffers, enabling optimized communication patterns.

By setting `CTAPolicy=2` in the NCCL configuration, the collective operations
use the copy engine instead of SMs, freeing up GPU compute resources for other
work.

## What This Example Does

1. **Configures NCCL with CTAPolicy=2** to enable copy engine for zero SM usage
2. **Allocates memory using NCCL allocator** (`ncclMemAlloc`) which provides
   memory compatible with symmetric windows
3. **Registers buffers as symmetric windows** using `ncclCommWindowRegister`
   with `NCCL_WIN_COLL_SYMMETRIC` flag
4. **Performs AllGather operation** using the symmetric memory with copy engine

## Building and Running

The advanced examples can be built using either pthread or MPI for
parallelization. pthread is the default choice. To use MPI the user needs to set
`MPI=1` at build time and can optionally provide a valid MPI installation under
`MPI_HOME`.

### Build
```shell
make [MPI=1] [MPI_HOME=<path-to-mpi>] [NCCL_HOME=<path-to-nccl>] [CUDA_HOME=<path-to-cuda>]
```

### Run when compiled for pthreads (default)
```shell
[NTHREADS=N] ./allgather_ce
```

### Run when compiled for MPI
```shell
mpirun -np <num_processes> ./allgather_ce
```

## Code Structure

### Key Components

1. **NCCL Configuration for Copy Engine**:
```c
// Configure NCCL to use copy engine (CTAPolicy=2) for zero SM usage
// Alternatively, this can also be done by setting Environment Variable NCCL_CTA_POLICY=ZERO
ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
config.CTAPolicy = 2;

// Initialize communicator with config
NCCLCHECK(ncclCommInitRankConfig(&comm, total_ranks, nccl_unique_id, my_rank, &config));
```

2. **Buffer Allocation and Window Registration**:
```c
size_t sendcount;       // Elements per rank
size_t send_size_bytes; // Size of send buffer
size_t recv_size_bytes; // Size of recv buffer (sendcount * total_ranks * sizeof(type))
void *d_sendbuff;
void *d_recvbuff;

// Allocate buffers using ncclMemAlloc (compatible with symmetric memory)
NCCLCHECK(ncclMemAlloc(&d_sendbuff, send_size_bytes));
NCCLCHECK(ncclMemAlloc(&d_recvbuff, recv_size_bytes));

ncclComm_t comm;
ncclWindow_t send_win;
ncclWindow_t recv_win;

// Register buffers as symmetric windows
NCCLCHECK(ncclCommWindowRegister(comm, d_sendbuff, send_size_bytes, &send_win, NCCL_WIN_COLL_SYMMETRIC));
NCCLCHECK(ncclCommWindowRegister(comm, d_recvbuff, recv_size_bytes, &recv_win, NCCL_WIN_COLL_SYMMETRIC));
```

3. **AllGather Operation**:
```c
cudaStream_t stream; // stream is set in cudaStreamCreate

// Perform AllGather with symmetric memory and copy engine
NCCLCHECK(ncclAllGather(d_sendbuff, d_recvbuff, sendcount, ncclFloat,
                        comm, stream));
```

4. **Window Deregistration and Cleanup**:
```c
// Deregister symmetric memory windows
NCCLCHECK(ncclCommWindowDeregister(comm, send_win));
NCCLCHECK(ncclCommWindowDeregister(comm, recv_win));

// Free buffers allocated with ncclMemAlloc
NCCLCHECK(ncclMemFree(d_sendbuff));
NCCLCHECK(ncclMemFree(d_recvbuff));
```

## Expected Output

### With 4 GPUs (using pthreads/MPI)
```
Starting AllGather example with 4 ranks (Copy Engine enabled)
  Rank 0 communicator initialized using device 0 (CTAPolicy=2)
  Rank 1 communicator initialized using device 1 (CTAPolicy=2)
  Rank 2 communicator initialized using device 2 (CTAPolicy=2)
  Rank 3 communicator initialized using device 3 (CTAPolicy=2)
Symmetric Memory allocation
  Rank 0 allocating 4.00 MB send buffer, 16.00 MB recv buffer
  Rank 1 allocating 4.00 MB send buffer, 16.00 MB recv buffer
  Rank 2 allocating 4.00 MB send buffer, 16.00 MB recv buffer
  Rank 3 allocating 4.00 MB send buffer, 16.00 MB recv buffer
  Rank 0 data initialized (value: 0)
  Rank 1 data initialized (value: 1)
  Rank 2 data initialized (value: 2)
  Rank 3 data initialized (value: 3)
Starting AllGather with 1048576 elements per rank (16 MB total)
AllGather completed successfully
Verification - Segment 0: Expected: 0.0, Got: 0.0
Verification - Segment 1: Expected: 1.0, Got: 1.0
Verification - Segment 2: Expected: 2.0, Got: 2.0
Verification - Segment 3: Expected: 3.0, Got: 3.0
Results verified correctly
  Rank 0 symmetric memory windows deregistered
  Rank 1 symmetric memory windows deregistered
  Rank 2 symmetric memory windows deregistered
  Rank 3 symmetric memory windows deregistered
All resources cleaned up successfully
Example completed - demonstrated symmetric memory + copy engine allgather
```

## Performance Benefits

### Copy Engine
- **Zero SM Usage**: Inside a single (MN)NVL domain, the collective operation uses the copy engine instead of
  SMs , freeing up compute resources
- **Computation Overlap**: Enables true overlap of communication with GPU
  computation kernels
- **Better Performance**: Achieves higher peak bandwidth for large message sizes (higher latency for small message sizes)

For more information on the performance benefits see the [Fusing Communication and Compute with New Device API and Copy Engine Collectives in NVIDIA NCCL 2.28](https://developer.nvidia.com/blog/fusing-communication-and-compute-with-new-device-api-and-copy-engine-collectives-in-nvidia-nccl-2-28/)
blog.

**Important**: Buffers must be allocated using the CUDA Virtual Memory
Management (VMM) API. NCCL provides the `ncclMemAlloc` convenience function for
symmetric memory registration. The `NCCL_WIN_COLL_SYMMETRIC` flag requires all
ranks to provide symmetric buffers consistently.

## Key Insights

- **Copy Engine Mode** is most beneficial for:
  - Applications that need to overlap communication with computation
  - Scenarios where SM resources are at a premium
  - Large-scale collectives that don't have arithmetic operations (AllGather, AlltoAll, Gather, Scatter)
- **ncclCommInitRankConfig** must be used to set the CTAPolicy
- **Window registration** must happen on all ranks for collective operations
- **Memory management** is critical - always deregister windows before freeing
  memory

## Common Issues and Solutions

1. **Window Registration Failure**: Buffers MUST be allocated with (VMM) API,
   e.g. `ncclMemAlloc` (not `cudaMalloc`) for symmetric memory.
2. **Allocation Error**: If `ncclMemAlloc` fails, check NCCL version (requires
   at least v2.27) and available memory
3. **Deregistration Order**: Always deregister windows before freeing memory or
   destroying communicators
4. **Symmetric Requirement**: All ranks must use `NCCL_WIN_COLL_SYMMETRIC`
   consistently in collective operations
5. **Memory Leaks**: Always use `ncclMemFree` for buffers allocated with
   `ncclMemAlloc`
6. **Copy Engine Not Supported**: CTAPolicy=2 only works inside a single (MN)NVL domain
