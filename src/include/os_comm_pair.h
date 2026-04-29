/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_OS_COMM_PAIR_H_
#define NCCL_OS_COMM_PAIR_H_

#include "nccl.h"
#include "os.h"
#include <cstddef>

// Platform-agnostic descriptor for communication endpoints
// On Linux: file descriptor (int)
// On Windows: socket (SOCKET)
typedef ncclSocketDescriptor ncclCommPairDescriptor;

// Invalid descriptor constant
#define NCCL_COMM_PAIR_INVALID NCCL_INVALID_SOCKET

// Creates a communication pair: two connected endpoints for data transfer
// pair[0] is intended for reading, pair[1] for writing (following pipe() convention)
// Either endpoint can technically perform both operations, but typical usage is unidirectional:
// one side writes to pair[1], the other side reads from pair[0]
// Returns ncclSuccess on success, error code on failure
ncclResult_t ncclOsCommPairCreate(ncclCommPairDescriptor pair[2]);

// Closes both endpoints of a communication pair
// Skips any descriptor that is already NCCL_COMM_PAIR_INVALID
// Resets both descriptors to NCCL_COMM_PAIR_INVALID after closing
// Returns ncclSuccess on success, error code on failure
ncclResult_t ncclOsCommPairClose(ncclCommPairDescriptor pair[2]);

// Writes data to the communication pair
// Returns ncclSuccess on success, error code on failure
// On success, *written contains the number of bytes written (may be less than len)
// Callers must loop to ensure all data is written
ncclResult_t ncclOsCommPairWrite(ncclCommPairDescriptor descriptor, const void* buf, size_t len, size_t* written);

// Reads data from the communication pair
// Returns ncclSuccess on success, error code on failure
// On success, *nread contains the number of bytes read (may be less than len; 0 indicates EOF)
// Callers must loop to ensure all expected data is read
ncclResult_t ncclOsCommPairRead(ncclCommPairDescriptor descriptor, void* buf, size_t len, size_t* nread);

#endif // NCCL_OS_COMM_PAIR_H_
