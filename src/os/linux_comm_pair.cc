/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "os_comm_pair.h"
#include "checks.h"

#include <unistd.h>

ncclResult_t ncclOsCommPairCreate(ncclCommPairDescriptor pair[2]) {
  int fds[2];
  SYSCHECK(pipe(fds), "pipe");
  pair[0] = fds[0];
  pair[1] = fds[1];
  return ncclSuccess;
}

ncclResult_t ncclOsCommPairClose(ncclCommPairDescriptor pair[2]) {
  for (int i = 0; i < 2; i++) {
    if (pair[i] != NCCL_COMM_PAIR_INVALID) {
      SYSCHECK(close(pair[i]), "close");
      pair[i] = NCCL_COMM_PAIR_INVALID;
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclOsCommPairWrite(ncclCommPairDescriptor descriptor, const void* buf, size_t len, size_t* written) {
  ssize_t n;
  SYSCHECK(n = write(descriptor, buf, len), "write");
  *written = (size_t)n;
  return ncclSuccess;
}

ncclResult_t ncclOsCommPairRead(ncclCommPairDescriptor descriptor, void* buf, size_t len, size_t* nread) {
  ssize_t n;
  SYSCHECK(n = read(descriptor, buf, len), "read");
  *nread = (size_t)n;
  return ncclSuccess;
}
