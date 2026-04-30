# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

"""NCCL constants and Python-only enums.

This module centralizes Python-only NCCL constants and flag enums that wrap
#define constants from the NCCL C headers. C-defined enums are re-exported
from nccl.core.typing.
"""

from enum import IntEnum, IntFlag

__all__ = [
    "NCCL_UNDEF_INT",
    "NCCL_UNDEF_FLOAT",
    "NCCL_SPLIT_NOCOLOR",
    "CTAPolicy",
    "CommShrinkFlag",
    "CommSuspendFlag",
    "WindowFlag",
]

# NCCL sentinel values for undefined config fields
NCCL_UNDEF_INT: int = -2147483648  # INT_MIN
"""NCCL sentinel value for undefined integer configuration fields."""

NCCL_UNDEF_FLOAT: float = -1.0
"""NCCL sentinel value for undefined float fields."""

# Communicator split constants
NCCL_SPLIT_NOCOLOR: int = -1
"""Color value for ncclCommSplit to indicate the rank will not be part of any group."""

# NCCL magic number
NCCL_MAGIC: int = 0xCAFEBEEF
"""Magic number for NCCL configuration structs (used for ABI validation)."""


# CTA (Cooperative Thread Array) Policy flags
class CTAPolicy(IntFlag):
    """NCCL performance policy for CTA scheduling.

    Policies can be combined with the bitwise OR operator so that NCCL can
    use different scheduling in different cases.
    """

    Default = 0x00
    """Default CTA policy."""
    Efficiency = 0x01
    """Optimize for efficiency."""
    Zero = 0x02
    """Zero-CTA optimization."""


# Communicator shrink flags
class CommShrinkFlag(IntEnum):
    """Flags for ncclCommShrink behavior."""

    Default = 0x00
    """Shrink the parent communicator."""
    Abort = 0x01
    """First terminate ongoing parent operations, then shrink the parent communicator."""


# Communicator suspend flags
class CommSuspendFlag(IntFlag):
    """Flags for ncclCommSuspend behavior."""

    Mem = 0x01
    """Suspend memory (release dynamic GPU memory allocations)."""


# Window registration flags
class WindowFlag(IntFlag):
    """Flags for window registration."""

    Default = 0x00
    """Default window registration."""
    CollSymmetric = 0x01
    """Collective symmetric window registration."""
    StrictOrdering = 0x02
    """Strict ordering for window operations."""
