# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Release free glibc heap pages retained by diffusion agent workers."""

import ctypes
import gc
import os
import sys
from functools import lru_cache
from typing import Callable

_ENABLED_VALUES = {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def _load_malloc_trim() -> Callable[[int], int] | None:
    """Load glibc's malloc_trim when it is available."""
    try:
        malloc_trim = ctypes.CDLL(None).malloc_trim
    except (AttributeError, OSError):
        return None
    malloc_trim.argtypes = [ctypes.c_size_t]
    malloc_trim.restype = ctypes.c_int
    return malloc_trim


def trim_host_memory() -> bool:
    """Collect dead objects and return free glibc arenas to Linux.

    This is opt-in because walking the Python heap and trimming glibc arenas has
    a small step-boundary latency cost.  It is intended for large diffusion
    rollout batches whose short-lived CPU tensors otherwise leave RSS at the
    allocator high-water mark.
    """
    enabled = os.getenv("VERL_OMNI_AGENT_MALLOC_TRIM", "").strip().lower() in _ENABLED_VALUES
    if not enabled:
        return False

    gc.collect()
    if sys.platform != "linux":
        return False

    malloc_trim = _load_malloc_trim()
    if malloc_trim is None:
        return False
    return bool(malloc_trim(0))
