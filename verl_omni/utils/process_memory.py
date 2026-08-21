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

import ctypes
import gc
import os
import resource
import sys


def process_memory_breakdown_bytes() -> dict[str, int]:
    """Return Linux smaps_rollup counters in bytes when available."""
    counters: dict[str, int] = {}
    try:
        with open("/proc/self/smaps_rollup") as smaps:
            for line in smaps:
                key, separator, remainder = line.partition(":")
                if not separator:
                    continue
                parts = remainder.split()
                if parts and parts[0].isdigit():
                    counters[key] = int(parts[0]) * 1024
    except OSError:
        pass
    return counters


def npu_host_memory_stats_bytes() -> dict[str, int]:
    """Return torch_npu pinned-host allocator byte counters when supported."""
    try:
        import torch_npu

        stats = torch_npu.npu.host_memory_stats()
    except (AttributeError, ImportError, RuntimeError):
        return {}
    return {
        key: int(value)
        for key, value in stats.items()
        if (key.startswith("allocated_bytes.") or key.startswith("active_bytes.")) and isinstance(value, int | float)
    }


def current_process_rss_bytes() -> int:
    """Return the current process RSS, falling back to peak RSS off Linux."""
    try:
        with open("/proc/self/statm") as statm:
            resident_pages = int(statm.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (FileNotFoundError, IndexError, OSError, ValueError):
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss is bytes on macOS and KiB on Linux/BSD.
        return int(peak_rss if sys.platform == "darwin" else peak_rss * 1024)


def collect_and_trim_process_memory() -> bool:
    """Release Python, torch_npu pinned-host, and glibc memory caches."""
    gc.collect()
    try:
        import torch_npu

        torch_npu.npu.host_empty_cache()
    except (AttributeError, ImportError, RuntimeError):
        pass
    if not sys.platform.startswith("linux"):
        return False
    try:
        libc = ctypes.CDLL(None)
        malloc_trim = libc.malloc_trim
        malloc_trim.argtypes = [ctypes.c_size_t]
        malloc_trim.restype = ctypes.c_int
        return bool(malloc_trim(0))
    except (AttributeError, OSError):
        return False
