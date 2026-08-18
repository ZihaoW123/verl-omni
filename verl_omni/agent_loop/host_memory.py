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

"""Diagnose and release host memory retained by diffusion agent workers."""

import asyncio
import ctypes
import gc
import json
import os
import re
import sys
from functools import lru_cache
from typing import Callable

import torch

_ENABLED_VALUES = {"1", "true", "yes", "on"}
_DEBUG_TAG = "[DEBUG-A3-HOSTMEM]"
_MAP_HEADER = re.compile(r"^[0-9a-f]+-[0-9a-f]+\s")
_MIB = 1024 * 1024


class _Mallinfo2(ctypes.Structure):
    _fields_ = [
        ("arena", ctypes.c_size_t),
        ("ordblks", ctypes.c_size_t),
        ("smblks", ctypes.c_size_t),
        ("hblks", ctypes.c_size_t),
        ("hblkhd", ctypes.c_size_t),
        ("usmblks", ctypes.c_size_t),
        ("fsmblks", ctypes.c_size_t),
        ("uordblks", ctypes.c_size_t),
        ("fordblks", ctypes.c_size_t),
        ("keepcost", ctypes.c_size_t),
    ]


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


@lru_cache(maxsize=1)
def _load_mallinfo2() -> Callable[[], _Mallinfo2] | None:
    """Load glibc's 64-bit allocator counters when available."""
    try:
        mallinfo2 = ctypes.CDLL(None).mallinfo2
    except (AttributeError, OSError):
        return None
    mallinfo2.argtypes = []
    mallinfo2.restype = _Mallinfo2
    return mallinfo2


def _read_proc_kib(path: str, wanted: set[str]) -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        with open(path) as proc_file:
            for line in proc_file:
                key, separator, remainder = line.partition(":")
                if separator and key in wanted:
                    values[key] = int(remainder.split()[0])
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    return values


def _mapping_category(pathname: str) -> str:
    if pathname == "[heap]":
        return "heap"
    if not pathname:
        return "anon"
    if pathname.startswith("/"):
        return "file"
    return "other"


def _read_smaps_categories() -> dict[str, int]:
    totals = {
        f"{category}_{metric}_kib": 0
        for category in ("heap", "anon", "file", "other")
        for metric in ("rss", "private_dirty")
    }
    category = "other"
    try:
        with open("/proc/self/smaps") as smaps:
            for line in smaps:
                if _MAP_HEADER.match(line):
                    fields = line.split(maxsplit=5)
                    category = _mapping_category(fields[5].strip() if len(fields) == 6 else "")
                elif line.startswith("Rss:"):
                    totals[f"{category}_rss_kib"] += int(line.split()[1])
                elif line.startswith("Private_Dirty:"):
                    totals[f"{category}_private_dirty_kib"] += int(line.split()[1])
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    return totals


def _allocator_stats() -> dict[str, int]:
    if sys.platform != "linux":
        return {}
    mallinfo2 = _load_mallinfo2()
    if mallinfo2 is None:
        return {}
    info = mallinfo2()
    return {
        "glibc_arena_bytes": info.arena,
        "glibc_inuse_bytes": info.uordblks,
        "glibc_free_bytes": info.fordblks,
        "glibc_mmap_bytes": info.hblkhd,
    }


def _process_memory_snapshot(*, include_mapping_categories: bool) -> dict[str, int]:
    status = _read_proc_kib(
        "/proc/self/status",
        {"VmRSS", "VmData", "VmLck", "RssAnon", "RssFile"},
    )
    rollup = _read_proc_kib("/proc/self/smaps_rollup", {"Private_Dirty", "Locked"})
    snapshot = {
        "rss_kib": status.get("VmRSS", 0),
        "rss_anon_kib": status.get("RssAnon", 0),
        "rss_file_kib": status.get("RssFile", 0),
        "vm_data_kib": status.get("VmData", 0),
        "vm_locked_kib": status.get("VmLck", 0),
        "private_dirty_kib": rollup.get("Private_Dirty", 0),
        "locked_kib": rollup.get("Locked", 0),
        **_allocator_stats(),
    }
    if include_mapping_categories:
        snapshot.update(_read_smaps_categories())
    return snapshot


def _live_python_stats() -> dict[str, int]:
    cpu_tensor_objects = 0
    cpu_tensor_logical_bytes = 0
    cpu_storages: dict[tuple[int, int], int] = {}
    for obj in gc.get_objects():
        try:
            # Some PyTorch sentinel objects implement a deprecated custom
            # ``isinstance`` path.  Checking their concrete type avoids both
            # that warning and arbitrary user-level ``__class__`` behavior.
            if not issubclass(type(obj), torch.Tensor) or obj.device.type != "cpu":
                continue
            cpu_tensor_objects += 1
            cpu_tensor_logical_bytes += obj.numel() * obj.element_size()
            storage = obj.untyped_storage()
            storage_bytes = storage.nbytes()
            cpu_storages[(storage.data_ptr(), storage_bytes)] = storage_bytes
        except (AttributeError, RuntimeError, TypeError):
            continue

    try:
        tasks = asyncio.all_tasks()
        asyncio_tasks = len(tasks)
        asyncio_pending_tasks = sum(not task.done() for task in tasks)
    except RuntimeError:
        asyncio_tasks = 0
        asyncio_pending_tasks = 0

    return {
        "cpu_tensor_objects": cpu_tensor_objects,
        "cpu_tensor_logical_bytes": cpu_tensor_logical_bytes,
        "cpu_tensor_storage_bytes": sum(cpu_storages.values()),
        "cpu_tensor_storages": len(cpu_storages),
        "asyncio_tasks": asyncio_tasks,
        "asyncio_pending_tasks": asyncio_pending_tasks,
    }


def _as_mib(value: int) -> float:
    return round(value / _MIB, 1)


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


def trim_and_log_host_memory(phase: str, global_step: object, payload_bytes: int = 0) -> bool:
    """Trim host memory and emit one diagnostic record when opted in."""
    diagnostics_enabled = os.getenv("VERL_OMNI_AGENT_MEMORY_DIAGNOSTICS", "").strip().lower() in _ENABLED_VALUES
    if not diagnostics_enabled:
        return trim_host_memory()

    before = _process_memory_snapshot(include_mapping_categories=False)
    trim_released = trim_host_memory()
    after = _process_memory_snapshot(include_mapping_categories=True)
    live = _live_python_stats()

    record: dict[str, object] = {
        "phase": phase,
        "global_step": global_step,
        "pid": os.getpid(),
        "trim_released": trim_released,
        "payload_mib": _as_mib(payload_bytes),
        "rss_before_mib": _as_mib(before.get("rss_kib", 0) * 1024),
        "rss_after_mib": _as_mib(after.get("rss_kib", 0) * 1024),
        "private_dirty_before_mib": _as_mib(before.get("private_dirty_kib", 0) * 1024),
        "private_dirty_after_mib": _as_mib(after.get("private_dirty_kib", 0) * 1024),
    }
    for key, value in after.items():
        if key.endswith("_bytes") or key.endswith("_kib"):
            scale = 1 if key.endswith("_bytes") else 1024
            record[f"{key.removesuffix('_bytes').removesuffix('_kib')}_mib"] = _as_mib(value * scale)
        else:
            record[key] = value
    for key, value in live.items():
        if key.endswith("_bytes"):
            record[f"{key.removesuffix('_bytes')}_mib"] = _as_mib(value)
        else:
            record[key] = value

    # Ray's worker logging configuration can filter module loggers. stdout is
    # captured reliably and flush=True preserves the step boundary ordering.
    print(f"{_DEBUG_TAG} {json.dumps(record, sort_keys=True)}", flush=True)
    return trim_released
