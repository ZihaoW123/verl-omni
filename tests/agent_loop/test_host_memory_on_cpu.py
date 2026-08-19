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

from verl_omni.agent_loop import host_memory


def test_trim_host_memory_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("VERL_OMNI_AGENT_MALLOC_TRIM", raising=False)
    monkeypatch.setattr(host_memory.gc, "collect", lambda: (_ for _ in ()).throw(AssertionError("unexpected gc")))

    assert host_memory.trim_host_memory() is False


def test_trim_host_memory_collects_and_calls_glibc(monkeypatch):
    calls = []
    monkeypatch.setenv("VERL_OMNI_AGENT_MALLOC_TRIM", "1")
    monkeypatch.setattr(host_memory.sys, "platform", "linux")
    monkeypatch.setattr(host_memory.gc, "collect", lambda: calls.append("gc"))
    monkeypatch.setattr(host_memory, "_load_malloc_trim", lambda: lambda padding: calls.append(("trim", padding)) or 1)

    assert host_memory.trim_host_memory() is True
    assert calls == ["gc", ("trim", 0)]


def test_trim_host_memory_is_portable(monkeypatch):
    monkeypatch.setenv("VERL_OMNI_AGENT_MALLOC_TRIM", "true")
    monkeypatch.setattr(host_memory.sys, "platform", "darwin")
    monkeypatch.setattr(host_memory.gc, "collect", lambda: None)

    assert host_memory.trim_host_memory() is False


def test_mapping_category_separates_plasma_from_files():
    assert host_memory._mapping_category("/dev/shm/plasmaAbCd") == "plasma"
    assert host_memory._mapping_category("/dev/shm/other") == "shmem"
    assert host_memory._mapping_category("/usr/lib/libc.so.6") == "file"


def test_diagnostics_distinguish_python_glibc_and_anonymous_memory(monkeypatch, capsys):
    snapshots = iter(
        [
            {"rss_kib": 200 * 1024, "private_dirty_kib": 180 * 1024},
            {
                "rss_kib": 150 * 1024,
                "rss_anon_kib": 140 * 1024,
                "private_dirty_kib": 130 * 1024,
                "anon_private_dirty_kib": 100 * 1024,
                "glibc_inuse_bytes": 80 * 1024 * 1024,
            },
        ]
    )
    monkeypatch.setenv("VERL_OMNI_AGENT_MEMORY_DIAGNOSTICS", "1")
    monkeypatch.setattr(host_memory, "trim_host_memory", lambda: True)
    monkeypatch.setattr(
        host_memory,
        "_process_memory_snapshot",
        lambda *, include_mapping_categories: next(snapshots),
    )
    monkeypatch.setattr(
        host_memory,
        "_live_python_stats",
        lambda: {
            "cpu_tensor_objects": 4,
            "cpu_tensor_storage_bytes": 64 * 1024 * 1024,
            "asyncio_tasks": 2,
        },
    )

    assert host_memory.trim_and_log_host_memory("return_ready", 7, 32 * 1024 * 1024) is True

    message = capsys.readouterr().out
    assert host_memory._DEBUG_TAG in message
    assert '"global_step": 7' in message
    assert '"rss_before_mib": 200.0' in message
    assert '"rss_after_mib": 150.0' in message
    assert '"anon_private_dirty_mib": 100.0' in message
    assert '"glibc_inuse_mib": 80.0' in message
    assert '"cpu_tensor_storage_mib": 64.0' in message
    assert '"payload_mib": 32.0' in message
