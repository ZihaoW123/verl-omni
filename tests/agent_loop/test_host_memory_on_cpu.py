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
