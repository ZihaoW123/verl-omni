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
import hashlib


def make_update_zmq_handle(base_handle: str, update_id: int | str | None) -> str:
    """Build a short per-rank IPC handle for one weight update."""
    if not base_handle.startswith("ipc://") or update_id is None:
        return base_handle

    identity = f"{base_handle}\0{update_id}".encode()
    digest = hashlib.blake2s(identity, digest_size=12).hexdigest()
    return f"ipc:///tmp/verl-weights-{digest}.sock"
