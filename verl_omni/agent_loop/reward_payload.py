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

import asyncio
from collections.abc import Iterable, Mapping
from typing import Any, Awaitable, Callable, TypeVar

_T = TypeVar("_T")


def select_tool_extra_fields(extra_fields: Mapping[str, Any], allowed_keys: Iterable[str] | None) -> dict[str, Any]:
    """Select rollout fields that a reward function explicitly consumes.

    ``None`` preserves the legacy behavior. An empty iterable disables forwarding
    rollout extras, which prevents training-only tensors from being serialized in
    every per-sample reward RPC.
    """
    if allowed_keys is None:
        return dict(extra_fields)

    allowed = set(allowed_keys)
    return {key: value for key, value in extra_fields.items() if key in allowed}


def create_reward_semaphore(max_inflight: int | None) -> asyncio.Semaphore | None:
    """Create an optional per-agent limit for in-flight reward RPCs."""
    if max_inflight is None:
        return None
    if max_inflight <= 0:
        raise ValueError(f"reward.max_inflight_per_agent must be positive, got {max_inflight}")
    return asyncio.Semaphore(max_inflight)


async def run_limited_reward_request(
    semaphore: asyncio.Semaphore | None, request_factory: Callable[[], Awaitable[_T]]
) -> _T:
    """Start a reward RPC only after an optional per-agent slot is available."""
    if semaphore is None:
        return await request_factory()
    async with semaphore:
        return await request_factory()
