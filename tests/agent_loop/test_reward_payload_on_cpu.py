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
import importlib.util
from pathlib import Path

import pytest
import torch

_MODULE_PATH = Path(__file__).parents[2] / "verl_omni" / "agent_loop" / "reward_payload.py"
_SPEC = importlib.util.spec_from_file_location("reward_payload", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
select_tool_extra_fields = _MODULE.select_tool_extra_fields
create_reward_semaphore = _MODULE.create_reward_semaphore
run_limited_reward_request = _MODULE.run_limited_reward_request


def test_select_tool_extra_fields_excludes_training_tensors():
    extra_fields = {
        "all_latents": torch.ones(2, 4, 8, 8),
        "prompt_embeds": torch.ones(16, 32),
        "lightweight_metadata": "keep-me",
    }

    selected = select_tool_extra_fields(extra_fields, ["lightweight_metadata"])

    assert selected == {"lightweight_metadata": "keep-me"}


def test_reward_request_limiter_applies_backpressure():
    async def run_requests():
        semaphore = create_reward_semaphore(1)
        active = 0
        peak_active = 0

        async def request():
            nonlocal active, peak_active
            active += 1
            peak_active = max(peak_active, active)
            await asyncio.sleep(0)
            active -= 1

        await asyncio.gather(*(run_limited_reward_request(semaphore, request) for _ in range(32)))
        return peak_active

    assert asyncio.run(run_requests()) == 1


def test_reward_request_limiter_rejects_non_positive_limit():
    with pytest.raises(ValueError, match="must be positive"):
        create_reward_semaphore(0)
