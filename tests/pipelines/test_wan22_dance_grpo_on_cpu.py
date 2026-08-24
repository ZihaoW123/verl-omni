# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import pytest
import torch
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from verl_omni.pipelines.wan22_dance_grpo.vllm_omni_rollout_adapter import (
    _unwrap_single_request,
    _validate_wan_rollout_inputs,
)


def test_wan22_dance_grpo_unwraps_vllm_omni_request_batch() -> None:
    request = SimpleNamespace(prompt={}, sampling_params=SimpleNamespace())
    request_batch = DiffusionRequestBatch(requests=[request])

    assert _unwrap_single_request(request_batch) is request
    assert _unwrap_single_request(request) is request


def test_wan22_dance_grpo_rejects_packed_request_batch() -> None:
    request_batch = DiffusionRequestBatch(requests=[SimpleNamespace(), SimpleNamespace()])

    with pytest.raises(ValueError, match="expects one request, got 2"):
        _unwrap_single_request(request_batch)


def test_wan22_dance_grpo_validates_token_prompt_contract() -> None:
    _validate_wan_rollout_inputs(
        prompt_ids=torch.tensor([1, 2, 3]),
        negative_prompt_ids=None,
        height=704,
        width=1280,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    )

    with pytest.raises(ValueError, match="both `prompt_ids` and `prompt_embeds`"):
        _validate_wan_rollout_inputs(
            prompt_ids=torch.tensor([1, 2, 3]),
            negative_prompt_ids=None,
            height=704,
            width=1280,
            prompt_embeds=torch.randn(1, 3, 8),
            negative_prompt_embeds=None,
        )
