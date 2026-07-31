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

"""CPU tests for diffusion rollout output shapes."""

import importlib.util
from pathlib import Path

import torch


def _load_rollout_replica_module():
    module_path = Path(__file__).parents[3] / "verl_omni/workers/rollout/replica.py"
    spec = importlib.util.spec_from_file_location("rollout_replica", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_unbatch_single_request_video_output():
    replica = _load_rollout_replica_module()

    video = replica.unbatch_single_request_video(torch.zeros(1, 81, 3, 8, 10))

    assert video.shape == (81, 3, 8, 10)


def test_unbatch_preserves_tchw_output():
    replica = _load_rollout_replica_module()

    video = replica.unbatch_single_request_video(torch.zeros(1, 3, 8, 10))

    assert video.shape == (1, 3, 8, 10)
