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

from omegaconf import OmegaConf

from verl_omni.trainer.diffusion.ray_diffusion_trainer import PolicyGradientRayTrainer
from verl_omni.utils.process_memory import collect_and_trim_process_memory, current_process_rss_bytes


def test_rollout_only_does_not_initialize_reward_loop():
    trainer = PolicyGradientRayTrainer.__new__(PolicyGradientRayTrainer)
    trainer.config = OmegaConf.create({"trainer": {"rollout_only": True}})

    reward_worker_handles = trainer._init_reward_loop()

    assert reward_worker_handles is None
    assert trainer.reward_loop_manager is None
    assert trainer.enable_agent_reward_loop is False


def test_process_rss_probe_returns_positive_bytes():
    collect_and_trim_process_memory()
    assert current_process_rss_bytes() > 0
