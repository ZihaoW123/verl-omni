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

from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from verl.protocol import DataProto

import verl_omni.agent_loop.diffusion_agent_loop as agent_loop_module
from verl_omni.agent_loop.diffusion_agent_loop import DiffusionAgentLoopManager, DiffusionAgentLoopWorker
from verl_omni.trainer.diffusion import ray_diffusion_trainer as trainer_module
from verl_omni.trainer.diffusion.ray_diffusion_trainer import PolicyGradientRayTrainer
from verl_omni.utils.process_memory import (
    collect_and_trim_process_memory,
    current_process_rss_bytes,
    npu_host_memory_stats_bytes,
    process_memory_breakdown_bytes,
)
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniHttpServer


class _FakeBatch:
    def __init__(self):
        self.batch = [object()]
        self.non_tensor_batch = {}
        self.meta_info = {}

    def __len__(self):
        return len(self.batch)

    def repeat(self, repeat_times, interleave):
        return self


class _FakeOutput:
    meta_info = {"timing": {}}

    def __len__(self):
        return 1


class _FakeRolloutManager:
    def __init__(self):
        self.calls = 0

    def generate_sequences(self, rollout_input):
        self.calls += 1
        return _FakeOutput()


class _FailOnSleepCheckpointManager:
    def sleep_replicas(self):
        raise AssertionError("rollout-only mode must keep replicas awake between steps")


class _FakeLogger:
    def log(self, data, step):
        pass


class _FakeMetrics:
    def model_dump(self):
        return {"generate_sequences": 1.0, "tool_calls": 0.0, "compute_score": 0.0, "num_preempted": -1}


class _ImmediateObjectRef:
    def __init__(self, value):
        self.value = value

    def __await__(self):
        async def resolve():
            return self.value

        return resolve().__await__()


class _RemoteGenerate:
    def __init__(self, output):
        self.output = output
        self.refs = []

    def remote(self, _chunk):
        ref = _ImmediateObjectRef(self.output)
        self.refs.append(ref)
        return ref


class _FakeAgentLoopWorker:
    def __init__(self, output):
        self.generate_sequences = _RemoteGenerate(output)


def _agent_loop_chunk(prompt_id):
    metrics = [
        {
            "generate_sequences": 1.0,
            "tool_calls": 0.0,
            "compute_score": 0.0,
            "num_preempted": -1,
        }
    ]
    metadata = np.empty(1, dtype=object)
    metadata[0] = np.array([prompt_id])
    return DataProto(
        batch=TensorDict({"prompts": torch.tensor([[prompt_id]])}, batch_size=1),
        non_tensor_batch={"metadata": metadata},
        meta_info={"metrics": metrics},
    )


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
    assert all(value >= 0 for value in process_memory_breakdown_bytes().values())
    assert all(value >= 0 for value in npu_host_memory_stats_bytes().values())


def test_diffusion_agent_loop_manager_frees_worker_ray_outputs(monkeypatch):
    freed_refs = []

    def fake_free(refs, *, local_only=False):
        freed_refs.append((tuple(refs), local_only))

    monkeypatch.setattr(agent_loop_module, "free_ray_object_refs", fake_free)

    manager = DiffusionAgentLoopManager.__new__(DiffusionAgentLoopManager)
    workers = [_FakeAgentLoopWorker(_agent_loop_chunk(1)), _FakeAgentLoopWorker(_agent_loop_chunk(2))]
    manager.agent_loop_workers = workers
    prompts = DataProto(
        batch=TensorDict({"prompts": torch.tensor([[1], [2]])}, batch_size=2),
        non_tensor_batch={},
        meta_info={},
    )

    output = manager.generate_sequences(prompts)

    refs = tuple(worker.generate_sequences.refs[0] for worker in workers)
    assert freed_refs == [(refs, False)]
    assert output.batch["prompts"].tolist() == [[1], [2]]
    workers[0].generate_sequences.output.non_tensor_batch["metadata"][0][0] = 99
    assert output.non_tensor_batch["metadata"][0].tolist() == [1]


def test_rollout_only_postprocess_drops_large_training_payloads():
    worker = DiffusionAgentLoopWorker.__new__(DiffusionAgentLoopWorker)
    inputs = [
        SimpleNamespace(
            prompt_ids=torch.tensor([[1, 2]]),
            metrics=_FakeMetrics(),
            response_diffusion_output=torch.empty((1, 8, 3, 64, 64), dtype=torch.uint8),
            response_logprobs=torch.empty((1, 2)),
            extra_fields={"all_latents": torch.empty((1, 3, 4, 8, 8))},
        ),
        SimpleNamespace(
            prompt_ids=torch.tensor([[3, 4]]),
            metrics=_FakeMetrics(),
            response_diffusion_output=torch.empty((1, 8, 3, 64, 64), dtype=torch.uint8),
            response_logprobs=torch.empty((1, 2)),
            extra_fields={"all_latents": torch.empty((1, 3, 4, 8, 8))},
        ),
    ]

    output = worker._postprocess_rollout_only(inputs)

    assert set(output.batch.keys()) == {"prompts"}
    assert output.batch["prompts"].tolist() == [[1, 2], [3, 4]]
    assert len(output.meta_info["metrics"]) == 2


def test_rollout_server_drops_payload_before_ray_transport():
    server = SimpleNamespace(_ar_mode=False, global_steps=3)
    final_res = SimpleNamespace(images=[object()])

    output = vLLMOmniHttpServer._process_output(
        server,
        final_res,
        params=None,
        sampling_params={"_verl_rollout_only_drop_outputs": True},
    )

    assert output.diffusion_output.dtype == torch.uint8
    assert output.diffusion_output.numel() == 0
    assert output.log_probs is None
    assert output.extra_fields == {"global_steps": 3}


def test_rollout_only_keeps_replicas_awake_across_steps(monkeypatch):
    class FakeDataProto:
        @staticmethod
        def from_single_dict(batch_dict):
            return _FakeBatch()

    monkeypatch.setattr(trainer_module, "DataProto", FakeDataProto)

    trainer = PolicyGradientRayTrainer.__new__(PolicyGradientRayTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {"rollout_only_memory_trim": False},
            "actor_rollout_ref": {"rollout": {"n": 1, "seed": None}},
        }
    )
    trainer.global_steps = 0
    trainer.total_training_steps = 2
    trainer.train_dataloader = [{"prompt": "test"}]
    trainer.actor_rollout_wg = object()
    trainer.async_rollout_manager = _FakeRolloutManager()
    trainer.checkpoint_manager = _FailOnSleepCheckpointManager()
    trainer._get_gen_batch = lambda batch: batch

    trainer._fit_rollout_only(_FakeLogger())

    assert trainer.async_rollout_manager.calls == 2
