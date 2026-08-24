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

import asyncio

import numpy as np
import torch
from tensordict import TensorDict
from verl.protocol import DataProto

import verl_omni.agent_loop.diffusion_agent_loop as agent_loop_module
from verl_omni.agent_loop.diffusion_agent_loop import DiffusionAgentLoopManager, DiffusionAgentLoopWorker


class _FakeBatch:
    meta_info = {}


class _ImmediateObjectRef:
    def __init__(self, value):
        self.value = value


class _RemoteGenerateToObjectStore:
    def __init__(self, output):
        self.output = output
        self.payload = agent_loop_module._data_proto_to_numpy_payload(output)
        self.data_refs = []
        self.outer_refs = []

    def remote(self, _chunk):
        data_ref = _ImmediateObjectRef(self.payload)
        outer_ref = _ImmediateObjectRef(data_ref)
        self.data_refs.append(data_ref)
        self.outer_refs.append(outer_ref)
        return outer_ref


class _FakeAgentLoopWorker:
    def __init__(self, output):
        self.generate_sequences_to_object_store = _RemoteGenerateToObjectStore(output)


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


def test_diffusion_agent_loop_manager_frees_worker_ray_outputs(monkeypatch):
    freed_refs = []

    def fake_free(refs, *, local_only=False):
        freed_refs.append((tuple(refs), local_only))

    def fake_get(refs):
        return [ref.value for ref in refs]

    monkeypatch.setattr(agent_loop_module, "free_ray_object_refs", fake_free)
    monkeypatch.setattr(agent_loop_module.ray, "get", fake_get)

    manager = DiffusionAgentLoopManager.__new__(DiffusionAgentLoopManager)
    workers = [_FakeAgentLoopWorker(_agent_loop_chunk(1)), _FakeAgentLoopWorker(_agent_loop_chunk(2))]
    manager.agent_loop_workers = workers
    prompts = DataProto(
        batch=TensorDict({"prompts": torch.tensor([[1], [2]])}, batch_size=2),
        non_tensor_batch={},
        meta_info={},
    )

    output = manager.generate_sequences(prompts)

    data_refs = tuple(worker.generate_sequences_to_object_store.data_refs[0] for worker in workers)
    outer_refs = tuple(worker.generate_sequences_to_object_store.outer_refs[0] for worker in workers)
    assert freed_refs == [(data_refs + outer_refs, False)]
    assert output.batch["prompts"].tolist() == [[1], [2]]
    workers[0].generate_sequences_to_object_store.output.non_tensor_batch["metadata"][0][0] = 99
    assert output.non_tensor_batch["metadata"][0].tolist() == [1]


def test_diffusion_agent_loop_worker_puts_large_output_in_object_store(monkeypatch):
    worker = DiffusionAgentLoopWorker.__new__(DiffusionAgentLoopWorker)
    output = _agent_loop_chunk(7)
    payload_ref = object()
    put_values = []

    async def fake_generate(_batch):
        return output

    def fake_put(value):
        put_values.append(value)
        return payload_ref

    worker.generate_sequences = fake_generate
    monkeypatch.setattr(agent_loop_module.ray, "put", fake_put)

    output_ref = asyncio.run(worker.generate_sequences_to_object_store(_FakeBatch()))

    assert output_ref is payload_ref
    assert len(put_values) == 1
    restored = agent_loop_module._data_proto_from_numpy_payload(put_values[0])
    assert restored.batch["prompts"].tolist() == [[7]]


def test_numpy_rollout_transport_round_trips_bfloat16_without_tensor_copy():
    source = DataProto(
        batch=TensorDict(
            {
                "bf16": torch.arange(6, dtype=torch.bfloat16).view(2, 3),
                "int64": torch.tensor([[1], [2]], dtype=torch.int64),
            },
            batch_size=2,
        ),
        non_tensor_batch={},
        meta_info={},
    )

    payload = agent_loop_module._data_proto_to_numpy_payload(source)
    restored = agent_loop_module._data_proto_from_numpy_payload(payload)

    assert restored.batch["bf16"].dtype is torch.bfloat16
    assert torch.equal(restored.batch["bf16"], source.batch["bf16"])
    assert torch.equal(restored.batch["int64"], source.batch["int64"])
    encoded_bf16 = payload.batch["bf16"][2]
    assert restored.batch["bf16"].untyped_storage().data_ptr() == encoded_bf16.ctypes.data
