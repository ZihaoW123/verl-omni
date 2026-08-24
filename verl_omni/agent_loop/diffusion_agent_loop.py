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
import copy
import os
import random
import warnings
from dataclasses import dataclass
from typing import Any, Optional

import hydra
import numpy as np
import ray
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict
from ray._private.internal_api import free as free_ray_object_refs
from tensordict import TensorDict
from verl.base_config import BaseConfig
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopManager,
    AgentLoopMetrics,
    DictConfigWrap,
    _agent_loop_registry,
)
from verl.experimental.agent_loop.utils import resolve_config_path
from verl.protocol import DataProto
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.dataset.rl_dataset import get_dataset_class
from verl.utils.profiler import simple_timer
from verl.utils.ray_utils import auto_await
from verl.utils.skip import SkipManager
from verl.workers.rollout.llm_server import LLMServerClient

from verl_omni.agent_loop.utils import maybe_per_rollout_seeds
from verl_omni.utils.process_memory import (
    collect_and_trim_process_memory,
    current_process_rss_bytes,
    npu_host_memory_stats_bytes,
    process_memory_breakdown_bytes,
)
from verl_omni.workers.config import DiffusionModelConfig, DiffusionRolloutConfig


def _config_to_sampling_dict(config: Optional[BaseConfig]) -> dict:
    if config is None:
        return {}
    return {k: v for k, v in config.items() if not k.startswith("_")}


def _pad_prompt_extra_field(key: str, value: torch.Tensor, target_length: int) -> torch.Tensor:
    if key in {"prompt_embeds", "negative_prompt_embeds"}:
        current_length = int(value.shape[0])
        if current_length > target_length:
            raise ValueError(
                f"{key} sequence length {current_length} exceeds max_prompt_embed_length={target_length}. "
                "Configure max_prompt_embed_length for the final embedding sequence, not only one text encoder."
            )
        return F.pad(value, (0, 0, 0, target_length - current_length), value=0)
    if key in {"prompt_embeds_mask", "negative_prompt_embeds_mask"}:
        current_length = int(value.shape[0])
        if current_length > target_length:
            raise ValueError(
                f"{key} sequence length {current_length} exceeds max_prompt_embed_length={target_length}. "
                "Configure max_prompt_embed_length for the final embedding sequence, not only one text encoder."
            )
        return F.pad(value, (0, target_length - current_length), value=0)
    return value


@dataclass
class _NumpyDataProtoPayload:
    """Ray transport that keeps large CPU tensors in NumPy/Plasma buffers."""

    batch_size: tuple[int, ...] | None
    batch: dict[str, tuple[Any, ...]] | None
    non_tensor_batch: dict[str, np.ndarray]
    meta_info: dict[str, Any]


def _tensor_to_numpy_payload(tensor: torch.Tensor) -> tuple[str, tuple[int, ...], np.ndarray]:
    """Expose a CPU tensor as bytes without allocating another tensor-sized buffer."""
    if tensor.device.type != "cpu":
        raise ValueError(f"Diffusion rollout transport requires CPU tensors, got {tensor.device}.")
    tensor = tensor.detach().contiguous()
    data = tensor.flatten().view(torch.uint8).numpy()
    return str(tensor.dtype).removeprefix("torch."), tuple(tensor.shape), data


def _tensor_from_numpy_payload(payload: tuple[str, tuple[int, ...], np.ndarray]) -> torch.Tensor:
    """Create a read-only tensor view over a Ray-owned NumPy buffer."""
    dtype_name, shape, data = payload
    dtype = getattr(torch, dtype_name)
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"Invalid torch dtype in rollout payload: {dtype_name!r}.")
    # Ray exposes Plasma-backed NumPy arrays as read-only. The view is only read
    # by DataProto.concat before its ObjectRef is freed, so no write is attempted.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
        raw = torch.from_numpy(data)
    return raw.view(dtype).view(shape)


def _data_proto_to_numpy_payload(output: DataProto) -> _NumpyDataProtoPayload:
    if output.batch is None:
        batch_size = None
        encoded_batch = None
    else:
        batch_size = tuple(output.batch.batch_size)
        encoded_batch = {}
        for key, value in output.batch.items():
            if value.is_nested:
                layout = str(value.layout).removeprefix("torch.")
                encoded_batch[key] = (layout, [_tensor_to_numpy_payload(item) for item in value.unbind()])
            else:
                encoded_batch[key] = _tensor_to_numpy_payload(value)
    return _NumpyDataProtoPayload(
        batch_size=batch_size,
        batch=encoded_batch,
        non_tensor_batch=output.non_tensor_batch,
        meta_info=output.meta_info,
    )


def _data_proto_from_numpy_payload(payload: _NumpyDataProtoPayload) -> DataProto:
    if payload.batch is None:
        batch = None
    else:
        decoded_batch = {}
        for key, value in payload.batch.items():
            if len(value) == 3:
                decoded_batch[key] = _tensor_from_numpy_payload(value)
            elif len(value) == 2:
                layout, tensors = value
                decoded_batch[key] = torch.nested.as_nested_tensor(
                    [_tensor_from_numpy_payload(tensor) for tensor in tensors],
                    layout=getattr(torch, layout),
                )
            else:
                raise ValueError(f"Invalid rollout tensor payload for {key!r}: expected 2 or 3 items.")
        batch = TensorDict(source=decoded_batch, batch_size=payload.batch_size)
    return DataProto(batch=batch, non_tensor_batch=payload.non_tensor_batch, meta_info=payload.meta_info)


class DiffusionAgentLoopOutput(BaseModel):
    """Agent loop output."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_ids: list[int]
    """Prompt token ids."""
    response_diffusion_output: Any
    """Response pixels (CHW/TCHW) as uint8 values in [0, 255], or floating-point latents."""
    response_logprobs: Optional[Any] = None
    """Log probabilities for the response tokens. (torch.Tensor)"""
    reward_score: Optional[float] = None
    """Reward score for the trajectory."""
    num_turns: int = 0
    """Number of chat turns, including user, assistant, tool."""
    metrics: AgentLoopMetrics
    """Auxiliary performance metrics"""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""


class _InternalDiffusionAgentLoopOutput(DiffusionAgentLoopOutput):
    """Internal agent loop output with padded sequences."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_ids: torch.Tensor
    """Padded prompt token ids."""
    response_diffusion_output: torch.Tensor
    """Response pixels (NCHW/NTCHW) as uint8 values in [0, 255], or floating-point latents."""
    response_logprobs: Optional[torch.Tensor] = None
    """Log probabilities over denoising timesteps."""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""


class DiffusionAgentLoopWorker:
    """Diffusion Agent loop worker takes a batch of messages and run each message in an agent loop.

    Args:
        config (DictConfig): whole config for main entrypoint.
        llm_client (LLMServerClient): Client for the LLM server replicas, produced by
            ``LLMServerManager.get_client()`` in the trainer.
        teacher_client (dict[str, LLMServerClient]): Not used by diffusion training; accepted to
            keep the constructor signature compatible with verl's ``AgentLoopManager.create()``,
            which positionally forwards a teacher client argument to each worker.
        reward_loop_worker_handles (List[ray.actor.ActorHandle]): Actor handles for streaming
            reward computation.
    """

    def __init__(
        self,
        config: DictConfig,
        llm_client: LLMServerClient,
        teacher_client: dict[str, LLMServerClient] | None = None,
        reward_loop_worker_handles: list[ray.actor.ActorHandle] = None,
    ):
        self.config = config
        rollout_config = config.actor_rollout_ref.rollout
        model_config = config.actor_rollout_ref.model
        self.rollout_config: DiffusionRolloutConfig = omega_conf_to_dataclass(rollout_config)
        self.model_config: DiffusionModelConfig = omega_conf_to_dataclass(model_config)

        if not hasattr(self, "server_manager"):
            self.server_manager = llm_client

        self.dataset_cls = get_dataset_class(config.data)
        self.reward_loop_worker_handles = reward_loop_worker_handles
        self.rollout_only = bool(config.trainer.get("rollout_only", False))
        self.rollout_only_memory_trim = bool(config.trainer.get("rollout_only_memory_trim", False))
        self.rollout_only_drop_outputs = bool(config.trainer.get("rollout_only_drop_outputs", False))
        self._rollout_only_rss_baseline_bytes: Optional[int] = None

        self.tokenizer = self.model_config.tokenizer
        self.processor = self.model_config.processor

        self.max_prompt_embed_length = self.rollout_config.max_prompt_embed_length
        if self.max_prompt_embed_length is None:
            self.max_prompt_embed_length = self.rollout_config.pipeline.max_sequence_length
        if self.max_prompt_embed_length <= 0:
            raise ValueError(f"max_prompt_embed_length must be positive, got {self.max_prompt_embed_length}.")

        hf_model_type = getattr(self.model_config.hf_config, "model_type", None)
        self.hf_model_type: str | None = hf_model_type if isinstance(hf_model_type, str) else None

        agent_loop_config_path = self.rollout_config.agent.agent_loop_config_path
        if agent_loop_config_path:
            resolved_path = resolve_config_path(agent_loop_config_path)
            agent_loop_configs = OmegaConf.load(resolved_path)
            for agent_loop_config in agent_loop_configs:
                _agent_loop_registry[agent_loop_config.name] = agent_loop_config
        if self.model_config.get("custom_chat_template", None) is not None:
            if self.model_config.processor is not None:
                self.model_config.processor.chat_template = self.model_config.custom_chat_template
            self.model_config.tokenizer.chat_template = self.model_config.custom_chat_template

    async def generate_sequences(self, batch: DataProto) -> DataProto:
        """Generate sequences from agent loop.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch with the following fields.

            - ``prompts``: ``[bsz, prompt_length]`` prompt token ids from dataset.
            - ``responses``: uint8 pixel output in ``[0, 255]``, typically
              ``[bsz, C, H, W]`` (image) or ``[bsz, T, C, H, W]`` (video).
              Latent output remains floating point.
            - ``rm_scores`` (optional): ``[bsz, 1]`` reward model scores.
            - ``meta_info``:

              - ``metrics``: ``List[dict]``, per-sample agent loop metrics.
              - ``reward_extra_keys`` (optional): ``List[str]``, keys for reward
                extra info for logging/validation.
        """
        config = self.rollout_config
        if self.rollout_only:
            if self.rollout_only_memory_trim:
                collect_and_trim_process_memory()
            self._log_rollout_only_memory(batch.meta_info.get("global_steps"), "start")

        sampling_params = {
            **_config_to_sampling_dict(config.pipeline),
            **_config_to_sampling_dict(config.algo),
            "logprobs": config.calculate_log_probs,
        }
        if self.rollout_only and self.rollout_only_drop_outputs:
            sampling_params["_verl_rollout_only_drop_outputs"] = True

        is_validate = batch.meta_info.get("validate", False)
        per_rollout_seeds: Optional[list[int]] = None

        if is_validate:
            sampling_params.update(_config_to_sampling_dict(config.val_kwargs.pipeline))
            sampling_params.update(_config_to_sampling_dict(config.val_kwargs.algo))
            sampling_params["seed"] = config.val_kwargs.seed
            sampling_params["logprobs"] = False
        else:
            sampling_params["global_steps"] = batch.meta_info["global_steps"]
            # Prefer trainer-assigned global indices so chunked workers derive the
            # same per-row seed regardless of local batch position / pack order.
            global_indices = batch.non_tensor_batch.get("_rollout_seed_global_idx")
            if global_indices is not None:
                global_indices = np.asarray(global_indices, dtype=np.int64).reshape(-1)
            per_rollout_seeds = maybe_per_rollout_seeds(batch.meta_info, len(batch), global_indices)

        if "agent_name" not in batch.non_tensor_batch:
            default_agent_loop = config.agent.default_agent_loop
            batch.non_tensor_batch["agent_name"] = np.array([default_agent_loop] * len(batch), dtype=object)

        tasks = []
        for i in range(len(batch)):
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            task_sampling_params = sampling_params.copy()
            if per_rollout_seeds is not None:
                task_sampling_params["seed"] = per_rollout_seeds[i]
            tasks.append(
                asyncio.create_task(self._run_agent_loop(task_sampling_params, validate=is_validate, **kwargs))
            )
        outputs = await asyncio.gather(*tasks)

        if self.rollout_only and self.rollout_only_drop_outputs:
            self._log_rollout_only_payload(batch.meta_info.get("global_steps"), outputs)
            output = self._postprocess_rollout_only(outputs)
        else:
            output = self._postprocess(outputs, input_non_tensor_batch=batch.non_tensor_batch)

        # Completed asyncio tasks retain their result. Drop both containers before
        # Ray serializes the compact diagnostic response.
        del outputs, tasks

        if self.rollout_only:
            if self.rollout_only_memory_trim:
                collect_and_trim_process_memory()
            self._log_rollout_only_memory(batch.meta_info.get("global_steps"), "end")

        return output

    async def generate_sequences_to_object_store(self, batch: DataProto) -> ray.ObjectRef:
        """Put a NumPy-backed rollout payload in Ray's object store.

        Ray's Torch deserializer allocates new CPU tensor storage for actor results.
        The aarch64 allocator does not reliably return that storage, so repeated
        multi-GiB rollouts grow the manager RSS. NumPy buffers remain Plasma-backed
        and are viewed without a tensor-sized receive-side allocation.
        """
        global_step = batch.meta_info.get("global_steps")
        output = await self.generate_sequences(batch)
        payload = _data_proto_to_numpy_payload(output)
        output_ref = ray.put(payload)
        del payload, output

        if self.rollout_only:
            if self.rollout_only_memory_trim:
                collect_and_trim_process_memory()
            self._log_rollout_only_memory(global_step, "after_put")

        return output_ref

    def _log_rollout_only_memory(self, global_step: Optional[int], phase: str) -> None:
        rss_bytes = current_process_rss_bytes()
        breakdown = process_memory_breakdown_bytes()
        npu_host_stats = npu_host_memory_stats_bytes()
        if self._rollout_only_rss_baseline_bytes is None:
            self._rollout_only_rss_baseline_bytes = rss_bytes
        baseline_delta_bytes = rss_bytes - self._rollout_only_rss_baseline_bytes
        private_bytes = breakdown.get("Private_Clean", 0) + breakdown.get("Private_Dirty", 0)
        shared_bytes = breakdown.get("Shared_Clean", 0) + breakdown.get("Shared_Dirty", 0)
        anonymous_bytes = breakdown.get("Anonymous", 0)
        pss_bytes = breakdown.get("Pss", 0)
        npu_host_allocated_bytes = npu_host_stats.get("allocated_bytes.current", 0)
        npu_host_active_bytes = npu_host_stats.get("active_bytes.current", 0)
        gib = 1024**3
        print(
            "[DEBUG-rollout-only-rss] "
            f"pid={os.getpid()} step={global_step} phase={phase} "
            f"rss_gib={rss_bytes / gib:.3f} pss_gib={pss_bytes / gib:.3f} "
            f"private_gib={private_bytes / gib:.3f} shared_gib={shared_bytes / gib:.3f} "
            f"anonymous_gib={anonymous_bytes / gib:.3f} "
            f"npu_host_allocated_gib={npu_host_allocated_bytes / gib:.3f} "
            f"npu_host_active_gib={npu_host_active_bytes / gib:.3f} "
            f"baseline_delta_gib={baseline_delta_bytes / gib:+.3f}",
            flush=True,
        )

    def _log_rollout_only_payload(
        self,
        global_step: Optional[int],
        outputs: list[_InternalDiffusionAgentLoopOutput],
    ) -> None:
        sizes: dict[str, int] = {}

        def add_size(key: str, value: Any) -> None:
            if isinstance(value, torch.Tensor):
                sizes[key] = sizes.get(key, 0) + value.numel() * value.element_size()
            elif isinstance(value, np.ndarray):
                sizes[key] = sizes.get(key, 0) + value.nbytes

        for item in outputs:
            add_size("responses", item.response_diffusion_output)
            add_size("rollout_log_probs", item.response_logprobs)
            for key, value in item.extra_fields.items():
                add_size(key, value)

        gib = 1024**3
        field_sizes = ",".join(f"{key}:{value / gib:.3f}" for key, value in sorted(sizes.items()))
        print(
            "[DEBUG-rollout-only-payload] "
            f"pid={os.getpid()} step={global_step} total_gib={sum(sizes.values()) / gib:.3f} "
            f"fields_gib={field_sizes}",
            flush=True,
        )

    def _postprocess_rollout_only(self, inputs: list[_InternalDiffusionAgentLoopOutput]) -> DataProto:
        """Return only the small fields required by AgentLoopManager diagnostics."""
        prompt_ids = torch.cat([item.prompt_ids for item in inputs], dim=0)
        metrics = [item.metrics.model_dump() for item in inputs]
        batch = TensorDict({"prompts": prompt_ids}, batch_size=len(inputs))
        return DataProto(batch=batch, non_tensor_batch={}, meta_info={"metrics": metrics})

    async def _run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        *,
        agent_name: str,
        validate: bool = False,
        **kwargs,
    ) -> _InternalDiffusionAgentLoopOutput:
        assert agent_name in _agent_loop_registry, (
            f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
        )

        agent_loop_config = _agent_loop_registry[agent_name]
        agent_loop = hydra.utils.instantiate(
            config=agent_loop_config,
            trainer_config=DictConfigWrap(config=self.config),
            server_manager=self.server_manager,
            tokenizer=self.tokenizer,
            processor=self.processor,
            dataset_cls=self.dataset_cls,
            data_config=DictConfigWrap(self.config.data),
            hf_model_type=self.hf_model_type,
            extra_tokenizer_map=self.model_config.extra_tokenizer_map,
        )
        output: DiffusionAgentLoopOutput = await agent_loop.run(sampling_params, **kwargs)
        return await self._agent_loop_postprocess(output, validate=validate, **kwargs)

    async def _agent_loop_postprocess(
        self, output, validate: bool = False, **kwargs
    ) -> _InternalDiffusionAgentLoopOutput:
        """Perform post-processing operations on the output of each individual agent loop."""
        # Pad extra tensor outputs from vllm-omni (e.g. prompt embeddings).
        extra_fields = {}
        for k, v in output.extra_fields.items():
            if isinstance(v, torch.Tensor):
                v = _pad_prompt_extra_field(k, v, self.max_prompt_embed_length)
                extra_fields[k] = v.unsqueeze(0)
            else:
                extra_fields[k] = v

        extra_fields["raw_prompt"] = kwargs["raw_prompt"]

        prompt_output = self.tokenizer.pad(
            {"input_ids": output.prompt_ids},
            padding="max_length",
            max_length=self.rollout_config.prompt_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if prompt_output["input_ids"].dim() == 1:
            prompt_output["input_ids"] = prompt_output["input_ids"].unsqueeze(0)
            prompt_output["attention_mask"] = prompt_output["attention_mask"].unsqueeze(0)

        response_diffusion_output = output.response_diffusion_output.unsqueeze(0)

        response_logprobs = None
        if output.response_logprobs is not None:
            response_logprobs = output.response_logprobs.unsqueeze(0)

        prompt_ids = prompt_output["input_ids"]
        extra_fields["attention_mask"] = prompt_output["attention_mask"]

        await self._compute_score(
            output,
            prompts=prompt_ids,
            responses=response_diffusion_output,
            kwargs=kwargs,
            validate=validate,
        )

        if "reward_extra_info" in output.extra_fields:
            extra_fields["reward_extra_info"] = output.extra_fields["reward_extra_info"]

        return _InternalDiffusionAgentLoopOutput(
            prompt_ids=prompt_ids,
            response_diffusion_output=response_diffusion_output,
            response_logprobs=response_logprobs,
            reward_score=output.reward_score,
            num_turns=output.num_turns,
            metrics=output.metrics,
            extra_fields=extra_fields,
        )

    async def _compute_score(self, output, prompts, responses, kwargs, validate: bool = False):
        """Compute reward score for single sample."""
        enable_async_reward = self.reward_loop_worker_handles is not None

        if output.reward_score is None and enable_async_reward:
            timing = {}
            with simple_timer("compute_score", timing):
                batch = TensorDict(
                    {
                        "prompts": prompts,  # [1, prompt_length]
                        "responses": responses,  # [1, C, H, W] or [1, T, C, H, W]
                    },
                    batch_size=1,
                )
                non_tensor_batch = {
                    **{k: np.array([v]) for k, v in kwargs.items()},
                    "__num_turns__": np.array([output.num_turns]),
                    "tool_extra_fields": np.array([output.extra_fields], dtype=object),
                }

                data = DataProto(
                    batch=batch,
                    non_tensor_batch=non_tensor_batch,
                    meta_info={"validate": validate},
                )
                selected_reward_loop_worker_handle = random.choice(self.reward_loop_worker_handles)
                result = await selected_reward_loop_worker_handle.compute_score.remote(data)
                output.reward_score = result["reward_score"]
                output.extra_fields["reward_extra_info"] = result["reward_extra_info"]
            output.metrics.compute_score = timing["compute_score"]

    def _postprocess(
        self,
        inputs: list[_InternalDiffusionAgentLoopOutput],
        input_non_tensor_batch: dict | None = None,
    ) -> DataProto:
        """Process the padded outputs from _run_agent_loop and combine them into a batch."""
        # Convert lists back to tensors and stack them to create a batch.
        prompt_ids = torch.cat([input.prompt_ids for input in inputs], dim=0)
        response_diffusion_output = torch.cat([input.response_diffusion_output for input in inputs], dim=0)
        optional_outputs = {}
        if inputs[0].response_logprobs is not None:
            optional_outputs["rollout_log_probs"] = torch.cat([input.response_logprobs for input in inputs], dim=0)

        # Handle extra fields that are tensors
        extra_keys = [k for k, v in inputs[0].extra_fields.items() if isinstance(v, torch.Tensor)]
        for key in extra_keys:
            optional_outputs[key] = torch.cat([input.extra_fields[key] for input in inputs], dim=0)
            for input in inputs:
                del input.extra_fields[key]

        batch = TensorDict(
            {
                "prompts": prompt_ids,  # [bsz, prompt_length]
                "responses": response_diffusion_output,  # [bsz, C, H, W] or [bsz, T, C, H, W]
                **optional_outputs,
            },
            batch_size=len(inputs),
        )

        scores = [input.reward_score for input in inputs]
        if all(score is not None for score in scores):
            rm_scores = torch.tensor(scores, dtype=torch.float32).unsqueeze(-1)
            batch["rm_scores"] = rm_scores

        non_tensor_batch = {
            "__num_turns__": np.array([input.num_turns for input in inputs], dtype=np.int32),
        }
        if input_non_tensor_batch:
            non_tensor_batch.update(input_non_tensor_batch)

        # add reward_extra_info to non_tensor_batch
        reward_extra_infos = [input.extra_fields.get("reward_extra_info", {}) for input in inputs]
        reward_extra_keys = sorted(set.intersection(*(set(info) for info in reward_extra_infos)))
        for key in reward_extra_keys:
            non_tensor_batch[key] = np.array([info[key] for info in reward_extra_infos])

        metrics = [input.metrics.model_dump() for input in inputs]
        # Collect extra fields from all inputs and convert them to np.ndarray
        extra_fields = {}
        all_keys = set(key for input_item in inputs for key in input_item.extra_fields)
        for key in all_keys:
            temp_arr = np.empty(len(inputs), dtype=object)
            temp_arr[:] = [input.extra_fields.get(key) for input in inputs]
            extra_fields[key] = temp_arr

        non_tensor_batch.update(extra_fields)

        # Only include reward_extra_keys in meta_info if rm_scores is in batch
        # This avoids conflicts when reward_tensor is merged later in ray_trainer.py
        if "rm_scores" in batch.keys():
            meta_info = {"metrics": metrics, "reward_extra_keys": reward_extra_keys}
        else:
            meta_info = {"metrics": metrics}

        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info=meta_info,
        )


class DiffusionAgentLoopManager(AgentLoopManager):
    """Agent-loop manager that uses zero-copy Ray inputs and releases them promptly.

    ``DataProto.concat`` copies the read-only NumPy-backed tensor views into normal
    owned tensors. The Ray objects can then be released before the next rollout.
    """

    @auto_await
    @SkipManager.annotate(role="rollout")
    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Dispatch rollout chunks, concatenate their results, and free their Ray objects."""
        global_step = prompts.meta_info.get("global_steps")
        chunks = prompts.chunk(len(self.agent_loop_workers))
        outer_refs = [
            worker.generate_sequences_to_object_store.remote(chunk)
            for worker, chunk in zip(self.agent_loop_workers, chunks, strict=True)
        ]
        # Do not await Ray ObjectRefs through asyncio. Completed Futures retain
        # their deserialized tensor results on aarch64 even after local deletion.
        data_refs = ray.get(outer_refs)
        payloads = ray.get(data_refs)
        outputs = [_data_proto_from_numpy_payload(payload) for payload in payloads]

        try:
            output = DataProto.concat(outputs)
            # torch.cat already detaches the TensorDict storage. Object-dtype
            # NumPy fields can still contain views into a Ray object, so deep-copy
            # the much smaller non-tensor payload before freeing the ObjectRefs.
            output.non_tensor_batch = copy.deepcopy(output.non_tensor_batch)
            metrics = [chunk_output.meta_info.pop("metrics") for chunk_output in outputs]
            timing = self._performance_metrics(metrics, output)
            output.meta_info = {"timing": timing, **outputs[0].meta_info}
            return output
        finally:
            # The concatenated DataProto is independent of the per-worker Ray
            # objects. Release both the explicit payload refs and the small actor
            # result refs before the next rollout.
            outputs.clear()
            payloads.clear()
            free_ray_object_refs(data_refs + outer_refs, local_only=False)
            if self.config.trainer.get("rollout_only", False):
                print(
                    "[DEBUG-rollout-ray-lifecycle] "
                    f"pid={os.getpid()} step={global_step} phase=manager_freed "
                    f"payload_refs={len(data_refs)} actor_refs={len(outer_refs)}",
                    flush=True,
                )
