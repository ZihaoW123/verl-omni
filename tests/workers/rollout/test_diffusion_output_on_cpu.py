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


def test_unbatch_single_request_video_output() -> None:
    replica = _load_rollout_replica_module()
    batched_video = torch.zeros(1, 81, 3, 8, 10)

    video = replica.unbatch_single_request_video(batched_video)

    assert video.shape == (81, 3, 8, 10)


def test_unbatch_single_request_video_preserves_tchw() -> None:
    replica = _load_rollout_replica_module()
    single_frame_video = torch.zeros(1, 3, 8, 10)

    video = replica.unbatch_single_request_video(single_frame_video)

    assert video.shape == (1, 3, 8, 10)
