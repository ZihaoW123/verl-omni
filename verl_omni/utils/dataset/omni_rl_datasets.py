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
"""Audio-aware RL dataset utilities for omni-modal training."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from omegaconf import DictConfig
from verl.utils.dataset.rl_dataset import RLHFDataset


def _ensure_audioread_ffmpeg() -> None:
    """Use imageio's bundled FFmpeg when no system decoder is available."""
    from audioread import ffdec

    if any(shutil.which(command) for command in ffdec.COMMANDS):
        return

    from imageio_ffmpeg import get_ffmpeg_exe

    ffmpeg_exe = get_ffmpeg_exe()
    if not Path(ffmpeg_exe).is_file():
        raise RuntimeError(
            "Audio extraction from video requires FFmpeg, but neither a system decoder nor imageio-ffmpeg's "
            f"bundled executable is available (resolved path: {ffmpeg_exe!r})."
        )
    ffdec.COMMANDS = (ffmpeg_exe,)


class QwenOmniRLHFDataset(RLHFDataset):
    """Adapt Qwen's multimodal media loader to verl's RL dataset interface.

    verl turns parquet media columns into structured messages. Qwen's
    ``process_mm_info`` then resolves image/audio/video paths into the media
    objects expected by the Qwen3-Omni processor and vLLM-Omni rollout.
    """

    @classmethod
    def _process_multi_modal_info(
        cls,
        messages: list[dict],
        image_patch_size: int,
        config: DictConfig,
    ) -> tuple[list[Any] | None, list[Any] | None, list[Any] | None]:
        from qwen_omni_utils import process_mm_info

        # Qwen returns (audios, images, videos); verl expects
        # (images, videos, audios). AVQA uses a standalone audio track, while
        # datasets such as OmniVideo-R1 can read the audio stream directly
        # from each video to avoid duplicating media on disk.
        use_audio_in_video = bool(config.get("use_audio_in_video", False))
        if use_audio_in_video:
            _ensure_audioread_ffmpeg()
        audios, images, videos = process_mm_info(
            messages,
            use_audio_in_video=use_audio_in_video,
            image_patch_size=image_patch_size,
        )
        return images, videos, audios
