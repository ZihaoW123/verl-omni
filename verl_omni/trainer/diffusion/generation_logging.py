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

import subprocess
import wave
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import numpy as np
import torch
from PIL import Image

_CHANNEL_COUNTS = {1, 3, 4}


def prepare_visual_outputs(outputs: torch.Tensor) -> tuple[Literal["image", "video"], np.ndarray]:
    """Convert batched diffusion outputs to uint8 BHWC images or BTHWC videos."""
    outputs = torch.as_tensor(outputs).detach().cpu().float()

    if outputs.ndim == 4:
        if outputs.shape[1] in _CHANNEL_COUNTS:
            media_type = "image"
            visual_outputs = outputs.permute(0, 2, 3, 1)
        elif outputs.shape[-1] in _CHANNEL_COUNTS:
            media_type = "image"
            visual_outputs = outputs
        else:
            # Some video backends return grayscale frames as BTHW.
            media_type = "video"
            visual_outputs = outputs.unsqueeze(-1)
    elif outputs.ndim == 5:
        media_type = "video"
        if outputs.shape[2] in _CHANNEL_COUNTS:  # BTCHW
            visual_outputs = outputs.permute(0, 1, 3, 4, 2)
        elif outputs.shape[1] in _CHANNEL_COUNTS:  # BCTHW
            visual_outputs = outputs.permute(0, 2, 3, 4, 1)
        elif outputs.shape[-1] in _CHANNEL_COUNTS:  # BTHWC
            visual_outputs = outputs
        else:
            raise ValueError(f"Cannot identify the channel axis of video outputs with shape {tuple(outputs.shape)}.")
    else:
        raise ValueError(f"Expected batched image/video outputs with 4 or 5 dimensions, got {tuple(outputs.shape)}.")

    visual_outputs = (visual_outputs * 255).round().clamp(0, 255).to(torch.uint8)
    if visual_outputs.shape[-1] == 1:
        visual_outputs = visual_outputs.expand(*visual_outputs.shape[:-1], 3)
    elif visual_outputs.shape[-1] == 4:
        visual_outputs = visual_outputs[..., :3]

    return media_type, visual_outputs.contiguous().numpy()


def batch_items(values: Any, batch_size: int, name: str) -> list[Any]:
    if values is None:
        return [None] * batch_size
    if isinstance(values, torch.Tensor | np.ndarray):
        if values.ndim == 0:
            return [values] * batch_size
        if values.shape[0] == batch_size:
            return list(values)
        if batch_size == 1:
            return [values]
        raise ValueError(f"{name} batch size {values.shape[0]} does not match visual batch size {batch_size}.")
    if isinstance(values, Sequence) and not isinstance(values, str | bytes):
        if len(values) != batch_size:
            raise ValueError(f"{name} batch size {len(values)} does not match visual batch size {batch_size}.")
        return list(values)
    return [values] * batch_size


def _audio_to_pcm16(audio: Any) -> np.ndarray:
    waveform = torch.as_tensor(audio).detach().cpu().float()
    while waveform.ndim > 2 and waveform.shape[0] == 1:
        waveform = waveform[0]
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim != 2:
        raise ValueError(f"Expected audio shape (T,) or (C,T), got {tuple(waveform.shape)}.")

    # Accept both channel-first and channel-last audio.
    if waveform.shape[0] > 8 and waveform.shape[1] <= 8:
        waveform = waveform.transpose(0, 1)
    if waveform.shape[0] > 2:
        waveform = waveform.mean(dim=0, keepdim=True)

    waveform = torch.nan_to_num(waveform).clamp(-1, 1)
    return (waveform.transpose(0, 1).numpy() * 32767).round().astype("<i2")


def _sample_rate_to_int(sample_rate: Any) -> int:
    if isinstance(sample_rate, torch.Tensor):
        sample_rate = sample_rate.detach().cpu().item()
    elif isinstance(sample_rate, np.ndarray):
        sample_rate = sample_rate.item()
    sample_rate = int(sample_rate)
    if sample_rate <= 0:
        raise ValueError(f"Audio sample rate must be positive, got {sample_rate}.")
    return sample_rate


def _write_wav(audio: Any, sample_rate: Any, path: Path) -> None:
    pcm = _audio_to_pcm16(audio)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(pcm.shape[1])
        wav_file.setsampwidth(2)
        wav_file.setframerate(_sample_rate_to_int(sample_rate))
        wav_file.writeframes(pcm.tobytes())


def _export_video_with_audio(
    video: np.ndarray,
    output_path: Path,
    fps: int,
    audio: Any,
    audio_sample_rate: Any,
    video_exporter: Callable[..., str],
    ffmpeg_exe: str,
) -> None:
    token = uuid4().hex
    silent_path = output_path.with_name(f".{output_path.stem}.{token}.silent.mp4")
    audio_path = output_path.with_name(f".{output_path.stem}.{token}.wav")
    try:
        video_exporter([Image.fromarray(frame) for frame in video], str(silent_path), fps=fps)
        _write_wav(audio, audio_sample_rate, audio_path)
        subprocess.run(
            [
                ffmpeg_exe,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(silent_path),
                "-i",
                str(audio_path),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                "-shortest",
                str(output_path),
            ],
            check=True,
        )
    finally:
        silent_path.unlink(missing_ok=True)
        audio_path.unlink(missing_ok=True)


def save_visual_outputs(
    outputs: torch.Tensor,
    output_dir: str | Path,
    *,
    fps: int,
    audios: Any = None,
    audio_sample_rates: Any = None,
    video_exporter: Callable[..., str] | None = None,
    ffmpeg_exe: str | None = None,
) -> tuple[Literal["image", "video"], list[str]]:
    """Save image/video outputs, muxing audio into video files when provided."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    media_type, visual_outputs = prepare_visual_outputs(outputs)

    if media_type == "image":
        output_paths = []
        for index, image in enumerate(visual_outputs):
            output_path = output_dir / f"{index}.jpg"
            Image.fromarray(image).save(output_path)
            output_paths.append(str(output_path))
        return media_type, output_paths

    if video_exporter is None:
        from diffusers.utils import export_to_video

        video_exporter = export_to_video

    audio_items = batch_items(audios, len(visual_outputs), "audio")
    sample_rate_items = batch_items(audio_sample_rates, len(visual_outputs), "audio_sample_rate")
    if any(audio is not None for audio in audio_items) and ffmpeg_exe is None:
        from imageio_ffmpeg import get_ffmpeg_exe

        ffmpeg_exe = get_ffmpeg_exe()

    output_paths = []
    for index, (video, audio, sample_rate) in enumerate(
        zip(visual_outputs, audio_items, sample_rate_items, strict=True)
    ):
        output_path = output_dir / f"{index}.mp4"
        if audio is None:
            video_exporter([Image.fromarray(frame) for frame in video], str(output_path), fps=fps)
        else:
            if sample_rate is None:
                raise ValueError("Audio sample rate is required when saving a video with audio.")
            _export_video_with_audio(
                video,
                output_path,
                fps,
                audio,
                sample_rate,
                video_exporter,
                ffmpeg_exe,
            )
        output_paths.append(str(output_path))
    return media_type, output_paths
