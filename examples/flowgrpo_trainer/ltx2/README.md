# LTX-2.3 text-to-audio-video FlowGRPO

This recipe trains `dg845/LTX-2.3-Diffusers` LoRA adapters with a diffusers +
FSDP2 actor, vLLM-Omni rollout, joint audio-video CPS transitions, and the CLAP
plus ImageBind rewards.
The checkpoint advertises `_class_name: LTX2Pipeline`; the registered rollout
adapter uses vLLM-Omni's LTX-2.3-specific `LTX23Pipeline` implementation behind
that checkpoint architecture key.

## Prepare data

The prompt corpus is derived from
[VidProM](https://huggingface.co/datasets/WenhaoWang/VidProM), a CC-BY-NC 4.0
dataset containing 1.67 million unique text-to-video prompts from real users.
This recipe uses the 50,000-prompt subset from
[`video_prompts.txt`](https://github.com/XueZeyue/DanceGRPO/blob/main/assets/video_prompts.txt)
with the following split:

- `train.txt`: 48,976 prompts
- `test.txt`: 1,024 prompts

Use `dataset/vid_prompt/train.txt` and `test.txt`:

```bash
python3 examples/flowgrpo_trainer/ltx2/prepare_data.py \
  --input_dir ./dataset/vid_prompt \
  --output_dir "$WORKSPACE/data/vid_prompt/verl_omni"
```

Both `--train_size` and `--val_size` default to `-1`, so all prompts in each
split are converted unless an explicit non-negative limit is provided.

## Install reward dependencies

CLAP uses the existing `transformers` and `torchaudio` dependencies. ImageBind
is optional software under the CC-BY-NC-SA 4.0 non-commercial license:

```bash
pip install 'git+https://github.com/facebookresearch/ImageBind.git'
pip install 'git+https://github.com/facebookresearch/pytorchvideo.git'
```

Review the ImageBind license before enabling this reward in your environment.

## Launch

```bash
bash examples/flowgrpo_trainer/ltx2/run_ltx2_3_t2av_lora.sh
```

The recipe defaults to 8 GPUs and vLLM-Omni tensor parallel size 8. Override
`NUM_GPUS`, `ROLLOUT_TP`, `MODEL_PATH`, `DATA_DIR`, `OUTPUT_DIR`, or
`TOTAL_TRAINING_STEPS` through environment variables. Extra Hydra overrides can
be appended to the command. One verl-omni global step consumes the same 48
unique prompts and 16 responses per prompt as one reference training epoch. The
default 15 global steps and a 24-prompt PPO mini-batch therefore reproduce the
reference recipe's 15 epochs and two optimizer updates per epoch.

The reference training recipe maintains a separate EMA evaluation copy.
The current verl-omni FlowGRPO trainer evaluates and checkpoints the live LoRA
policy, so the EMA-only evaluation behavior is the one reference option not
mapped by this launch script.
