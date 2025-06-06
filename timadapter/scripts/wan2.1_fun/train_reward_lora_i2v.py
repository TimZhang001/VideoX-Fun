"""Modified from VideoX-Fun/scripts/wan2.1_fun/train_lora.py
"""
#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import argparse
import gc
import json
import logging
import math
import os
import random
import shutil
import sys
from contextlib import contextmanager
from typing import List, Optional

import accelerate
import diffusers
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import DDIMScheduler, FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from einops import rearrange
from omegaconf import OmegaConf
from packaging import version
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers.utils import ContextManagers
import datasets

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

sys.path.append("/mnt/vision-gen-ssd/zhangss/VideoX-Fun")
sys.path.append("/home/zhangss/remote/DPO/VideoRL/VideoX-Fun")

import videox_fun.reward.reward_fn as reward_fn
from videox_fun.models import (AutoencoderKLWan, CLIPModel, WanT5EncoderModel, WanTransformer3DModel)
from videox_fun.utils.lora_utils import create_network, merge_lora
from videox_fun.utils.utils import save_videos_grid
from timadapter.tools.process_video_frames import process_video_frames
from timadapter.tools.VideoPromptDataset import load_prompts_videos

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")

logger = get_logger(__name__, log_level="INFO")

def _get_t5_prompt_embeds(
    tokenizer,
    text_encoder,
    prompt = None,
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 512,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_attention_mask = text_inputs.attention_mask
    untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
        removed_text = tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
        logger.warning(
            "The following part of your input was truncated because `max_sequence_length` is set to "
            f" {max_sequence_length} tokens: {removed_text}"
        )

    seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
    prompt_embeds = text_encoder(text_input_ids.to(device), attention_mask=prompt_attention_mask.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    return [u[:v] for u, v in zip(prompt_embeds, seq_lens)]

def encode_prompt(
    tokenizer,
    text_encoder,
    prompt,
    negative_prompt,
    do_classifier_free_guidance: bool = True,
    num_videos_per_prompt: int = 1,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    max_sequence_length: int = 512,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    r"""
    Encodes the prompt into text encoder hidden states.

    Args:
        prompt (`str` or `List[str]`, *optional*):
            prompt to be encoded
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation. If not defined, one has to pass
            `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
            less than `1`).
        do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
            Whether to use classifier free guidance or not.
        num_videos_per_prompt (`int`, *optional*, defaults to 1):
            Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
        prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
        negative_prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
            argument.
        device: (`torch.device`, *optional*):
            torch device
        dtype: (`torch.dtype`, *optional*):
            torch dtype
    """
    prompt = [prompt] if isinstance(prompt, str) else prompt
    if prompt is not None:
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    if prompt_embeds is None:
        prompt_embeds = _get_t5_prompt_embeds(
            tokenizer,
            text_encoder,
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )

    if do_classifier_free_guidance and negative_prompt_embeds is None:
        negative_prompt = negative_prompt or ""
        negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

        if prompt is not None and type(prompt) is not type(negative_prompt):
            raise TypeError(
                f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                f" {type(prompt)}."
            )
        elif batch_size != len(negative_prompt):
            raise ValueError(
                f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                " the batch size of `prompt`."
            )

        negative_prompt_embeds = _get_t5_prompt_embeds(
            tokenizer,
            text_encoder,
            prompt=negative_prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )

    return prompt_embeds, negative_prompt_embeds

def filter_kwargs(cls, kwargs):
    import inspect
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return filtered_kwargs

# Modified from EasyAnimateInpaintPipeline.prepare_extra_step_kwargs
def prepare_extra_step_kwargs(scheduler, generator, eta):
    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
    # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
    # and should be between [0, 1]
    import inspect
    
    accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs["eta"] = eta

    # check if the scheduler accepts generator
    accepts_generator = "generator" in set(inspect.signature(scheduler.step).parameters.keys())
    if accepts_generator:
        extra_step_kwargs["generator"] = generator
    return extra_step_kwargs

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--use_came",
        action="store_true",
        help="whether to use came",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing (for DiT) to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--vae_gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing (for VAE) to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_model_info", action="store_true", help="Whether or not to report more info about model (such as norm, grad)."
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="image2video-fun-rl-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    
    parser.add_argument(
        "--rank",
        type=int,
        default=128,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--network_alpha",
        type=int,
        default=64,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help=(
            "The config of the model in training."
        ),
    )
    parser.add_argument(
        "--transformer_path",
        type=str,
        default=None,
        help=("If you want to load the weight from other transformers, input its path."),
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help=("If you want to load the weight from other vaes, input its path."),
    )
    parser.add_argument("--save_state", action="store_true", help="Whether or not to save state.")

    parser.add_argument(
        "--use_deepspeed", action="store_true", help="Whether or not to use deepspeed."
    )
    parser.add_argument(
        "--low_vram", action="store_true", help="Whether enable low_vram mode."
    )

    parser.add_argument(
        "--prompt_path",
        type=str,
        default="normal",
        help="The path to the training prompt file.",
    )
    parser.add_argument(
        '--train_sample_height', 
        type=int,
        default=384,
        help='The height of sampling videos in training'
    )
    parser.add_argument(
        '--train_sample_width', 
        type=int,
        default=672,
        help='The width of sampling videos in training'
    )
    parser.add_argument(
        "--video_length", 
        type=int,
        default=49,
        help="The number of frames to generate in training and validation."
    )
    parser.add_argument(
        '--eta', 
        type=float,
        default=0.0,
        help='eta parameter for the DDIM sampler. this controls the amount of noise injected into the sampling process, '
        'with 0.0 being fully deterministic and 1.0 being equivalent to the DDPM sampler.'
    )
    parser.add_argument(
        "--guidance_scale", 
        type=float,
        default=6.0,
        help="The classifier-free diffusion guidance."
    )
    parser.add_argument(
        "--num_inference_steps", 
        type=int,
        default=50,
        help="The number of denoising steps in training and validation."
    )
    parser.add_argument(
        "--num_decoded_latents",
        type=int,
        default=3,
        help="The number of latents to be decoded."
    )
    parser.add_argument(
        "--num_sampled_frames",
        type=int,
        default=None,
        help="The number of sampled frames for the reward function."
    )
    parser.add_argument(
        "--reward_fn", 
        type=str,
        default="aesthetic_loss_fn",
        help='The reward function.'
    )
    parser.add_argument(
        "--reward_fn_kwargs",
        type=str,
        default=None,
        help='The keyword arguments of the reward function.'
    )
    parser.add_argument(
        "--backprop",
        action="store_true",
        default=False,
        help="Whether to use the reward backprop training mode.",
    )
    parser.add_argument(
        "--backprop_step_list",
        nargs="+",
        type=int,
        default=None,
        help="The preset step list for reward backprop. If provided, overrides `backprop_strategy`."
    )
    parser.add_argument(
        "--backprop_strategy",
        choices=["last", "tail", "uniform", "random"],
        default="last",
        help="The strategy for reward backprop."
    )
    parser.add_argument(
        "--stop_latent_model_input_gradient",
        action="store_true",
        default=False,
        help="Whether to stop the gradient of the latents during reward backprop.",
    )
    parser.add_argument(
        "--backprop_random_start_step",
        type=int,
        default=0,
        help="The random start step for reward backprop. Only used when `backprop_strategy` is random."
    )
    parser.add_argument(
        "--backprop_random_end_step",
        type=int,
        default=50,
        help="The random end step for reward backprop. Only used when `backprop_strategy` is random."
    )
    parser.add_argument(
        "--backprop_num_steps",
        type=int,
        default=5,
        help="The number of steps for backprop. Only used when `backprop_strategy` is tail/uniform/random."
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def set_parse(args):
    args.config_path = "config/wan2.1/wan_civitai.yaml"
    args.pretrained_model_name_or_path="models/Diffusion_Transformer/Wan2.1-Fun-1.3B-InP"
    args.rank=32
    args.network_alpha=16
    args.train_batch_size=1
    args.gradient_accumulation_steps=1
    args.max_train_steps=10000
    args.checkpointing_steps=200
    args.learning_rate=1e-05
    args.seed=42
    args.output_dir="output/Fun_1.3b/I2V_HPSReward_H"
    args.gradient_checkpointing
    args.mixed_precision="bf16"
    args.adam_weight_decay=3e-2
    args.adam_epsilon=1e-10
    args.max_grad_norm=0.3
    args.prompt_path="/mnt/vision-gen-ks3/Video_Generation/DataSets/CustomDataSet/humanvid0402/humanvid-h_recaption.jsonl"
    args.train_sample_height=480
    args.train_sample_width=832
    args.num_inference_steps=40
    args.video_length=41
    args.num_decoded_latents=4
    args.reward_fn="HPSReward"
    args.reward_fn_kwargs='{"version": "v2.1"}'
    args.backprop_strategy= "tail"
    args.backprop_num_steps=5
    args.backprop = True
    args.report_to="wandb"

    return args


def main():
    args = parse_args()
    #args = set_parse(args)

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    config = OmegaConf.load(args.config_path)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    
    # Sanity check for backprop
    if args.backprop:
        if args.backprop_step_list is not None:
            logger.warning(
                f"The backprop_strategy {args.backprop_strategy} will be ignored "
                f"when using backprop_step_list {args.backprop_step_list}."
            )
            assert any(step <= args.num_inference_steps - 1 for step in args.backprop_step_list)
        else:
            if args.backprop_strategy in set(["tail", "uniform", "random"]):
                assert args.backprop_num_steps <= args.num_inference_steps - 1
            if args.backprop_strategy == "random":
                assert args.backprop_random_start_step <= args.backprop_random_end_step
                assert args.backprop_random_end_step <= args.num_inference_steps - 1

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed, device_specific=True)
        rng = np.random.default_rng(np.random.PCG64(args.seed + accelerator.process_index))
    else:
        rng = None

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer3d) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Load scheduler, tokenizer and models.
    noise_scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )

    # Get Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
    )

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        # Get Text encoder
        text_encoder = WanT5EncoderModel.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
            additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )
        text_encoder.eval()
        # Get Vae
        vae = AutoencoderKLWan.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, config['vae_kwargs'].get('vae_subpath', 'vae')),
            additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
        )
        vae.eval()

    # Get Transformer
    transformer3d = WanTransformer3DModel.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
    )

    # Get Clip Image Encoder
    clip_image_encoder = CLIPModel.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
    )
    clip_image_encoder.eval()

    # Freeze vae and text_encoder and set transformer3d to trainable
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    transformer3d.requires_grad_(False)
    clip_image_encoder.requires_grad_(False)

    # Lora will work with this...
    network = create_network(
        1.0,
        args.rank,
        args.network_alpha,
        text_encoder,
        transformer3d,
        neuron_dropout=None,
        add_lora_in_attn_temporal=True,
    )
    network.apply_to(text_encoder, transformer3d, args.train_text_encoder and not args.training_with_video_token_length, True)

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                safetensor_save_path = os.path.join(output_dir, f"lora_diffusion_pytorch_model.safetensors")
                save_model(safetensor_save_path, accelerator.unwrap_model(models[-1]))

        accelerator.register_save_state_pre_hook(save_model_hook)
        # Save the model weights directly before save_state instead of using a hook.
        # accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        transformer3d.enable_gradient_checkpointing()
    
    if args.vae_gradient_checkpointing:
        # Since 3D casual VAE need a cache to decode all latents autoregressively, .Thus, gradient checkpointing can only be 
        # enabled when decoding the first batch (i.e. the first three) of latents, in which case the cache is not being used.
        
        # num_decoded_latents > 3 is support in EasyAnimate now.
        # if args.num_decoded_latents > 3:
        #     raise ValueError("The vae_gradient_checkpointing is not supported for num_decoded_latents > 3.")
        vae.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    elif args.use_came:
        try:
            from came_pytorch import CAME
        except:
            raise ImportError(
                "Please install came_pytorch to use CAME. You can do so by running `pip install came_pytorch`"
            )

        optimizer_cls = CAME
    else:
        optimizer_cls = torch.optim.AdamW

    logging.info("Add network parameters")
    trainable_params = list(filter(lambda p: p.requires_grad, network.parameters()))
    trainable_params_optim = network.prepare_optimizer_params(args.learning_rate / 2, args.learning_rate, args.learning_rate)

    # Init optimizer
    if args.use_came:
        optimizer = optimizer_cls(
            trainable_params_optim,
            lr=args.learning_rate,
            betas=(0.9, 0.999, 0.9999), 
            eps=(1e-30, 1e-16)
        )
    else:
        optimizer = optimizer_cls(
            trainable_params_optim,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    
    # loss function
    reward_fn_kwargs = {}
    if args.reward_fn_kwargs is not None:
        reward_fn_kwargs = json.loads(args.reward_fn_kwargs)
    if accelerator.is_main_process:
        # Check if the model is downloaded in the main process.
        loss_fn = getattr(reward_fn, args.reward_fn)(device="cpu", dtype=weight_dtype, **reward_fn_kwargs)
    accelerator.wait_for_everyone()
    loss_fn = getattr(reward_fn, args.reward_fn)(device=accelerator.device, dtype=weight_dtype, **reward_fn_kwargs)

    # Get RL training prompts
    if accelerator.is_main_process:
        prompt_list, video_list = load_prompts_videos(args.prompt_path)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(prompt_list) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    network, optimizer, lr_scheduler = accelerator.prepare(network, optimizer, lr_scheduler)

    # Move text_encode and vae to gpu and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    transformer3d.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device)
    clip_image_encoder.to(accelerator.device, dtype=weight_dtype)
    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(prompt_list) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop("backprop_step_list", None)
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(prompt_list)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            global_step = int(path.split("-")[1])

            initial_global_step = global_step

            from safetensors.torch import load_file, safe_open
            state_dict = load_file(os.path.join(os.path.join(args.output_dir, path), "lora_diffusion_pytorch_model.safetensors"))
            m, u = accelerator.unwrap_model(network).load_state_dict(state_dict, strict=False)
            print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
    else:
        initial_global_step = 0

    # function for saving/removing
    def save_model(ckpt_file, unwrapped_nw):
        os.makedirs(args.output_dir, exist_ok=True)
        accelerator.print(f"\nsaving checkpoint: {ckpt_file}")
        unwrapped_nw.save_weights(ckpt_file, weight_dtype, None)

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        train_reward = 0.0

        # In the following training loop, randomly select training prompts and use the 
        # `EasyAnimatePipelineInpaint` to sample videos, calculate rewards, and update the network.
        for _ in range(num_update_steps_per_epoch):
            random_indices   = random.choices(range(len(prompt_list)), k=args.train_batch_size)
            train_prompt     = [prompt_list[i] for i in random_indices]
            train_video_path = [video_list[i] for i in random_indices]
            train_images     = [process_video_frames(video_path, [args.train_sample_height, args.train_sample_width]) for video_path in train_video_path]
            logger.info(f"train_prompt: {train_prompt}")
            logger.info(f"train_video:  {train_video_path}")

            # default height and width
            height = int(args.train_sample_height // 16 * 16)
            width  = int(args.train_sample_width  // 16 * 16)
            
            # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
            # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
            # corresponds to doing no classifier free guidance.
            do_classifier_free_guidance = args.guidance_scale > 1.0
            
            # Reduce the vram by offload text encoders
            if args.low_vram:
                torch.cuda.empty_cache()
                text_encoder.to(accelerator.device)

            # Encode input prompt
            (
                prompt_embeds,
                negative_prompt_embeds
            ) = encode_prompt(
                tokenizer,
                text_encoder,
                train_prompt,
                negative_prompt=[""] * len(train_prompt),
                device=accelerator.device,
                dtype=weight_dtype,
                do_classifier_free_guidance=do_classifier_free_guidance,
            )
            if do_classifier_free_guidance:
                prompt_embeds = negative_prompt_embeds + prompt_embeds

            # Reduce the vram by offload text encoders
            if args.low_vram:
                text_encoder.to("cpu")
                torch.cuda.empty_cache()

            # Prepare timesteps
            if hasattr(noise_scheduler, "use_dynamic_shifting") and noise_scheduler.use_dynamic_shifting:
                noise_scheduler.set_timesteps(args.num_inference_steps, device=accelerator.device, mu=1)
            else:
                noise_scheduler.set_timesteps(args.num_inference_steps, device=accelerator.device)
            timesteps = noise_scheduler.timesteps

            # Prepare latent variables
            vae_scale_factor = vae.spacial_compression_ratio
            latent_shape = [
                args.train_batch_size,
                vae.config.latent_channels,
                int((args.video_length - 1) // vae.temporal_compression_ratio + 1) if args.video_length != 1 else 1,
                args.train_sample_height // vae_scale_factor,
                args.train_sample_width // vae_scale_factor,
            ]

            with accelerator.accumulate(transformer3d):
                latents = torch.randn(*latent_shape, device=accelerator.device, dtype=weight_dtype)

                # Prepare inpaint latents if it needs.
                # Use zero latents if we want to t2v.
                mask_latents = torch.zeros_like(latents)[:, :4].to(latents.device, latents.dtype)
                masked_video_latents = torch.zeros_like(latents).to(latents.device, latents.dtype)

                mask_input = torch.cat([mask_latents] * 2) if do_classifier_free_guidance else mask_latents
                masked_video_latents_input = (torch.cat([masked_video_latents] * 2) if do_classifier_free_guidance else masked_video_latents)
                inpaint_latents = torch.cat([mask_input, masked_video_latents_input], dim=1).to(latents.dtype)

                if hasattr(noise_scheduler, "init_noise_sigma"):
                    latents = latents * noise_scheduler.init_noise_sigma

                clip_context = []
                for index in range(args.train_batch_size):
                    clip_image    = TF.to_tensor(train_images[index]).sub_(0.5).div_(0.5).to(latents.device, latents.dtype)
                    _clip_context = clip_image_encoder([clip_image[:, None, :, :]])

                    if rng is None:
                        zero_init_clip_in = np.random.choice([True, False], p=[0.1, 0.9])
                    else:
                        zero_init_clip_in = rng.choice([True, False], p=[0.1, 0.9])

                    clip_context.append(_clip_context if not zero_init_clip_in else torch.zeros_like(_clip_context))
                            
                clip_context = torch.cat(clip_context)
                clip_context = ( torch.cat([clip_context] * 2) if do_classifier_free_guidance else clip_context)
                
                generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
                # Prepare extra step kwargs.
                extra_step_kwargs = prepare_extra_step_kwargs(noise_scheduler, generator, args.eta)

                bsz, channel, num_frames, height, width = latents.size()
                target_shape = (vae.latent_channels, num_frames, width, height)
                seq_len = math.ceil(
                    (target_shape[2] * target_shape[3]) /
                    (accelerator.unwrap_model(transformer3d).config.patch_size[1] * accelerator.unwrap_model(transformer3d).config.patch_size[2]) *
                    target_shape[1]
                )

                # Denoising loop
                if args.backprop:
                    if args.backprop_step_list is None:
                        if args.backprop_strategy == "last":
                            backprop_step_list = [args.num_inference_steps - 1]
                        elif args.backprop_strategy == "tail":
                            backprop_step_list = list(range(args.num_inference_steps))[-args.backprop_num_steps:]
                        elif args.backprop_strategy == "uniform":
                            interval = args.num_inference_steps // args.backprop_num_steps
                            random_start = random.randint(0, interval)
                            backprop_step_list = [random_start + i * interval for i in range(args.backprop_num_steps)]
                        elif args.backprop_strategy == "random":
                            backprop_step_list = random.sample(
                                range(args.backprop_random_start_step, args.backprop_random_end_step + 1), args.backprop_num_steps
                            )
                        else:
                            raise ValueError(f"Invalid backprop strategy: {args.backprop_strategy}.")
                    else:
                        backprop_step_list = args.backprop_step_list
                
                for i, t in enumerate(tqdm(timesteps)):
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    if hasattr(noise_scheduler, "scale_model_input"):
                        latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
                    
                    # expand scalar t to 1-D tensor to match the 1st dim of latent_model_input
                    t_expand = torch.tensor([t] * latent_model_input.shape[0], device=accelerator.device).to(
                        dtype=latent_model_input.dtype
                    )

                    # Whether to enable DRTune: https://arxiv.org/abs/2405.00760
                    if args.stop_latent_model_input_gradient:
                        latent_model_input = latent_model_input.detach()

                    # predict noise model_output
                    with torch.cuda.amp.autocast(dtype=weight_dtype):
                    #with accelerator.autocast():
                        noise_pred = transformer3d(
                            x=latent_model_input,
                            context=prompt_embeds,
                            t=t_expand,
                            seq_len=seq_len,
                            y=inpaint_latents,
                            clip_fea=clip_context
                        )

                    # Optimize the denoising results only for the specified steps.
                    if i in backprop_step_list:
                        noise_pred = noise_pred
                    else:
                        noise_pred = noise_pred.detach()

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred[0], noise_pred[1]
                        noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = noise_scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # decode latents (tensor)
                # latents = latents.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
                # Since the casual VAE decoding consumes a large amount of VRAM, and we need to keep the decoding 
                # operation within the computational graph. Thus, we only decode the first args.num_decoded_latents 
                # to calculate the reward.
                # TODO: Decode all latents but keep a portion of the decoding operation within the computational graph.
                sampled_latent_indices = list(range(args.num_decoded_latents))
                sampled_latents = latents[:, :, sampled_latent_indices, :, :]
                sampled_frames = vae.decode(sampled_latents.to(vae.device, vae.dtype)).sample
                sampled_frames = sampled_frames.clamp(-1, 1)
                sampled_frames = (sampled_frames / 2 + 0.5).clamp(0, 1)  # [-1, 1] -> [0, 1]

                if global_step % args.checkpointing_steps == 0:
                    saved_file = f"sample-{global_step}-{accelerator.process_index}.mp4"
                    save_videos_grid(sampled_frames.to(torch.float32).detach().cpu(),
                                     os.path.join(args.output_dir, "train_sample", saved_file), fps=16)
                    train_images[0].save(os.path.join(args.output_dir, "train_sample", f"train_images_{global_step}.png"))
                
                if args.num_sampled_frames is not None:
                    num_frames     = sampled_frames.size(2) - 1
                    sampled_frames_indices = torch.linspace(0, num_frames, steps=args.num_sampled_frames).long()
                    sampled_frames = sampled_frames[:, :, sampled_frames_indices, :, :]
                
                # compute loss and reward
                loss, reward   = loss_fn(sampled_frames, train_prompt)

                # Gather the losses and rewards across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                avg_reward = accelerator.gather(reward.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                train_reward += avg_reward.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    total_norm = accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                    # If use_deepspeed, `total_norm` cannot be logged by accelerator.
                    if not args.use_deepspeed:
                        accelerator.log({"total_norm": total_norm}, step=global_step)
                    else:
                        if hasattr(optimizer, "optimizer") and hasattr(optimizer.optimizer, "_global_grad_norm"):
                            accelerator.log({"total_norm":  optimizer.optimizer._global_grad_norm}, step=global_step)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss, "train_reward": train_reward}, step=global_step)
                train_loss = 0.0
                train_reward = 0.0

                if global_step % args.checkpointing_steps == 0:
                    # DeepSpeed requires saving weights on every device; saving weights only on the main process would cause issues.
                    if args.use_deepspeed or accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)
                        
                        if not args.save_state:
                            safetensor_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.safetensors")
                            save_model(safetensor_save_path, accelerator.unwrap_model(network))
                            logger.info(f"Saved safetensor to {safetensor_save_path}")
                        else:
                            accelerator_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(accelerator_save_path)
                            logger.info(f"Saved state to {accelerator_save_path}")
            
            logs = {"step_loss": loss.detach().item(), "step_reward": reward.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            
            if global_step >= args.max_train_steps:
                break

if __name__ == "__main__":
    main()
