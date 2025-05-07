import os
import sys
import argparse
import json

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoTokenizer

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

sys.path.append("/mnt/vision-gen-ssd/zhangss/VideoX-Fun")

from videox_fun.dist import set_multi_gpus_devices, shard_model
from videox_fun.models import (AutoencoderKLWan, WanT5EncoderModel, AutoTokenizer,
                              WanTransformer3DModel)
from videox_fun.models.cache_utils import get_teacache_coefficients
from videox_fun.pipeline import WanPipeline
from videox_fun.utils.fp8_optimization import (convert_model_weight_to_float8, replace_parameters_by_name,
                                              convert_weight_dtype_wrapper)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import (filter_kwargs, get_image_to_video_latent,
                                   save_videos_grid)
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

def parse_args():
    parser = argparse.ArgumentParser(description="Video Generation Script")
    
    # GPU和内存配置
    parser.add_argument("--GPU_memory_mode", type=str, default="model_full_load",
                       choices=["model_full_load", "model_cpu_offload", 
                               "model_cpu_offload_and_qfloat8", "sequential_cpu_offload"],
                       help="GPU内存优化模式")
    # # Multi GPUs config
    # Please ensure that the product of ulysses_degree and ring_degree equals the number of GPUs used. For example, if you are using 8 GPUs, you can set ulysses_degree = 2 and ring_degree = 4.
    # If you are using 1 GPU, you can set ulysses_degree = 1 and ring_degree = 1.
    parser.add_argument("--ulysses_degree", type=int, default=1, help="多GPU Ulysses并行度")
    parser.add_argument("--ring_degree",    type=int, default=1, help="多GPU Ring并行度")
    parser.add_argument("--fsdp_dit",       type=bool, default=False, help="Use FSDP to save more GPU memory in multi gpus.")
    parser.add_argument("--compile_dit",    type=bool, default=False, help="Compile will give a speedup in fixed resolution and need a little GPU memory.")

    # TeaCache配置
    parser.add_argument("--enable_teacache", action="store_true", help="启用TeaCache（默认启用）")
    
    # # Recommended to be set between 0.05 and 0.20. A larger threshold can cache more steps, speeding up the inference process, 
    # but it may cause slight differences between the generated content and the original content.
    parser.add_argument("--teacache_threshold", type=float, default=0.10,  help="TeaCache缓存阈值")
    
    # The number of steps to skip TeaCache at the beginning of the inference process, which can reduce the impact of TeaCache on generated video quality.
    parser.add_argument("--num_skip_start_steps", type=int, default=5, help="跳过的初始推理步数")
    parser.add_argument("--teacache_offload", action="store_true",  help="将TeaCache张量卸载到CPU")
    
    parser.add_argument("--cfg_skip_ratio", type=int,  default=0,     help="Skip some cfg steps in inference")
    parser.add_argument("--enable_riflex",  type=bool, default=False, help="将TeaCache张量卸载到CPU")
    parser.add_argument("--riflex_k",       type=int,  default=6,     help="将TeaCache张量卸载到CPU")
    
    
    # 模型路径
    parser.add_argument("--config_path", type=str, default="config/wan2.1/wan_civitai.yaml", help="配置文件路径")
    parser.add_argument("--model_name",  type=str, default="models/Diffusion_Transformer/Wan2.1-T2V-1.3B", help="主模型路径")
    parser.add_argument("--transformer_path", type=str, default=None, help="Transformer检查点路径")
    parser.add_argument("--vae_path",    type=str, default=None, help="VAE检查点路径")
    parser.add_argument("--lora_path",   type=str, default=None, help="LoRA模型路径")
    
    # 生成参数
    parser.add_argument("--sample_size",  type=int, nargs=2, default=[480,832], help="生成尺寸（高,宽）")
    parser.add_argument("--video_length", type=int, default=81, help="视频长度（帧数）")
    parser.add_argument("--fps",          type=int, default=16, help="输出视频帧率")
    parser.add_argument("--seed",         type=int, default=43, help="随机种子")
    parser.add_argument("--num_inference_steps", type=int, default=40, help="推理步数")
    parser.add_argument("--lora_weight",  type=float, default=0.55, help="LoRA权重")
    
    parser.add_argument("--prompt_path",     type=str, default="asset/prompt/chatgpt_custom_human_activity.txt", help="生成提示语")
    parser.add_argument("--negative_prompt", type=str, default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走", help="负面提示语")
    parser.add_argument("--guidance_scale",  type=float, default=6.0, help="指导系数")
    parser.add_argument("--sampler_name",    type=str, default="Flow_Unipc", help="Choose the sampler in Flow")
    parser.add_argument("--shift",           type=int, default=3, help="480p video 3, 720p video 5")

    # 输出配置
    parser.add_argument("--base_save_path",  type=str, default="samples", help="输出基础路径")
    
    parser.set_defaults(enable_teacache=True)
    return parser.parse_args()

def save_results(sample, save_path, video_length, fps, sample_num):
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    suffix = str(sample_num).zfill(4)
    if video_length == 1:
        video_path = os.path.join(save_path, suffix + ".png")

        image = sample[0, :, 0]
        image = image.transpose(0, 1).transpose(1, 2)
        image = (image * 255).numpy().astype(np.uint8)
        image = Image.fromarray(image)
        image.save(video_path)
    else:
        video_path = os.path.join(save_path, suffix + ".mp4")
        save_videos_grid(sample, video_path, fps=fps)

def load_prompts(prompt_path, prompt_column="prompt", start_idx=None, end_idx=None):
    prompt_list = []
    if prompt_path.endswith(".txt"):
        with open(prompt_path, "r") as f:
            for line in f:
                prompt_list.append(line.strip())
    elif prompt_path.endswith(".jsonl"):
        with open(prompt_path, "r") as f:
            for line in f.readlines():
                item = json.loads(line)
                prompt_list.append(item[prompt_column])
    else:
        raise ValueError("The prompt_path must end with .txt or .jsonl.")
    prompt_list = prompt_list[start_idx:end_idx]

    return prompt_list

def load_model(args, config, device):
    # 模型加载
    weight_dtype = torch.bfloat16
    transformer  = WanTransformer3DModel.from_pretrained(
        os.path.join(args.model_name, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
        transformer_additional_kwargs = OmegaConf.to_container(config['transformer_additional_kwargs']),
        low_cpu_mem_usage=True if args.fsdp_dit else False,
        torch_dtype=weight_dtype,)

    if args.transformer_path is not None:
        print(f"From checkpoint: {args.transformer_path}")
        if args.transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(args.transformer_path)
        else:
            state_dict = torch.load(args.transformer_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = transformer.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

    # Get Vae
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(args.model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
    ).to(weight_dtype)

    if args.vae_path is not None:
        print(f"From checkpoint: {args.vae_path}")
        if args.vae_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(args.vae_path)
        else:
            state_dict = torch.load(args.vae_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = vae.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

    # Get Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
    )

    # Get Text encoder
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(args.model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
        additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )
    text_encoder = text_encoder.eval()

    # Get Scheduler
    Choosen_Scheduler = scheduler_dict = {
        "Flow": FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler,}[args.sampler_name]
    if args.sampler_name == "Flow_Unipc" or args.sampler_name == "Flow_DPM++":
        config['scheduler_kwargs']['shift'] = 1
    scheduler = Choosen_Scheduler(
        **filter_kwargs(Choosen_Scheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )

    # Get Pipeline
    pipeline = WanPipeline(transformer=transformer,
                           vae=vae,
                           tokenizer=tokenizer,
                           text_encoder=text_encoder,
                           scheduler=scheduler,)
    
    if args.ulysses_degree > 1 or args.ring_degree > 1:
        from functools import partial
        transformer.enable_multi_gpus_inference()
        if args.fsdp_dit:
            shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
            pipeline.transformer = shard_fn(pipeline.transformer)
            print("Add FSDP")

    if args.compile_dit:
        for i in range(len(pipeline.transformer.blocks)):
            pipeline.transformer.blocks[i] = torch.compile(pipeline.transformer.blocks[i])
        print("Add Compile")

    if args.GPU_memory_mode == "sequential_cpu_offload":
        replace_parameters_by_name(transformer, ["modulation",], device=device)
        transformer.freqs = transformer.freqs.to(device=device)
        pipeline.enable_sequential_cpu_offload(device=device)
    elif args.GPU_memory_mode == "model_cpu_offload_and_qfloat8":
        convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",])
        convert_weight_dtype_wrapper(transformer, weight_dtype)
        pipeline.enable_model_cpu_offload(device=device)
    elif args.GPU_memory_mode == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=device)
    elif args.GPU_memory_mode == "model_full_load_and_qfloat8":
        convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",])
        convert_weight_dtype_wrapper(transformer, weight_dtype)
        pipeline.to(device=device)
    else:
        pipeline.to(device=device)

    coefficients = get_teacache_coefficients(args.model_name) if args.enable_teacache else None
    if coefficients is not None:
        print(f"Enable TeaCache with threshold {args.teacache_threshold} and skip the first {args.num_skip_start_steps} steps.")
        pipeline.transformer.enable_teacache(
            coefficients, args.num_inference_steps, args.teacache_threshold, num_skip_start_steps=args.num_skip_start_steps, offload=args.teacache_offload
        )

    generator = torch.Generator(device=device).manual_seed(args.seed)

    if args.lora_path is not None:
        pipeline = merge_lora(pipeline, args.lora_path, args.lora_weight, device=device)

    return pipeline, generator, vae, config

def main(args):

    device = set_multi_gpus_devices(args.ulysses_degree, args.ring_degree)
    config = OmegaConf.load(args.config_path)
    
    # 动态生成输出路径 
    # "models/Diffusion_Transformer/Wan2.1-T2V-1.3B" -> Wan2.1-T2V-1.3B
    model_base_name = os.path.basename(args.model_name)
    if args.lora_path:
        lora_name = os.path.splitext(os.path.basename(args.lora_path))[0]
        save_path = os.path.join(args.base_save_path, model_base_name, lora_name)
    else:
        save_path = os.path.join(args.base_save_path, model_base_name, "nolora")
    print("save path is " + save_path)
    os.makedirs(save_path, exist_ok=True)    
    
    # 模型加载
    pipeline, generator, vae, config = load_model(args, config, device)
    prompt_list = load_prompts(args.prompt_path)
    
    # 实际生成逻辑（保持原有流程，将硬编码变量替换为args参数）
    with torch.no_grad():
        video_length  = int((args.video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if args.video_length != 1 else 1
        latent_frames = (video_length - 1) // vae.config.temporal_compression_ratio + 1
        sample_num    = 0

        if args.enable_riflex:
            pipeline.transformer.enable_riflex(k = args.riflex_k, L_test = latent_frames)

        for prompt in prompt_list:
            print("prompt is: " + prompt + "\n")
            sample = pipeline(prompt, 
                              num_frames = video_length,
                              negative_prompt = args.negative_prompt,
                              height      = args.sample_size[0],
                              width       = args.sample_size[1],
                              generator   = generator,
                              guidance_scale = args.guidance_scale,
                              num_inference_steps = args.num_inference_steps,
                              cfg_skip_ratio = args.cfg_skip_ratio,
                              shift = args.shift,).videos
                

            if args.ulysses_degree * args.ring_degree > 1:
                import torch.distributed as dist
                if dist.get_rank() == 0:
                    save_results(sample, save_path, args.video_length, args.fps, sample_num)
            else:
                save_results(sample, save_path, args.video_length,  args.fps, sample_num)
            sample_num += 1

        if args.lora_path is not None:
            pipeline = unmerge_lora(pipeline, args.lora_path, args.lora_weight, device=device)

if __name__ == "__main__":
    args = parse_args()
    main(args)



