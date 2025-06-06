# Wan2.1-Fun-I2V-14B 多卡训练

export MODEL_NAME="Wan2.1-Fun-V1.1-1.3B-InP"
export MODEL_PATH="models/Diffusion_Transformer/$MODEL_NAME"
export TRAIN_PROMPT_PATH="/mnt/vision-gen-ks3/Video_Generation/DataSets/CustomDataSet/humanvid0402/humanvid-h_recaption.jsonl"
export BACKPROP_NUM_STEPS=2
export HF_ENDPOINT="https://hf-mirror.com"
export DATATYPE="bf16" # bf16, fp16, no
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

# 480x832 360x560
accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard timadapter/scripts/wan2.1_fun/train_reward_lora_i2v_deepspeed.py \
   --config_path="config/wan2.1/wan_civitai.yaml" \
   --pretrained_model_name_or_path=$MODEL_PATH \
   --rank=32 \
   --network_alpha=16 \
   --train_batch_size=1 \
   --gradient_accumulation_steps=1 \
   --dataloader_num_workers=8 \
   --max_train_steps=10000 \
   --checkpointing_steps=100 \
   --learning_rate=1e-05 \
   --seed=42 \
   --output_dir="output/$MODEL_NAME/I2V_HPSReward" \
   --gradient_checkpointing \
   --mixed_precision=$DATATYPE \
   --adam_weight_decay=3e-2 \
   --adam_epsilon=1e-10 \
   --max_grad_norm=0.3 \
   --prompt_path=$TRAIN_PROMPT_PATH \
   --train_sample_height=480  \
   --train_sample_width=832 \
   --video_length=41 \
   --num_inference_steps=40 \
   --num_decoded_latents=2 \
   --reward_fn="HPSReward" \
   --reward_fn_kwargs='{"version": "v2.1"}' \
   --backprop_strategy="tail" \
   --backprop_num_steps=$BACKPROP_NUM_STEPS \
   --backprop \
   --use_deepspeed \
   --low_vram \
   --save_state \
   --tracker_project_name "i2v-$MODEL_NAME-fine-tune" \
   --report_to wandb 