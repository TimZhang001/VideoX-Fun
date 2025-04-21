export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-Fun-1.3B-InP"
export TRAIN_PROMPT_PATH="/mnt/vision-gen-ks3/Video_Generation/DataSets/CustomDataSet/humanvid0402/humanvid-v_recaption.jsonl"
# Performing validation simultaneously with training will increase time and GPU memory usage.
# export VALIDATION_PROMPT_PATH="VideoX-Fun/asset/prompt/chatgpt_custom_actpred.txt"

export CUDA_VISIBLE_DEVICES=7
accelerate launch --num_processes=1 --mixed_precision="bf16" timadapter/scripts/wan2.1_fun/train_reward_lora.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --rank=32 \
  --network_alpha=16 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=10000 \
  --checkpointing_steps=200 \
  --learning_rate=1e-05 \
  --seed=42 \
  --output_dir="output/I2V_Fun_1.3b_HPSReward" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --max_grad_norm=0.3 \
  --prompt_path=$TRAIN_PROMPT_PATH \
  --train_sample_height=480 \
  --train_sample_width=832 \
  --num_inference_steps=40 \
  --video_length=41 \
  --num_decoded_latents=3 \
  --reward_fn="HPSReward" \
  --reward_fn_kwargs='{"version": "v2.1"}' \
  --backprop_strategy "tail" \
  --backprop_num_steps 3 \
  --backprop \
  --report_to wandb 
  #--vae_gradient_checkpointing 