
# 1.3B 单卡训练
export MODEL_NAME="Wan2.1-Fun-V1.1-1.3B-In"
export MODEL_PATH="models/Diffusion_Transformer/$MODEL_NAME"
export TRAIN_PROMPT_PATH="/mnt/vision-gen-ks3/Video_Generation/DataSets/CustomDataSet/humanvid0402/humanvid-h_recaption.jsonl"
export BACKPROP_NUM_STEPS=4
export CUDA_VISIBLE_DEVICES=1
export HF_ENDPOINT="https://hf-mirror.com"
export DATATYPE="bf16" # bf16, fp16, no

accelerate launch --num_processes=1 --mixed_precision=$DATATYPE timadapter/scripts/wan2.1_fun/train_reward_lora_i2v.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_PATH \
  --rank=32 \
  --network_alpha=16 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=10000 \
  --checkpointing_steps=200 \
  --learning_rate=1e-05 \
  --seed=42 \
  --output_dir="output/$MODEL_NAME/I2V_HPSReward" \
  --gradient_checkpointing \
  --mixed_precision=$DATATYPE \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --max_grad_norm=0.3 \
  --prompt_path=$TRAIN_PROMPT_PATH \
  --train_sample_height=480 \
  --train_sample_width=832 \
  --num_inference_steps=40 \
  --video_length=41 \
  --num_decoded_latents=2 \
  --reward_fn="HPSReward" \
  --reward_fn_kwargs='{"version": "v2.1"}' \
  --backprop_strategy "tail" \
  --backprop_num_steps $BACKPROP_NUM_STEPS \
  --backprop 
  #--report_to wandb \
  #--tracker_project_name="$MODEL_NAME"
  #--vae_gradient_checkpointing 