export CUDA_VISIBLE_DEVICES=7
python timadapter/example/wan2.1/predict_t2v.py \
 --model_name "models/Diffusion_Transformer/Wan2.1-T2V-1.3B" \
 --num_inference_steps 40 \
 --prompt_path "asset/prompt/chatgpt_custom_human_activity.txt" \
 --base_save_path "samples" \
 --lora_path  "output_dir_HPSReward/checkpoint-2300.safetensors" 
 #--lora_path  "output_dir_AestheticReward/checkpoint-2200.safetensors"

 