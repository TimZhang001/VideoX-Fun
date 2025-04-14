export CUDA_VISIBLE_DEVICES=7
python timadapter/example/wan2.1_fun/predict_t2v.py \
 --model_name "models/Diffusion_Transformer/Wan2.1-Fun-1.3B-InP" \
 --num_inference_steps 40 \
 --prompt_path "asset/prompt/chatgpt_custom_human_activity.txt" \
 --base_save_path "samples" \
 --lora_path  "models/alibaba-pai/Wan2.1-Fun-Reward-LoRAs/Wan2.1-Fun-1.3B-InP-MPS.safetensors" 
 #--lora_path  "models/alibaba-pai/Wan2.1-Fun-Reward-LoRAs/Wan2.1-Fun-1.3B-InP-HPS2.1.safetensors" 

