export CUDA_VISIBLE_DEVICES=5
python VideoX-Fun/timadapter/example/wan2.1/predict_t2v.py \
 --model_name "models/Diffusion_Transformer/Wan2.1-T2V-1.3B" \
 --lora_path  "" \
 --num_inference_steps 40 \
 --prompt_path "asset/prompt/chatgpt_custom_human_activity.txt" \
 --base_save_path "samples"

