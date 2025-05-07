export CUDA_VISIBLE_DEVICES=7
export PROMPT_PATH="/mnt/vision-gen-ssd/zhangss/VideoX-Fun/samples/baseline_testcase/55_test_prompt/part4.json"

# sample_size 高度 宽度 
python timadapter/example/wan2.1/predict_t2v.py \
 --model_name "models/Diffusion_Transformer/Wan2.1-T2V-1.3B" \
 --num_inference_steps 40 \
 --prompt_path $PROMPT_PATH \
 --base_save_path "samples/T2V" \
 --sample_size 480 832 \
 --video_length 81 \
 --lora_path  "output_dir_HPSReward/checkpoint-2300.safetensors" \
 --shift 3 # 480p video 3, 720p video 5

 