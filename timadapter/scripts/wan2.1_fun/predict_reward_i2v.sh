export CUDA_VISIBLE_DEVICES=0
export MODEL_NAME="Wan2.1-Fun-14B-InP"
export PROMPT_PATH="/mnt/vision-gen-ssd/zhangss/VideoX-Fun/samples/baseline_testcase/55_test_prompt/part4.json"

# sample_size  高,宽
python timadapter/example/wan2.1_fun/predict_i2v.py \
 --model_name "models/Diffusion_Transformer/$MODEL_NAME" \
 --num_inference_steps 40 \
 --video_length 41 \
 --prompt_path $PROMPT_PATH \
 --sample_size 480 832 \
 --lora_weight 0.55 \
 --base_save_path "samples/$MODEL_NAME/I2V" \
 --lora_path  "output/$MODEL_NAME/I2V_HPSReward/checkpoint-700.safetensors" \
 --shift 3 # 480p video 3, 720p video 5

