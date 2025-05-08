
export MODEL_NAME="Wan2.1-Fun-V1.1-1.3B-InP"
export OUTPUT_DIR="output/$MODEL_NAME/I2V_HPSReward" 
python timadapter/scripts/zero_to_bf16.py $OUTPUT_DIR/checkpoint-700 $OUTPUT_DIR/checkpoint-700-outputs \
    --max_shard_size 80GB \
    --safe_serialization