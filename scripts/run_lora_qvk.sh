#!/bin/bash

# Ensure the script exits if any command fails
set -e

# Define the arguments
MODEL_NAME="EleutherAI/pythia-160m"
DTYPE="float32"
USE_FLASH_ATTENTION_2=false
LORA_R=(1)
LORA_ALPHA=(32)
LORA_DROPOUT=(0.1)
TARGET_MODULES=(q_proj v_proj k_proj)
PROJECT_NAME="hidden-capacity-lora-qvk"
MAX_LENGTH=(8 16 32 64 80 96 128 192 256)
NUM_ITERATIONS=5000
NUM_SAMPLES=25
LR=1e-2
WEIGHT_DECAY=0.01

TEXTS_PATH=./data/fanfics_1k_chunks.csv
SAVE_PATH=./runs_lora


# Run the Python script with the arguments
python train.py \
    --model_name "$MODEL_NAME" \
    --project_name "$PROJECT_NAME" \
    --dtype "$DTYPE" \
    $( [ "$USE_FLASH_ATTENTION_2" = true ] && echo "--use_flash_attention_2" ) \
    --lora_r "${LORA_R[@]}" \
    --lora_alpha "${LORA_ALPHA[@]}" \
    --lora_dropout "${LORA_DROPOUT[@]}" \
    --max_length "${MAX_LENGTH[@]}" \
    --num_iterations "$NUM_ITERATIONS" \
    --num_samples "$NUM_SAMPLES" \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --texts_path "$TEXTS_PATH" \
    --save_path "$SAVE_PATH"
