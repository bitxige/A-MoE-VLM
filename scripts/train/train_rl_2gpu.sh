#!/bin/bash
# ==============================================================================
# Reinforcement Learning (GRPO) Training Script for A-MoE Logic-V Alignment
# 
# This script orchestrates a dual-GPU training pipeline:
# - GPU 1: Hosts the 32B Reward Model (Arbiter) via a local FastAPI server.
# - GPU 0: Executes GRPO training on the 3B Base Model using ms-swift.
# ==============================================================================

# Exit immediately if a command exits with a non-zero status
set -e

# ================= Configuration =================
# Paths (Modify these according to your local setup)
BASE_DIR="."
MODEL_PATH="${BASE_DIR}/models/Qwen2.5-3B-Instruct"
TRAIN_DATA="${BASE_DIR}/dataset/complete/train_data_filtered_6144.jsonl"
VAL_DATA="${BASE_DIR}/dataset/complete/val_data_filtered_6144.jsonl"

SERVER_SCRIPT="${BASE_DIR}/train/reward/start_consistency_model_server.py"
OUTPUT_DIR="${BASE_DIR}/results/rl_amoe_logic_v"
LOG_DIR="${BASE_DIR}/train/logs"
SERVER_LOG="${LOG_DIR}/referee_server.log"

# Define System Prompt for GRPO
SYSTEM_PROMPT=$'You are a helpful assistant.
You must first think about the reasoning process in the mind and then provide the user with the answer.
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively.

Here is an example of the required format:
User: Which direction is the sun rising from?
A. West
B. East
C. North
Assistant: <think> The sun rises in the East. Therefore, option B is correct. </think>
<answer> B </answer>

Now, answer the user\'s question following this exact format. Ensure logic consistency.'
# =================================================

# --- Cleanup Trap ---
# Ensures the background server is killed if the script is terminated (e.g., Ctrl+C)
cleanup() {
    echo -e "\n[INFO] Script terminated. Cleaning up background processes..."
    if pgrep -f "$SERVER_SCRIPT" > /dev/null; then
        pkill -f "$SERVER_SCRIPT"
        echo "[INFO] Referee server stopped."
    fi
}
trap cleanup EXIT INT TERM

# --- [STEP 1] Start the Referee Service (GPU 1) ---
start_referee_service() {
    echo "======================================================"
    echo "[STEP 1/3] Initializing Reward Arbiter Server on GPU 1"
    echo "======================================================"

    if pgrep -f "$SERVER_SCRIPT" > /dev/null; then
        echo "✅ Arbiter server is already running."
        return 0
    fi

    mkdir -p "$LOG_DIR"
    > "$SERVER_LOG"

    echo "[INFO] Booting server... (Logs: $SERVER_LOG)"
    # Start server in background on GPU 1
    nohup env CUDA_VISIBLE_DEVICES=1 python "$SERVER_SCRIPT" > "$SERVER_LOG" 2>&1 &
    
    # Health check loop (Timeout: 60s)
    echo -n "[INFO] Waiting for service to become ready "
    for i in {1..30}; do
        if curl -s http://localhost:8000/docs > /dev/null 2>&1; then
            echo -e "\n✅ Arbiter Server is online and ready!"
            return 0 
        fi
        echo -n "."
        sleep 2
    done

    echo -e "\n❌ [ERROR] Server initialization timed out. Check logs: $SERVER_LOG"
    return 1
}

# --- [STEP 2] Execute GRPO Training (GPU 0) ---
main() {
    # 1. Initialize Server
    start_referee_service
    
    echo "======================================================"
    echo "[STEP 2/3] Configuring Training Environment (GPU 0)"
    echo "======================================================"
    
    export CUDA_VISIBLE_DEVICES=0
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export WANDB_MODE=offline      # Keep offline for stability
    export PYTHONUNBUFFERED=1      # Real-time log streaming
    
    mkdir -p "$OUTPUT_DIR"

    echo "======================================================"
    echo "[STEP 3/3] Launching ms-swift GRPO Engine"
    echo "======================================================"

    swift rlhf \
        --rlhf_type grpo \
        --model "$MODEL_PATH" \
        --train_type lora \
        --lora_rank 32 \
        --lora_alpha 64 \
        --torch_dtype bfloat16 \
        --system "${SYSTEM_PROMPT}" \
        --dataset "$TRAIN_DATA" \
        --val_dataset "$VAL_DATA" \
        --dataloader_num_workers 4 \
        --num_train_epochs 12 \
        --learning_rate 5e-6 \
        --warmup_ratio 0.05 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --gradient_checkpointing \
        --beta 0.005 \
        --temperature 0.7 \
        --num_generations 4 \
        --num_iterations 1 \
        --max_length 8192 \
        --max_completion_length 2048 \
        --eval_steps 200 \
        --save_steps 200 \
        --save_only_model true \
        --save_total_limit 200 \
        --output_dir "$OUTPUT_DIR" \
        --logging_steps 1 \
        --report_to wandb \
        --log_completions true \
        --log_level info \
        --reward_weights 7.0 1.0 1.0 1.0 \
        --reward_funcs choice_accuracy strict_format logical_consistency stability \
        --external_plugins train/reward/choice_accuracy_reward.py \
                           train/reward/strict_format_reward.py \
                           train/reward/consistency_reward_local_amoe.py \
                           train/reward/stability_reward.py \
        2>&1 | tee "${OUTPUT_DIR}/grpo_training.log"

    echo "======================================================"
    echo "🎉 GRPO Training Session Completed Successfully!"
    echo "======================================================"
}

# Execute main function
main