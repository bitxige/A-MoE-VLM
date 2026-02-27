# Evaluate A-MoE on EgoSchema using Qwen-3B core
python eval_amoe.py \
    --base_model_path "Qwen/Qwen2.5-3B-Instruct" \
    --model_path "./your_lora_weights" \
    --input_file "data/egoschema_subset.jsonl" \
    --output_file "results/amoe_egoschema.jsonl" \
    --arbiter_api_url "http://localhost:8000/v1/chat/completions"


# Evaluate a single Generalist expert utilizing Self-Consistency on EgoSchema using Qwen-3B core
python eval_baseline_sc_tiebreak.py \
    --base_model_path "Qwen/Qwen2.5-3B-Instruct" \
    --model_path "./your_lora_weights" \
    --input_file "data/egoschema_subset.jsonl" \
    --output_file "results/eval_baseline_sc_tiebreak_egoschema.jsonl" \
    --arbiter_api_url "http://localhost:8000/v1/chat/completions"


# Example Run Baseline for SFT Model Evaluation
python eval_egoschema_baseline.py \
    --model_path "Qwen/Qwen2.5-7B-Instruct" \
    --input_file "data/egoschema_subset.jsonl" \
    --output_file "results/7B-egoschema_predictions.jsonl" \
    --max_new_tokens 2048


# Example Run Command for SFT Model Evaluation
python eval_sft_egoschema.py \
    --base_model_path "Qwen/Qwen2.5-3B-Instruct" \
    --lora_path "checkpoints/Qwen2.5-3B-Video-SFT/checkpoint-504" \
    --input_file "data/egoschema_subset.jsonl" \
    --output_file "results/sft_egoschema_predictions.jsonl" \
    --max_new_tokens 1024 \
    --seed 3407


python extract_egoschema_logs_72B.py \
    --model_path "Qwen/Qwen2.5-VL-72B-Instruct-AWQ" \
    --video_dir "dataset/videos" \
    --input_json "dataset/egoschema.json" \
    --output_file "results/72b_logs.jsonl" \
    --max_frames 24