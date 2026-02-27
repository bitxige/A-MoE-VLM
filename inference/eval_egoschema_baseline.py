# -*- coding: utf-8 -*-
"""
Baseline Evaluation Script for EgoSchema
This script evaluates a base Large Language Model (e.g., Qwen) on the EgoSchema dataset.
It uses standard Zero-Shot Prompting and forces the model to generate Chain-of-Thought (CoT) 
reasoning via Chat Template pre-filling.
"""

import argparse
import json
import logging
import re
import os
import sys
import random
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed as hf_set_seed
import warnings

# ==========================================
# 0. Global Setup & Logging
# ==========================================
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout), 
        logging.FileHandler("eval_egoschema_baseline.log", mode='w', encoding='utf-8')
    ]
)
log = logging.getLogger(__name__)

def set_deterministic(seed=3407):
    """Ensure reproducibility across runs."""
    seed = int(seed) % (2**32 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==========================================
# 1. Utilities (EgoSchema Adapted)
# ==========================================
def build_prompt_and_label(item):
    """
    Converts an EgoSchema JSON item into a standard multiple-choice prompt.
    EgoSchema provides options in separate keys (option_a to option_e) and answers as direct letters.
    """
    question = item.get("question", "")
    # EgoSchema 专有：日志字段名为 vlm_analysis
    video_log = item.get("vlm_analysis", "") 
    # EgoSchema 专有：直接提供正确字母
    correct_label = item.get("answer", "").strip().upper() 

    # EgoSchema 专有：选项分散在各个独立字段中
    candidates = [
        item.get("option_a", ""),
        item.get("option_b", ""),
        item.get("option_c", ""),
        item.get("option_d", ""),
        item.get("option_e", "")
    ]
    
    options_str = ""
    for i, candidate in enumerate(candidates):
        if candidate: # 确保选项不为空
            letter = chr(ord('A') + i)
            options_str += f"{letter}. {candidate}\n"

    # 如果数据集里没有答案（极少见），给个默认值防止报错
    if not correct_label:
        correct_label = "UNKNOWN"

    user_content = (
        "You are an AI assistant. Please answer the following multiple-choice question based on the provided video log.\n\n"
        f"Video Log:\n{video_log}\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{options_str}\n"
        "Please reason step-by-step and output your final answer (just the letter) within <answer> and </answer> tags."
    )
    
    return user_content, correct_label

def extract_answer(text):
    """
    Extracts the final answer from the model's output using regex.
    Strictly filters for A, B, C, D, or E to adapt to EgoSchema.
    """
    has_think = "<think>" in text and "</think>" in text
    has_answer = "<answer>" in text and "</answer>" in text
    is_format_correct = has_think and has_answer

    pred = None
    match = re.search(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
    if match:
        raw_pred = match.group(1).strip()
        # EgoSchema 专有：严格限制正则只抓取 A 到 E 的大写字母
        letter_match = re.search(r"[A-E]", raw_pred.upper())
        if letter_match:
            pred = letter_match.group(0)
    
    return is_format_correct, pred

# ==========================================
# 2. Main Evaluation Logic
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Baseline Evaluation for EgoSchema via Textual Logs")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the HuggingFace model")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL dataset")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the detailed prediction results")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum number of tokens to generate")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed for reproducibility")
    args = parser.parse_args()

    log.info(f"🚀 Initializing EgoSchema Baseline Evaluation | Seed: {args.seed}")
    set_deterministic(args.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Model & Tokenizer
    log.info(f"Loading model from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        torch_dtype=torch.float16, 
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    # 2. Load Data
    log.info(f"Loading dataset from: {args.input_file}")
    test_data = []
    if os.path.exists(args.input_file):
        with open(args.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    test_data.append(json.loads(line))
    else:
        log.error(f"Dataset file not found: {args.input_file}")
        return

    correct_count = 0
    format_correct_count = 0
    total = len(test_data)
    results = []

    log.info(f"Starting inference on {total} samples...")

    # 3. Inference Loop
    with torch.no_grad():
        for item in tqdm(test_data, desc="Evaluating"):
            # 解析 EgoSchema 的特有格式
            user_content, label = build_prompt_and_label(item)
            
            messages = [
                {"role": "system", "content": "You are a helpful and logical visual reasoning expert."},
                {"role": "user", "content": user_content}
            ]
            
            # Apply standard chat template
            prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # 强制触发思维链 (Pre-fill)
            prompt += "<think>\n"
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            outputs = model.generate(
                **inputs, 
                max_new_tokens=args.max_new_tokens, 
                do_sample=False, # 贪婪解码 (Zero-shot)
                pad_token_id=tokenizer.eos_token_id
            )
            
            generated_ids = outputs[0][inputs.input_ids.shape[1]:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # 补回 <think> 标签以便解析
            full_response = "<think>\n" + response
            
            is_format_ok, pred = extract_answer(full_response)
            is_correct = (pred == label) if pred and label else False
            
            if is_format_ok: format_correct_count += 1
            if is_correct: correct_count += 1
            
            # 记录结果 (EgoSchema 的唯一标识符通常是 id)
            results.append({
                "id": item.get("id", "unknown"),
                "question": item.get("question", ""),
                "label_letter": label,
                "pred_letter": pred,
                "is_correct": is_correct,
                "format_ok": is_format_ok,
                "model_response": full_response
            })

    # 4. Save Results
    log.info(f"Saving detailed predictions to: {args.output_file}")
    os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else '.', exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

    # 5. Output Metrics Report
    acc = correct_count / total if total > 0 else 0
    format_rate = format_correct_count / total if total > 0 else 0
    
    log.info("=" * 50)
    log.info(f"📊 EgoSchema Baseline Evaluation Report (Total: {total})")
    log.info(f"1. Format Compliance Rate : {format_rate:.2%}")
    log.info(f"2. Reasoning Accuracy     : {acc:.2%}")
    log.info("=" * 50)
    
    # 打印前 2 个例子
    log.info("\n[🔍 Case Study - Top 2]")
    for i in range(min(2, len(results))):
        log.info(f"--- Example {i+1} ---")
        log.info(f"ID: {results[i]['id']}")
        log.info(f"Question: {results[i]['question']}")
        log.info(f"Ground Truth: {results[i]['label_letter']} | Prediction: {results[i]['pred_letter']}")
        log.info(f"Output Snippet: {results[i]['model_response'][:200]}...\n")

if __name__ == "__main__":
    main()