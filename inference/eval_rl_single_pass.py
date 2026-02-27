# -*- coding: utf-8 -*-
"""
Single-Pass RL Expert Evaluation Script for EgoSchema
This script evaluates the single 3B model optimized via GRPO (Logic-V).
It employs greedy decoding (do_sample=False) to test the intrinsic, 
un-ensembled reasoning capability of the aligned agent.
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
from peft import PeftModel
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
        logging.FileHandler("eval_rl_single_pass.log", mode='w', encoding='utf-8')
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
# 1. Utilities
# ==========================================
def build_prompt_dynamic(sample: dict, tokenizer, max_seq_length: int) -> str:
    """
    Constructs the prompt for the RL-aligned agent, adapting to EgoSchema's format.
    Includes context truncation to prevent OOM errors.
    """
    vlm_log = sample.get('vlm_analysis', '').strip()
    context_text = f"[Video Log]\n{vlm_log}"
    
    question_text = sample.get('question', '')
    
    candidates = [
        sample.get('option_a', ''),
        sample.get('option_b', ''),
        sample.get('option_c', ''),
        sample.get('option_d', ''),
        sample.get('option_e', '')
    ]
    
    options_text = ""
    for i, cand in enumerate(candidates):
        if cand: 
            letter = chr(ord('A') + i)
            options_text += f"{letter}. {cand}\n"
    options_text = options_text.strip()

    # System prompt aligned with the Logic-V training phase
    prompt = (
        "You are an expert video analyst with strong logical reasoning skills.\n"
        "Your task is to analyze the provided Video Log and answer the multiple-choice question.\n\n"
        "### Format Requirements:\n"
        "1. First, think step-by-step inside <think> tags. Analyze the video log events and eliminate incorrect options.\n"
        "2. Then, provide the final answer inside <answer> tags. The answer must be a single letter (A, B, C, D, or E).\n\n"
        "### Example:\n"
        "User: \n"
        "[Video Log] ...\n"
        "Question: What is the person holding?\n"
        "Options:\n"
        "A. A cup\n"
        "B. A book\n"
        "C. A phone\n"
        "D. A pen\n"
        "E. A key\n\n"
        "Assistant: <think> The log mentions 'drinking water', which implies a container. A cup is the most logical object for this action. Options B, C, D, E are unrelated to drinking. </think>\n"
        "<answer> A </answer>\n\n"
        "### Now it's your turn:\n"
        f"User:\n{context_text}\n\n"
        f"Question: {question_text}\n"
        f"Options:\n{options_text}\n\n"
        "Assistant:"
    )
    
    # Pre-tokenize to check length and trigger warning if truncation will occur
    total_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    if total_tokens > max_seq_length:
        log.warning(f"Video {sample.get('id', 'Unknown')}: Prompt length ({total_tokens}) exceeds max_seq_length ({max_seq_length}). Left-side truncation will occur.")

    return prompt

def extract_and_clean(text):
    """
    Extracts the final answer using robust regex patterns.
    Optimized to handle edge cases where the RL agent might deviate slightly from strict formatting.
    """
    pred_answer = None
    clean_text = text
    
    match_tag = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if match_tag:
        content = match_tag.group(1).strip()
        m = re.search(r"([A-E])", content, re.IGNORECASE)
        if m: pred_answer = m.group(1).upper()

    if not pred_answer:
        patterns = [
            r"correct answer is[:\s\.]*([A-E])",
            r"correct choice is[:\s\.]*([A-E])",
            r"answer is[:\s\.]*([A-E])",
            r"Answer[:\s\.]*([A-E])",
            r"appropriate choice is[:\s\.]*([A-E])", 
            r"best choice is[:\s\.]*([A-E])",
            r"choice[:\s\.]*([A-E])",
            r"Option[:\s\.]*([A-E])",
            r"([A-E])\s+is the correct",
        ]
        
        for p in patterns:
            matches = list(re.finditer(p, text, re.IGNORECASE))
            if matches:
                pred_answer = matches[-1].group(1).upper()
                break

    if pred_answer:
        last_keyword_idx = -1
        keywords = ["answer is", "choice is", "Option", "<answer>", "appropriate choice is"]
        for kw in keywords:
            idx = text.rfind(kw)
            if idx > last_keyword_idx: last_keyword_idx = idx
        
        if last_keyword_idx != -1:
            cutoff_idx = min(len(text), last_keyword_idx + 100)
            clean_text = text[:cutoff_idx] + "\n\n[Generation Cleaned]"
            
    return pred_answer, clean_text

# ==========================================
# 2. Main Evaluation Logic
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Single-Pass Evaluation for the RL-aligned Agent")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to the HuggingFace base model")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the RL LoRA weights")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL dataset")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the prediction results")
    parser.add_argument("--max_seq_length", type=int, default=6000, help="Maximum allowed context window")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Maximum generation length")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed for reproducibility")
    args = parser.parse_args()

    log.info(f"🚀 Initializing RL Single-Pass Evaluation | Seed: {args.seed}")
    set_deterministic(args.seed)

    # 1. Load Tokenizer
    log.info(f"Loading Base Model from: {args.base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    tokenizer.truncation_side = 'left' 

    # 2. Load Model
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path, 
        torch_dtype=torch.float16, 
        device_map="auto", 
        trust_remote_code=True
    )

    # 3. Mount LoRA
    if args.model_path and os.path.exists(args.model_path):
        log.info(f"🔗 Mounting RL LoRA Adapter from: {args.model_path}")
        try:
            model = PeftModel.from_pretrained(base_model, args.model_path)
            log.info("⚡ LoRA Adapter mounted successfully.")
        except Exception as e:
            log.error(f"Failed to load LoRA: {e}")
            sys.exit(1)
    else:
        log.warning("⚠️ No valid LoRA path provided. Running pure Base Model.")
        model = base_model

    model.eval()

    # 4. Load Dataset
    log.info(f"📂 Reading data from: {args.input_file}")
    data = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        try: 
            data = json.load(f)
        except: 
            f.seek(0)
            data = [json.loads(line) for line in f if line.strip()]

    # 5. Resumption Logic
    processed_ids = set()
    os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else '.', exist_ok=True)
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try: 
                    d = json.loads(line)
                    pid = d.get('id')
                    if pid and pid != "N/A": processed_ids.add(str(pid))
                except: pass
        log.info(f"⏩ Resuming... Skipped {len(processed_ids)} already processed samples.")

    f_out = open(args.output_file, 'a', encoding='utf-8')
    
    correct = 0
    valid_format = 0
    total = 0

    log.info("⚡ Starting Greedy Inference Loop...")
    
    # 6. Inference Loop
    for idx, item in enumerate(tqdm(data, desc="Evaluating")):
        raw_id = item.get('id')
        item_id = str(raw_id) if raw_id is not None else f"Line_{idx}"

        if item_id in processed_ids: 
            continue
        
        prompt = build_prompt_dynamic(item, tokenizer, args.max_seq_length)
        
        # Apply truncation explicitly to safeguard against OOM
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=args.max_seq_length
        ).to(model.device)
        
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=args.max_new_tokens, 
                    do_sample=False,  # Strict greedy decoding for deterministic single-pass eval
                    pad_token_id=tokenizer.eos_token_id
                )
            
            new_tokens = outputs[0][inputs.input_ids.shape[1]:]
            raw_response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # Reconstruct the response with <think> tag if the model started generating immediately
            full_raw_text = "<think>\n" + raw_response if not raw_response.startswith("<think>") else raw_response
            
            pred, clean_text = extract_and_clean(full_raw_text)
            
            correct_letter = item.get('answer', '').strip().upper()
            is_correct = (pred.upper() == correct_letter) if pred and correct_letter else False
            
            target_text = item.get(f'option_{correct_letter.lower()}', '') if correct_letter else ''
            
            res = {
                "id": item_id,
                "target_text": target_text,
                "correct_letter": correct_letter,
                "pred_letter": pred,
                "is_correct": is_correct,
                "response": raw_response 
            }
            
            f_out.write(json.dumps(res, ensure_ascii=False) + "\n")
            f_out.flush() 
            
            total += 1
            if is_correct: correct += 1
            if pred: valid_format += 1
            
        except Exception as e:
            log.error(f"Error processing item {item_id}: {e}")
            torch.cuda.empty_cache()

    f_out.close()
    
    # 7. Final Report
    if total > 0:
        log.info("=" * 50)
        log.info(f"📊 RL Single-Pass Evaluation Report")
        log.info(f"Accuracy     : {correct}/{total} = {correct/total*100:.2f}%")
        log.info(f"Format Valid : {valid_format/total*100:.2f}%")
        log.info("=" * 50)
        
    log.info("✅ Evaluation Complete.")

if __name__ == "__main__":
    main()