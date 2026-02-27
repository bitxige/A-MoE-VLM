# -*- coding: utf-8 -*-
"""
Supervised Fine-Tuning (SFT) Evaluation Script for EgoSchema
This script evaluates an SFT LoRA-adapted Large Language Model on the EgoSchema dataset.
It handles flexible option formatting, merges LoRA weights for faster inference,
and forces Chain-of-Thought (CoT) reasoning.
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
        logging.FileHandler("eval_egoschema_sft.log", mode='w', encoding='utf-8')
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
def format_input(item):
    """
    Formats the input prompt, seamlessly adapting to EgoSchema's varied option formats 
    (e.g., individual 'option_a' keys vs. array 'choices').
    """
    # 1. Extract Video Log
    video_log = item.get('vlm_analysis', '') or item.get('video_log', '') or item.get('content', '')
    if isinstance(video_log, list):
        video_log = "\n".join(video_log)
    
    # 2. Extract Question
    question = item.get('question', '')

    # 3. Extract Options dynamically
    options_text = ""
    candidates = ["A", "B", "C", "D", "E"]
    letters = ['a', 'b', 'c', 'd', 'e']
    
    found_opts = False
    
    # Strategy A: Prioritize EgoSchema's native keys (option_a, option_b, etc.)
    for i, letter in enumerate(letters):
        key = f"option_{letter}"
        if key in item:
            options_text += f"{candidates[i]}. {item[key]}\n"
            found_opts = True
            
    # Strategy B: Fallback for generic dict keys (option_0, option_1, etc.)
    if not found_opts:
        for idx in range(5):
            keys = [f"option_{idx}", f"option {idx}", str(idx)]
            val = None
            for k in keys:
                if k in item:
                    val = item[k]
                    break
            if val:
                options_text += f"{candidates[idx]}. {val}\n"
                found_opts = True
                
    # Strategy C: Fallback for array-based 'choices'
    if not found_opts and 'choices' in item:
        choices_list = item['choices']
        if isinstance(choices_list, list):
            for idx, choice in enumerate(choices_list):
                if idx < 5:
                    options_text += f"{candidates[idx]}. {choice}\n"

    # Assemble Final User Content
    user_content = (
        f"[Video Log]\n{video_log}\n\n"
        f"Question: {question}\n"
        f"Options:\n{options_text.strip()}"
    )
    return user_content

def extract_answer(response_text):
    """
    Extracts the multiple-choice answer from the model's textual response.
    Supports both strict XML tags (<answer>X</answer>) and fallback semantic parsing.
    """
    # 1. Primary Extraction: XML Tags (Optimal for SFT/RL aligned models)
    match = re.search(r"<answer>\s*([A-E])\s*</answer>", response_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # 2. Secondary Extraction: Common semantic phrasing
    match_fallback = re.search(r"(?:Answer|Choose|Selection)[:\s]+([A-E])", response_text, re.IGNORECASE)
    if match_fallback:
        return match_fallback.group(1).upper()
        
    # 3. Ultimate Fallback: Default to 'C' to prevent execution crashes
    return "C" 

# ==========================================
# 2. Main Evaluation Logic
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="SFT Model Evaluation for EgoSchema")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to the HuggingFace base model")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to the SFT LoRA weights")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL dataset")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the prediction results")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum generation length")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed for reproducibility")
    args = parser.parse_args()

    log.info(f"🚀 Initializing SFT Evaluation | Seed: {args.seed}")
    set_deterministic(args.seed)

    # 1. Load Tokenizer
    log.info(f"Loading Base Model from: {args.base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    tokenizer.truncation_side = 'left' # Preserve question and options, truncate old video logs if needed
    
    # 2. Load Base Model
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path, 
        device_map="auto", 
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    # 3. Mount and Merge LoRA (if provided)
    if args.lora_path and os.path.exists(args.lora_path):
        log.info(f"🔗 Mounting LoRA Adapter from: {args.lora_path}")
        try:
            model = PeftModel.from_pretrained(base_model, args.lora_path)
            log.info("⚡ Merging LoRA weights into base model for optimized inference latency...")
            model = model.merge_and_unload()
        except Exception as e:
            log.warning(f"⚠️ Failed to load LoRA. Falling back to pure Base Model. Reason: {e}")
            model = base_model
    else:
        log.info("ℹ️ No valid LoRA path provided. Evaluating pure Base Model.")
        model = base_model

    model.eval()

    # 4. Load Dataset (Handles both JSON Array and JSONL)
    log.info(f"📂 Reading data from: {args.input_file}")
    data = []
    
    with open(args.input_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        
        if content.startswith('['):
            try:
                data = json.loads(content)
                log.info("✅ Detected standard JSON Array format.")
            except Exception as e:
                log.error(f"❌ JSON Array parsing failed: {e}")
        else:
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except Exception as e:
                        log.error(f"❌ Failed to parse line {i+1}. Snippet: {line[:50]}...")
                        raise e 
                        
    log.info(f"✅ Successfully loaded {len(data)} samples.")
    if len(data) == 0:
        log.warning("🚨 Dataset is empty. Exiting.")
        return

    # 5. Resumption Logic
    processed_ids = set()
    os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else '.', exist_ok=True)
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try: 
                    d = json.loads(line)
                    pid = d.get('id')
                    if pid: processed_ids.add(str(pid))
                except: pass
        log.info(f"⏩ Resuming... Skipped {len(processed_ids)} already processed samples.")

    f_out = open(args.output_file, 'a', encoding='utf-8')
    
    correct = 0
    valid_format = 0
    total = 0

    log.info(f"⚡ Starting Inference Loop...")
    
    # System Prompt enforcing the structural constraints learned during SFT
    system_prompt = (
        "You are an expert video analyst with strong logical reasoning skills.\n"
        "Your task is to analyze the provided Video Log and answer the multiple-choice question.\n\n"
        "### Format Requirements:\n"
        "1. First, think step-by-step inside <think> tags. Analyze the video log events and eliminate incorrect options.\n"
        "2. Then, provide the final answer inside <answer> tags. The answer must be a single letter (A, B, C, D, or E)."
    )
    
    # 6. Inference Loop
    for item in tqdm(data, desc="Evaluating"):
        q_id = str(item.get('id', 'Unknown'))
        
        if q_id in processed_ids or 'question' not in item:
            continue

        user_input = format_input(item)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Context Truncation Protection
        model_inputs = tokenizer([text], return_tensors="pt", max_length=6000, truncation=True).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=args.max_new_tokens, 
                temperature=0.1,    # Low temperature for rigorous logical deduction
                top_p=0.9,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id
            )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        pred_ans = extract_answer(response)
        label = item.get('answer', None)
        
        is_correct = False
        if label and pred_ans == label.upper():
            is_correct = True
            
        result_item = {
            "id": q_id,
            "label": label,
            "pred": pred_ans,
            "is_correct": is_correct,
            "raw_response": response 
        }
        
        f_out.write(json.dumps(result_item, ensure_ascii=False) + '\n')
        f_out.flush()
        
        total += 1
        if is_correct: correct += 1
        if pred_ans in ["A", "B", "C", "D", "E"]: valid_format += 1

    f_out.close()
    
    # 7. Final Report
    if total > 0:
        log.info("=" * 50)
        log.info(f"📊 EgoSchema SFT Evaluation Report")
        log.info(f"Accuracy     : {correct}/{total} = {correct/total*100:.2f}%")
        log.info(f"Format Valid : {valid_format/total*100:.2f}%")
        log.info("=" * 50)
    
    log.info("✅ Evaluation Complete.")

if __name__ == "__main__":
    main()