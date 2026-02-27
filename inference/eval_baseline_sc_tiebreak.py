# -*- coding: utf-8 -*-
"""
Baseline Inference Script: Self-Consistency with Arbiter Tie-Break (Group 3)
This script evaluates a single Generalist expert utilizing Self-Consistency (N votes).
If a strict tie occurs among the top predictions, it escalates to the Arbiter API for resolution.
"""

import argparse
import json
import logging
import re
import os
import sys
import random
import time
import requests
import numpy as np
import torch
from collections import defaultdict, Counter
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed as hf_set_seed
from peft import PeftModel
import transformers
import warnings

# ==========================================
# 0. Global Setup
# ==========================================
MAX_CONTEXT_WINDOW = 8192
transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")

os.environ['no_proxy'] = 'localhost,127.0.0.1'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'

API_HEADERS = {"Content-Type": "application/json"}

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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("inference_baseline_sc.log", mode='w', encoding='utf-8')]
)
log = logging.getLogger(__name__)

# ==========================================
# 1. Utilities
# ==========================================
class SmartBudgetManager:
    """Manages the token budget for long video logs to prevent out-of-memory errors."""
    def __init__(self, tokenizer, max_seq_len=8192):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.input_limit = max_seq_len - 2048 - 600 

    def get_context(self, full_text, fixed_count=32):
        pattern = r"(\[?Frame\s*\d+\s*[:\.]?.*?)(?=\[?Frame\s*\d+\s*[:\.]?|\Z|>)"
        all_frames = re.findall(pattern, full_text, flags=re.DOTALL | re.IGNORECASE)
        
        if not all_frames: 
            tokens = self.tokenizer.encode(full_text, add_special_tokens=False)
            if len(tokens) > self.input_limit:
                 return self.tokenizer.decode(tokens[:self.input_limit])
            return full_text
            
        total_frames = len(all_frames)
        if total_frames <= fixed_count:
            ctx = "\n".join(all_frames)
        else:
            indices = np.linspace(0, total_frames - 1, fixed_count, dtype=int)
            indices = sorted(list(set(indices)))
            ctx = "\n".join([all_frames[i] for i in indices])

        tokens = self.tokenizer.encode(ctx, add_special_tokens=False)
        if len(tokens) <= self.input_limit:
            return ctx
        else:
            return self.tokenizer.decode(tokens[:self.input_limit])

def extract_answer_and_reasoning(text):
    """Extracts the final answer (A-E) and the reasoning chain from the model output."""
    reasoning = text
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    if think_match:
        reasoning = think_match.group(1).strip()
    
    match_tag = re.search(r"<answer>.*?([A-E]).*?</answer>", text, re.DOTALL | re.IGNORECASE)
    if match_tag:
        return match_tag.group(1).upper(), reasoning
    
    patterns = [
        r"correct answer is[:\s\.]*([A-E])\b",
        r"correct choice is[:\s\.]*([A-E])\b",
        r"answer is[:\s\.]*([A-E])\b",
        r"Answer[:\s\.]*([A-E])\b",
        r"Option[:\s\.]*([A-E])\b",
        r"choice[:\s\.]*([A-E])\b",
        r"([A-E])\s+is the correct",
        r"select option[:\s\.]*([A-E])\b",
        r"Therefore,\s*([A-E])\b",
        r"So,\s*([A-E])\b"
    ]
    for p in patterns:
        matches = list(re.finditer(p, text, re.IGNORECASE))
        if matches:
            return matches[-1].group(1).upper(), reasoning
            
    cleaned_end = text.strip()[-100:] 
    last_char_match = list(re.finditer(r"(?:^|[^a-zA-Z])([A-E])(?:$|[^a-zA-Z])", cleaned_end))
    if last_char_match:
        return last_char_match[-1].group(1).upper(), reasoning
        
    return None, reasoning

# ==========================================
# 2. Baseline Prompts & Arbiter API
# ==========================================
class BaselineLogic:
    """Contains prompt generation and API call logic for the baseline execution."""
    
    @staticmethod
    def get_user_prompt(context_text, question_text, options_text):
        return (
            "You are an expert video analyst with strong logical reasoning skills.\n"
            "Your task is to analyze the provided Video Log and answer the multiple-choice question.\n\n"
            
            "### 🛑 CRITICAL REASONING RULES (READ CAREFULLY):\n"
            "1. **Specificity Over Generality**: If Option A is 'Cleaning the kitchen' (Broad) and Option B is 'Washing dishes' (Specific), and the video ONLY shows washing dishes, **CHOOSE B**.\n"
            "2. **Cycle Completion (For Manual Tasks)**: In cyclical manual labor (e.g., making bricks, cooking), if the log shows the START of a cycle (e.g., smoothing sand) but cuts off before the END, **assume the cycle completes**. Prefer options that describe the FULL cycle (including cleanup/reset) over options that stop abruptly, provided the start was visible.\n"
            "3. **Evidence-Based**: Do not hallucinate objects not in the log.\n\n"

            "### Format Requirements:\n"
            "1. First, think step-by-step inside <think> tags. Check for specificity and cyclical patterns.\n"
            "2. Then, provide the final answer inside <answer> tags. The answer must be a single letter (A, B, C, D, or E).\n\n"

            "### Example:\n"
            "User: \n"
            "Question: What is the person doing?\n"
            "Options: A. Making bricks and cleaning the mold. B. Making bricks.\n"
            "Log: 'Frame 0: Person smooths sand... Frame 10: Fills mold... Frame 20: Drops brick... [End of Log]'\n"
            "Assistant: <think> The log shows a cyclical task. Frame 0 shows 'smoothing sand' (prep/cleaning). Option A includes 'cleaning', which matches the cyclical nature implied by Frame 0, even if the end is cut off. Option B is incomplete. </think>\n"
            "<answer> A </answer>\n\n"

            "### Now it's your turn:\n"
            f"User:\n{context_text}\n\n"
            f"Question: {question_text}\n"
            f"Options:\n{options_text}\n\n"
            "Assistant:"
        )

    @staticmethod
    def get_arbiter_prompt(question, context, opt_a_char, opt_a_desc, opt_a_reason, opt_b_char, opt_b_desc, opt_b_reason):
        return (
            "You are a Forensic Video Logic Analyst. Your goal is to identify the Truth by analyzing the 'Underlying Physical Actions', ignoring fancy storytelling.\n\n"
            
            "### 🛑 CORE PHILOSOPHY: PHYSICS > STORY\n"
            "The Video Log is imperfect. Candidates often hallucinate specific game titles (e.g., 'Catan', 'Monopoly') or adjectives (e.g., 'Strategic', 'Careful') to sound convincing.\n"
            "**YOU MUST IGNORE THESE FLAVOR TEXTS.** Focus ONLY on the hand-object interaction.\n\n"

            "### ⚖️ THE 3 LAWS OF SEMANTIC JUDGMENT:\n"
            "1. **THE 'FLAVOR TRAP' LAW (Penalty for Adjectives)**\n"
            "   - IF Option A says: 'Playing a strategic board game involving resource management'.\n"
            "   - IF Option B says: 'Placing cards on a table'.\n"
            "   - VERDICT: **PREFER B**. Option A hallucinates 'strategy' and 'resources' which are invisible. B describes the visible physics.\n\n"

            "2. **THE 'INTERACTION' LAW (Action > Noun)**\n"
            "   - IF Log shows: 'Hand moving flat object to table'.\n"
            "   - Option A: 'Playing Board Game' (Implies pieces/dice).\n"
            "   - Option B: 'Playing Card Game' (Implies flat objects).\n"
            "   - VERDICT: **PICK B**. Match the *shape* of the interaction (Flat object -> Card).\n\n"

            "3. **THE 'NON-EXISTENT' LAW**\n"
            "   - Reject an option ONLY if it requires an object that is CLEARLY ABSENT (e.g., 'Rolling dice' when no dice are seen).\n"
            "   - Do not reject 'Cards' just because the Log calls them 'Papers'.\n\n"

            f"--- 1. QUESTION ---\n\"{question}\"\n\n"
            
            f"--- 2. IMPERFECT VIDEO LOG ---\n{context}\n\n"
            
            f"--- 3. CANDIDATE ACCOUNTS ---\n"
            f"🔴 Account {opt_a_char}: \"{opt_a_desc}\"\n"
            f"   Logic: \"{opt_a_reason}\"\n\n"
            
            f"🔵 Account {opt_b_char}: \"{opt_b_desc}\"\n"
            f"   Logic: \"{opt_b_reason}\"\n\n"
            
            "--- 4. JUDGMENT PROTOCOL (Step-by-Step) ---\n"
            "Step 1: **Strip the Flavor**. Remove all adjectives (e.g., 'strategic', 'fun', 'careful'). What is the *bare bone* physical action?\n"
            "Step 2: **Physics Match**. Match the Log's movement (e.g., 'shuffling', 'placing') to the Options. Which one fits the *mechanics* better?\n"
            "Step 3: **Occam's Razor**. If one option requires assuming invisible rules (e.g., 'Catan'), and the other just describes visible objects (e.g., 'Cards'), choose the simpler one.\n"
            "Step 4: **Verdict**. Choose the account that is physically most accurate, even if less detailed.\n\n"
            
            "--- 5. FINAL REPORT ---\n"
            "<think>\n"
            "1. Strip Flavor Analysis: ...\n"
            "2. Physics Match: ...\n"
            "3. Verdict:\n"
            "</think>\n"
            f"<answer> {opt_a_char} or {opt_b_char} </answer>\n"
        )

    @staticmethod
    def call_arbiter_api(prompt, api_url, model_name, fallback_ans):
        """Calls the high-capacity Arbiter via API for hostile verification."""
        payload = {
            "model": model_name,
            "messages": [{"role": "system", "content": "You are a pragmatic log analyst."}, {"role": "user", "content": prompt}],
            "max_tokens": 1024, "temperature": 0.1, "seed": 3407 
        }
        for _ in range(3):
            try:
                r = requests.post(api_url, headers=API_HEADERS, json=payload, timeout=300)
                if r.status_code == 200: return r.json()['choices'][0]['message']['content']
                time.sleep(1)
            except Exception as e:
                log.warning(f"API call failed: {e}. Retrying...")
                time.sleep(1)
        return f"<think> Service Unreachable. </think>\n<answer> {fallback_ans} </answer>"

# ==========================================
# 3. Main Evaluation Logic
# ==========================================
def run_baseline(model, tokenizer, full_item, budget_manager, args, pid):
    """Executes N-vote Self-Consistency with Arbiter resolution for strict ties."""
    
    # 1. Local deterministic seed
    try: nums = re.sub(r'\D', '', str(pid))
    except: nums = ""
    clean_pid = int(nums[:18]) if nums else len(str(full_item))
    local_seed = (args.seed + clean_pid) % (2**32 - 1)
    set_deterministic(local_seed)

    # 2. Extract Data (EgoSchema Format Alignment)
    raw_log = full_item.get('vlm_analysis', '') or full_item.get('video_log', '') or full_item.get('question', '')
    ctx = budget_manager.get_context(raw_log) 
    clean_question = full_item.get('question', '').strip()

    options_text = ""
    options_map = {} 
    candidates_chars = ["A", "B", "C", "D", "E"]
    
    for char in candidates_chars:
        keys = [f"option_{char.lower()}", f"option {char.lower()}", f"Option {char}"]
        val = None
        for k in keys:
            if full_item.get(k):
                val = full_item[k]
                break
        if val:
            options_text += f"{char}. {val}\n"
            options_map[char] = val 
            
    if not options_text:
        c_pattern = r"(Choices:|Options:|Choices\s*:)\s*(.*)"
        c_match = re.search(c_pattern, full_item.get('question', ''), re.DOTALL | re.IGNORECASE)
        if c_match:
            options_text = c_match.group(2).strip()
            options_map = defaultdict(lambda: "Description not available (Parsed from text).")
        else:
            options_text = "Options: A, B, C, D, E"
            options_map = defaultdict(lambda: "Description not available.")

    final_prompt = BaselineLogic.get_user_prompt(ctx, clean_question, options_text)

    # 3. Generate N Votes (Self-Consistency)
    inputs = tokenizer(final_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, 
            max_new_tokens=2048,
            do_sample=True, 
            temperature=0.7, 
            num_return_sequences=args.n_votes
        )

    # 4. Extract Votes and Rationale
    votes = []
    reasoning_map = {} 
    
    for i in range(len(out)):
        raw_text = tokenizer.decode(out[i][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        ans, reason = extract_answer_and_reasoning(raw_text)
        if ans and ans in candidates_chars: 
            votes.append(ans)
            if ans not in reasoning_map or len(reason) > len(reasoning_map[ans]):
                reasoning_map[ans] = reason

    if not votes:
        return "C", {}, {"Error": "No valid votes extracted"} 

    # 5. Vote Statistics
    vote_counts = Counter(votes)
    sorted_votes = sorted(vote_counts.items(), key=lambda x: x[1], reverse=True)
    
    top1_opt, top1_val = sorted_votes[0]
    final_pred = top1_opt
    logic_tag = f"Majority_Vote({top1_val}/{args.n_votes})"
    verifier_out = None

    # 6. Strict Tie-Break Logic
    if len(sorted_votes) >= 2:
        top2_opt, top2_val = sorted_votes[1]
        is_strict_tie = (top1_val == top2_val)
        
        if is_strict_tie:
            # Trigger Arbiter for tie-breaking
            opt_a_desc = options_map.get(top1_opt, "Unknown")
            opt_a_reason = reasoning_map.get(top1_opt, "")
            opt_b_desc = options_map.get(top2_opt, "Unknown")
            opt_b_reason = reasoning_map.get(top2_opt, "")
            
            v_prompt = BaselineLogic.get_arbiter_prompt(
                clean_question, ctx, 
                top1_opt, opt_a_desc, opt_a_reason,
                top2_opt, opt_b_desc, opt_b_reason
            )
            
            v_result = BaselineLogic.call_arbiter_api(v_prompt, args.arbiter_api_url, args.arbiter_model_name, top1_opt)
            v_ans, _ = extract_answer_and_reasoning(v_result)
            
            # Ensure Arbiter selects one of the tied options
            if v_ans and v_ans in [top1_opt, top2_opt]: 
                final_pred = v_ans
                logic_tag = f"Verifier_TieBreak({top1_val}v{top2_val})->{v_ans}"
                verifier_out = v_result
            else:
                logic_tag = f"Verifier_Fallback({top1_val}v{top2_val})->{top1_opt}"
                verifier_out = v_result

    expert_details = {
        "Generalist": f"{top1_opt}({top1_val}/{args.n_votes})", 
        "Logic": logic_tag,
        "Verifier_Think": verifier_out
    }
    
    return final_pred, dict(vote_counts), expert_details

# ==========================================
# 4. Script Execution
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Baseline Inference: Self-Consistency with Arbiter Tie-Break")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to the lightweight base model")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the trained LoRA adapter")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the evaluation dataset (JSONL)")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the inference results")
    parser.add_argument("--n_votes", type=int, default=5, help="Number of trajectories sampled for Self-Consistency")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed for reproducibility")
    parser.add_argument("--arbiter_api_url", type=str, default="http://localhost:8000/v1/chat/completions", help="API Endpoint for the Arbiter")
    parser.add_argument("--arbiter_model_name", type=str, default="Qwen/Qwen2.5-32B-Instruct-AWQ", help="Model name registered in the Arbiter API")
    args = parser.parse_args()

    log.info(f"🚀 Initializing Baseline (SC + Tie-Break) | Seed: {args.seed}")
    set_deterministic(args.seed)
    
    # Load Models
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(args.base_model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    
    if args.model_path:
        log.info(f"➕ Loading Adapter: {args.model_path}")
        model = PeftModel.from_pretrained(model, args.model_path)
    
    model.eval()
    budget_manager = SmartBudgetManager(tokenizer)

    # Data Loading
    with open(args.input_file, 'r', encoding='utf-8') as f:
        try: data = json.load(f)
        except: 
            f.seek(0)
            data = [json.loads(line) for line in f if line.strip()]
    
    # Resumption Logic
    processed = set()
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r') as f:
            for line in f:
                try: processed.add(str(json.loads(line).get('id')))
                except: pass

    f_out = open(args.output_file, 'a', encoding='utf-8')
    correct = 0
    total = 0

    for item in tqdm(data, desc="Baseline Eval"):
        pid = str(item.get('id') or item.get('Question_id'))
        if pid in processed: continue
        
        try:
            final_pred, vote_counts, details = run_baseline(
                model, tokenizer, item, budget_manager, args, pid
            )
            
            gt = item.get('answer') or item.get('answer_label')
            is_correct = (final_pred == gt) if gt else False
            if is_correct: correct += 1
            if gt: total += 1
            
            marker = "✅" if is_correct else "❌"
            log.info(f"🆔 {pid} | {marker} Final:{final_pred} GT:{gt} | {details.get('Logic')}")
            
            res = {
                "id": pid, 
                "label": gt, 
                "pred": final_pred, 
                "is_correct": is_correct, 
                "vote_counts": vote_counts, 
                "expert_details": details
            }
            f_out.write(json.dumps(res, ensure_ascii=False) + "\n")
            f_out.flush()

        except Exception as e:
            import traceback
            traceback.print_exc()
            log.error(f"Error {pid}: {e}")
            torch.cuda.empty_cache()

    f_out.close()
    
    if total > 0:
        log.info(f"🎉 Baseline Evaluation Complete. Final Accuracy: {correct/total:.2%}")

if __name__ == "__main__":
    main()