# -*- coding: utf-8 -*-
"""
A-MoE (Agentic Mixture-of-Experts) Inference Script
This script implements the Consensus & Hostile Verification (CAHV) mechanism 
for evaluating on the EgoSchema benchmark.
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

# Disable proxy for local API calls
os.environ['no_proxy'] = 'localhost,127.0.0.1'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'

API_HEADERS = {"Content-Type": "application/json"}

def set_deterministic(seed=42):
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
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("inference_amoe.log", mode='w', encoding='utf-8')]
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
# 2. Experts & Arbiter Prompts
# ==========================================
class Experts:
    """Instantiates virtual experts using role-playing system prompts."""
    
    @staticmethod
    def get_generalist(question, context, options):
        prompt = (
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
            f"User:\n{context}\n\n"
            f"Question: {question}\n"
            f"Options:\n{options}\n\n"
            "Assistant:"
        )
        return prompt

    @staticmethod
    def get_navigator(question, context, options):
        return (
            "You are a Spatial Navigation Specialist. Your task is to analyze the agent's movement trajectory and spatial orientation over time.\n"
            f"--- Video Log ---\n{context}\n\n"
            f"Question: {question}\n"
            f"Options:\n{options}\n\n"
            "Instructions:\n"
            "1. **Trajectory Mapping**: Identify the sequence of 'Move', 'Turn', and 'Rotate' actions. Where did the agent start and where did it end?\n"
            "2. **Spatial Consistency**: If the question asks about relative position (e.g., 'behind', 'left of'), trace the agent's head rotation to confirm.\n"
            "3. **Logical Inference**: Discard options that contradict the physical path taken.\n"
            "4. **Ignore Broad Labels**: Focus on PHYSICAL movements. If the user moves plates, prefer 'Moving plates' over 'Cleaning house'.\n"
            "Format: <think> [Step-by-step trajectory analysis] </think>\n<answer> Option </answer>\n"
        )

    @staticmethod
    def get_observer(question, context, options):
        return (
            "You are a Visual Evidence Analyst. Your task is to verify the existence, state, and interaction of objects in the video log.\n"
            f"--- Video Log ---\n{context}\n\n"
            f"Question: {question}\n"
            f"Options:\n{options}\n\n"
            "Instructions:\n"
            "1. **Fact Checking**: Scan the log for keywords mentioned in the options. If an object (e.g., 'red cup') is never mentioned, that option is likely false.\n"
            "2. **Implicit Actions**: If the log shows a 'Result' (e.g., a clean mold), infer the 'Action' (Cleaning) even if the camera missed the hand movement.\n"
            "3. **Elimination**: Strictly eliminate options that hallucinate objects not present in the log.\n"
            "Format: <think> [Evidence-based verification] </think>\n<answer> Option </answer>\n"
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
            "max_tokens": 2048, "temperature": 0.1, "seed": 3407 
        }
        for _ in range(3):
            try:
                r = requests.post(api_url, headers=API_HEADERS, json=payload, timeout=600)
                if r.status_code == 200: return r.json()['choices'][0]['message']['content']
                time.sleep(1)
            except Exception as e: 
                log.warning(f"API call failed: {e}. Retrying...")
                time.sleep(1)
        return f"<think> Service Unreachable. </think>\n<answer> {fallback_ans} </answer>"


# ==========================================
# 3. Main Logic: CAHV Protocol
# ==========================================
def run_cahv(model, tokenizer, full_item, budget_manager, args, pid):
    """Executes the Consensus & Hostile Verification (CAHV) dynamic routing algorithm."""
    
    # 1. Local deterministic seed for reproducible sampling
    try: nums = re.sub(r'\D', '', str(pid))
    except: nums = ""
    clean_pid = int(nums[:18]) if nums else len(str(full_item))
    local_seed = (args.seed + clean_pid) % (2**32 - 1)
    set_deterministic(local_seed)
    
    # 2. Extract Data
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

    # =======================================================
    # Multi-Agent Inference
    # =======================================================
    all_cast_votes = []; expert_details = {}

    # --- Phase 1: Generalist Inference ---
    p_gen = Experts.get_generalist(clean_question, ctx, options_text)
    inputs = tokenizer(p_gen, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=2048, do_sample=True, temperature=0.7, num_return_sequences=args.n_votes)
    
    gen_v = []; gen_r_list = []
    for i in range(len(out)):
        ans, r = extract_answer_and_reasoning(tokenizer.decode(out[i][inputs.input_ids.shape[1]:], skip_special_tokens=True))
        if ans and ans in candidates_chars: gen_v.append(ans); gen_r_list.append(r)
    
    if not gen_v: return "C", {"Status": "Fail"}, {}, {} 
    gen_choice, gen_count = Counter(gen_v).most_common(1)[0]
    all_cast_votes.extend(gen_v)
    expert_details["Generalist"] = f"{gen_choice}({gen_count}/{args.n_votes})"
    
    valid_gen_reasons = [r for a, r in zip(gen_v, gen_r_list) if a == gen_choice]
    gen_reasoning = max(valid_gen_reasons, key=len) if valid_gen_reasons else "Standard Logic."

    # --- Phase 2: Specialists Inference (Navigator & Observer) ---
    spec_results = {}; spec_reasons = {}; spec_counts = {}
    for name, p_func in {"Navigator": Experts.get_navigator, "Observer": Experts.get_observer}.items():
        p_text = p_func(clean_question, ctx, options_text)
        inputs = tokenizer(p_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=2048, do_sample=True, temperature=0.7, num_return_sequences=args.n_votes)
        v = []; r_list = []
        for i in range(len(out)):
            ans, r = extract_answer_and_reasoning(tokenizer.decode(out[i][inputs.input_ids.shape[1]:], skip_special_tokens=True))
            if ans and ans in candidates_chars: v.append(ans); r_list.append(r)
        
        if v:
            all_cast_votes.extend(v)
            top, count = Counter(v).most_common(1)[0]
            spec_results[name] = top; spec_counts[name] = count; expert_details[name] = f"{top}({count}/{args.n_votes})"
            valid_spec_reasons = [r for a, r in zip(v, r_list) if a == top]
            spec_reasons[name] = max(valid_spec_reasons, key=len) if valid_spec_reasons else "Spec Logic."
        else: 
            spec_results[name] = None; spec_counts[name] = 0

    nav_choice = spec_results.get("Navigator"); nav_count = spec_counts.get("Navigator", 0)
    obs_choice = spec_results.get("Observer"); obs_count = spec_counts.get("Observer", 0)

    vote_counts = Counter(all_cast_votes)
    sorted_votes = sorted(vote_counts.items(), key=lambda x: x[1], reverse=True)
    top1_opt, top1_val = sorted_votes[0]
    top2_val = sorted_votes[1][1] if len(sorted_votes) > 1 else 0
    vote_diff = top1_val - top2_val
    
    # ==========================================
    # Phase 3: Dynamic Consensus Gating
    # ==========================================
    
    # Condition 0: Specialist Supermajority Lock
    if nav_choice == obs_choice and nav_choice:
        total_spec_votes = nav_count + obs_count
        if total_spec_votes > 9:
            expert_details["Logic"] = f"Specialist_Supermajority_Lock({total_spec_votes}_Votes)"
            return nav_choice, expert_details, dict(vote_counts), {}

    # Condition 0.5: Navigator Supremacy
    if nav_count >= 5 and gen_count <= 3 and obs_count <= 3 and nav_choice:
         expert_details["Logic"] = f"Nav_Supremacy_Lock(Nav{nav_count}_vs_Others)"
         return nav_choice, expert_details, dict(vote_counts), {}

    # Condition 1: Navigator Coup (Strong override)
    if gen_count <= 2 and nav_count >= 4 and nav_choice != gen_choice and obs_count < 4 and nav_choice:
        expert_details["Logic"] = "Weak_Gen_Nav_Coup_Fast"
        return nav_choice, expert_details, dict(vote_counts), {}

    # Condition 1.5: Generalist-Navigator Alliance
    if gen_choice == nav_choice:
        gen_nav_total = gen_count + nav_count
        if gen_nav_total > 6 and obs_count < 3:
            expert_details["Logic"] = f"Gen_Nav_Alliance_Lock({gen_nav_total}_Votes)"
            return gen_choice, expert_details, dict(vote_counts), {}

    # Condition 1.6: Generalist-Observer Alliance
    if gen_choice == obs_choice:
        total_gen_obs = gen_count + obs_count
        if total_gen_obs > 7 and nav_count <= 2:
            expert_details["Logic"] = f"Gen_Obs_Alliance_Weak_Nav({total_gen_obs}v{nav_count})"
            return gen_choice, expert_details, dict(vote_counts), {}

    # Condition 2: Generalist High Confidence Protection
    if gen_count >= 5:
        is_global_winner = (gen_choice == top1_opt)
        is_safe_lead = (vote_diff > 1)
        if is_global_winner and is_safe_lead:
            expert_details["Logic"] = "Gen_Full_Confidence_Lock(5/5_Safe)"
            return gen_choice, expert_details, dict(vote_counts), {}

    # Condition 2.2: Generalist Negotiation (4 votes)
    if gen_count == 4:
        nav_strong_dissent = (nav_choice != gen_choice and nav_count >= 3 and nav_choice)
        obs_strong_dissent = (obs_choice != gen_choice and obs_count >= 4 and obs_choice)
        if not (nav_strong_dissent or obs_strong_dissent):
            expert_details["Logic"] = "Gen_High_Confidence_Weak_Spec"
            return gen_choice, expert_details, dict(vote_counts), {}

    # Condition 2.5: Observer Rescue (Tie Compatible)
    if obs_count >= 4:
        max_votes = max(vote_counts.values())
        is_obs_top_tier = (vote_counts[obs_choice] == max_votes)
        
        if is_obs_top_tier and obs_choice != gen_choice:
            nav_strong_dissent = (nav_choice != obs_choice and nav_count >= 3)
            if not nav_strong_dissent and gen_count <= 2:
                expert_details["Logic"] = f"Obs_High_Conf_Rescue({obs_count}_Votes)"
                return obs_choice, expert_details, dict(vote_counts), {}

    # Condition 3: Weak Consensus
    if gen_choice == nav_choice:
        total_consensus = gen_count + nav_count
        if total_consensus >= 7 and obs_count <= 3:
            expert_details["Logic"] = f"Gen_Nav_Strong_Consensus({total_consensus}_Votes)"
            return gen_choice, expert_details, dict(vote_counts), {}

    # Condition 4: Deadlock Breaking
    if len(sorted_votes) >= 2:
        if (top1_val - top2_val) == 1:
            if top1_opt == nav_choice:
                expert_details["Logic"] = f"Deadlock_Top1_Nav_Trust({top1_val}v{top2_val})"
                return nav_choice, expert_details, dict(vote_counts), {}
            elif top1_opt == obs_choice and nav_count < 3:
                expert_details["Logic"] = "Deadlock_Top1_Obs_Weak_Nav"
                return obs_choice, expert_details, dict(vote_counts), {}
            elif top1_opt == gen_choice and nav_count < 3:
                expert_details["Logic"] = "Deadlock_Top1_Gen_Weak_Nav"
                return gen_choice, expert_details, dict(vote_counts), {}

    # ==========================================
    # Phase 4: Hostile Verification Escalation
    # ==========================================
    opt_a_choice = gen_choice
    opt_a_reason = gen_reasoning
    opt_b_choice = None
    opt_b_reason = None
    logic_tag = ""

    # Case A: Specialist Civil War (Nav vs Obs)
    if gen_count <= 2 and nav_choice != obs_choice and nav_count >= 3 and obs_count >= 3:
        opt_a_choice = nav_choice 
        opt_a_reason = spec_reasons.get("Navigator")
        opt_b_choice = obs_choice 
        opt_b_reason = spec_reasons.get("Observer")
        logic_tag = "Specialist_Civil_War" 
    
    else:
        # Case B: Mixed Doubles (Nav + Obs vs Gen)
        if nav_choice == obs_choice and nav_choice != gen_choice and nav_choice: 
            opt_b_choice = nav_choice
            opt_b_reason = spec_reasons.get("Navigator") # Escalate only Navigator's logic
            logic_tag = "Consensus_Vs_Gen"
        
        # Case C: Navigator challenges Generalist
        elif nav_choice != gen_choice and nav_choice:
            nav_threshold = 3 if gen_count >= 4 else 2
            if nav_count >= nav_threshold:
                opt_b_choice = nav_choice
                opt_b_reason = spec_reasons.get("Navigator")
                logic_tag = f"Nav({nav_count})_Vs_Gen({gen_count})"

        # Case D: Observer challenges Generalist
        elif obs_choice != gen_choice and obs_choice:
            obs_threshold = 4 if gen_count >= 3 else 3
            if obs_count >= obs_threshold:
                opt_b_choice = obs_choice
                opt_b_reason = spec_reasons.get("Observer")
                logic_tag = "Obs_Vs_Gen"

    # Default Fallback if no valid challenger found
    if not opt_b_choice: 
        expert_details["Logic"] = "No_Strong_Challenger_Fallback"
        return gen_choice, expert_details, dict(vote_counts), {}

    # Proceed to Arbiter Call
    raw_tokens = tokenizer.encode(raw_log, add_special_tokens=False)
    available = MAX_CONTEXT_WINDOW - 1024
    if len(raw_tokens) <= available: 
        v_ctx = raw_log
    else: 
        v_ctx = tokenizer.decode(raw_tokens[:800]) + "\n...[Skipped]...\n" + tokenizer.decode(raw_tokens[-(available-850):])

    opt_a_desc = options_map.get(opt_a_choice, f"Content of Option {opt_a_choice}")
    opt_b_desc = options_map.get(opt_b_choice, f"Content of Option {opt_b_choice}")

    arbiter_prompt = Experts.get_arbiter_prompt(
        clean_question, v_ctx, 
        opt_a_choice, opt_a_desc, opt_a_reason, 
        opt_b_choice, opt_b_desc, opt_b_reason
    )
    
    v_out = Experts.call_arbiter_api(arbiter_prompt, args.arbiter_api_url, args.arbiter_model_name, gen_choice)
    final_ans, _ = extract_answer_and_reasoning(v_out)
    
    if final_ans and final_ans in candidates_chars:
        expert_details["Verifier"] = final_ans
        expert_details["Verifier_Think"] = v_out 
        expert_details["Logic"] = f"Verifier_Escalation_{logic_tag}"
        return final_ans, expert_details, dict(vote_counts), {}
    
    return gen_choice, expert_details, dict(vote_counts), {}

# ==========================================
# 4. Script Execution
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="A-MoE Inference Framework for Visual Reasoning")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to the lightweight base model (e.g., Qwen-3B)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the RL LoRA adapter")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the evaluation dataset (JSONL)")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the inference results")
    parser.add_argument("--n_votes", type=int, default=5, help="Number of trajectories sampled per expert")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed for reproducibility")
    parser.add_argument("--arbiter_api_url", type=str, default="http://localhost:8000/v1/chat/completions", help="API Endpoint for the 32B Arbiter")
    parser.add_argument("--arbiter_model_name", type=str, default="Qwen/Qwen2.5-32B-Instruct-AWQ", help="Model name registered in the Arbiter API")
    args = parser.parse_args()

    log.info(f"🚀 Initializing A-MoE Framework | Seed: {args.seed}")
    set_deterministic(args.seed)
    
    # Load Models
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(args.base_model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
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

    for item in tqdm(data, desc="A-MoE Eval"):
        pid = str(item.get('id') or item.get('Question_id'))
        if pid in processed: continue
        
        try:
            final_pred, expert_details, total_dist, _ = run_cahv(
                model, tokenizer, item, budget_manager, args, pid
            )
            
            gt = item.get('answer')
            is_correct = (final_pred == gt) if gt else False
            if is_correct: correct += 1
            if gt: total += 1
            
            marker = "✅" if is_correct else "❌"
            log.info(f"🆔 {pid} | {marker} Final:{final_pred} GT:{gt} | Acc:{correct/total:.1%} | {expert_details.get('Logic')}")
            
            res = {
                "id": pid, 
                "label": gt, 
                "pred": final_pred, 
                "is_correct": is_correct, 
                "vote_counts": total_dist, 
                "expert_details": expert_details
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
        log.info(f"🎉 Evaluation Complete. Final Accuracy: {correct/total:.2%}")

if __name__ == "__main__":
    main()