# -*- coding: utf-8 -*-
"""
Consistency Reward Plugin for GRPO (A-MoE Logic-V)
This module acts as the client side of the Reward Server.
It enforces a "Blind Grading" protocol: the Arbiter receives ONLY the student's 
<think> process (without the final answer). The reward is then scaled dynamically 
based on the Arbiter's confidence (calculated via Logit Entropy).
"""

import json
import logging
import re
from typing import Dict, List

import numpy as np
import requests
from swift.plugin import ORM, orms

# ==========================================
# Configuration & Constants
# ==========================================
NUM_CHOICES = 8  # Maximum number of options (A-H)
MAX_ENTROPY = np.log2(NUM_CHOICES) 

log = logging.getLogger(__name__)

# ==========================================
# Core Reward Function Plugin
# ==========================================
class ConsistencyReward(ORM):
    """
    Evaluates the logical validity of an agent's reasoning chain.
    Reward = (Is_Consistent) * (1.0 - Normalized_Entropy)
    """
    
    ANSWER_PATTERN = r"<answer>(.*?)</answer>"
    THINK_PATTERN = r"<think>(.*?)</think>"
    OPTION_PATTERN = re.compile(r"([A-H])") 

    def __init__(self, api_url: str = "http://127.0.0.1:8000/v1/chat/completions", **kwargs):
        super().__init__()
        self.api_url = kwargs.get('api_url', api_url)
            
        log.info("Initialized ConsistencyReward (Blind-Review Arbiter Protocol).")
        log.info(f" -> Arbiter API Endpoint: {self.api_url}")
        log.info(f" -> Epistemic Scaling Enabled | Max Entropy Threshold: {MAX_ENTROPY:.4f}")

    def _extract(self, text: str, pattern: str) -> str:
        """Efficiently extracts content bounded by specific XML tags."""
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else ""

    def _normalize_answer(self, answer: str) -> str:
        """Standardizes the extracted answer to a single uppercase character (A-H)."""
        if not answer: return ""
        answer = answer.strip()
        matches = self.OPTION_PATTERN.findall(answer)
        return matches[-1].upper() if matches else answer.upper()

    def _extract_question(self, messages: List[Dict]) -> str:
        """Extracts the raw user question from the standard ms-swift message payload."""
        for msg in messages:
            if isinstance(msg, dict) and msg.get('role') == 'user':
                return msg.get('content', "")
        return ""

    def call_arbiter_api(self, question: str, think_content: str) -> tuple[str, float]:
        """
        Executes the 'Blind Grading' query against the Arbiter API.
        The student's final answer is deliberately omitted to prevent the Arbiter 
        from 'cheating' or being biased by the conclusion.
        """
        prompt = (
            "[ROLE]\n"
            "You are a 'Logic Grader'. Your task is to evaluate a student's reasoning.\n\n"
            "[CONTEXT: The Real Question]\n"
            f"{question}\n\n"
            "[STUDENT'S WORK]\n"
            f"Thinking Process: {think_content}\n\n"
            "[YOUR TASK]\n"
            "Your job is to follow the student's 'Thinking Process' *exactly* as written. "
            "Based *only* on that 'Thinking Process' (using the 'Context' if needed to understand it), "
            "what answer (A-H) would you *expect* this logic to produce?\n"
            "Do not try to find the *correct* answer. Only find the answer that the *student's logic* implies.\n\n"
            "[OUTPUT]\n"
            "Respond *only* with the single letter (A-H) that the 'Thinking Process' implies."
        )

        headers = {"Content-Type": "application/json"}
        data = {
            "model": "qwen-arbiter", 
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 5,
            "temperature": 0.01 
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=data, timeout=20)
            
            if response.status_code == 200:
                res_json = response.json()
                if 'choice' in res_json and 'entropy' in res_json:
                    arbiter_choice = self._normalize_answer(res_json['choice'])
                    entropy = float(res_json['entropy'])
                    
                    if entropy == -1.0:
                        log.debug("Arbiter API returned invalid entropy (-1.0). Network or VRAM issue likely.")
                        return arbiter_choice, -1.0 
                    return arbiter_choice, entropy
                else:
                    log.warning("Arbiter API response missing expected 'choice' or 'entropy' keys.")
                    return "", -1.0 
            else:
                log.warning(f"Arbiter API HTTP Error: {response.status_code} - {response.text}")
                return "", -1.0
        except requests.exceptions.RequestException as e:
            log.debug(f"Arbiter API Connection Failed (Likely heavy load): {e}")
            return "", -1.0

    def __call__(self, completions: List[str], solution: List[str] = None, **kwargs) -> List[float]:
        """Evaluates logical consistency and assigns a dynamically scaled reward."""
        rewards = []
        
        # 1. Extract Question Context
        questions = []
        if 'messages' in kwargs and kwargs['messages'] is not None:
            messages_batch = kwargs['messages']
            for msg_list in messages_batch:
                questions.append(self._extract_question(msg_list))
        
        if len(questions) != len(completions):
            log.error(f"Batch dimension mismatch: {len(questions)} questions vs {len(completions)} completions.")
            return [0.0] * len(completions)

        # 2. Iterate over generation trajectories
        for i, completion in enumerate(completions):
            try:
                # Isolate cognitive blocks
                think_content = self._extract(completion, self.THINK_PATTERN)
                student_answer_raw = self._extract(completion, self.ANSWER_PATTERN)
                
                # Strict Format Check: Reject if tags are missing or malformed
                if not think_content or not student_answer_raw:
                    rewards.append(0.0) 
                    continue

                student_choice = self._normalize_answer(student_answer_raw)
                question = questions[i]
                
                # Query Arbiter (Blind Evaluation)
                arbiter_choice, entropy = self.call_arbiter_api(question, think_content) 

                # Base consistency check
                base_reward = 0.0
                if arbiter_choice and student_choice and arbiter_choice == student_choice:
                    base_reward = 1.0

                # Epistemic Reward Scaling (Penalty for high uncertainty)
                p_weight = 1.0 
                if entropy == -1.0 or entropy is None:
                    log.debug(f"Sample {i} - Entropy inaccessible. Defaulting to unscaled reward.")
                else:
                    normalized_entropy = max(0.0, min(1.0, entropy / MAX_ENTROPY))
                    p_weight = 1.0 - normalized_entropy
                
                final_reward = base_reward * p_weight
                
                # Verbose logging for real-time training monitoring
                log.info(
                    f"[ConsistencyReward] ID: {i} | "
                    f"Arbiter: {arbiter_choice} vs Student: {student_choice} | "
                    f"Base Rwd: {base_reward} | Entropy: {entropy:.4f} | Weight: {p_weight:.4f} | "
                    f"FINAL RWD: {final_reward:.4f}"
                )
                
                rewards.append(final_reward)
                    
            except Exception as e:
                log.error(f"Failed to process reward for trajectory {i}: {e}")
                rewards.append(0.0)

        return rewards

# Register the plugin with the ms-swift ORM framework exactly as before
orms['logical_consistency'] = ConsistencyReward