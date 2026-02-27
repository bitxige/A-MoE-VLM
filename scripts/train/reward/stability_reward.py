# -*- coding: utf-8 -*-
"""
Stability Reward Function for GRPO Alignment
This module implements an intra-group consistency reward to encourage robust, 
long-chain System 2 reasoning. It evaluates the Jaccard similarity between 
the <think> blocks of correct generations within the same prompt group, 
incorporating a length-incentive factor to heavily penalize "reward hacking" 
(i.e., generating artificially short reasoning paths).
"""

import logging
import re
from typing import List

from swift.plugin import ORM, orms

log = logging.getLogger(__name__)

class StabilityReward(ORM):
    """
    Rewards the model when multiple sampled trajectories for the same prompt 
    yield logically consistent (similar) reasoning paths leading to the correct answer.
    """
    
    # --- Regular Expressions for Answer Extraction ---
    BOXED_PATTERN = r"\$\\boxed\{([A-H])\}\$"
    ANSWER_PATTERN = r"answer\s+([A-H])\.?"
    SIMPLE_DOT_PATTERN = r"(?:^|[^A-Za-z])([A-H])\s*\."
    SIMPLE_PATTERN = r"(?:^|[^A-Za-z])([A-H])(?:$|[^A-Za-z])"

    def __init__(
        self, 
        similarity_threshold: float = 0.3, 
        min_len: int = 50, 
        target_len: int = 1000, 
        reward_value: float = 1.0
    ):
        """
        Args:
            similarity_threshold (float): Minimum Jaccard similarity to trigger the reward.
            min_len (int): Hard threshold. Reasoning chains shorter than this are disqualified.
            target_len (int): The target length for the reasoning chain to achieve the full reward.
            reward_value (float): The base multiplier for the reward.
        """
        super().__init__()
        self.similarity_threshold = similarity_threshold
        self.min_len = min_len
        self.target_len = target_len
        self.reward_value = reward_value
        log.info(
            f"Initialized StabilityReward | Anti-Hack Mode: ON "
            f"(Thresh={similarity_threshold}, MinLen={min_len}, TargetLen={target_len})"
        )

    def normalize_answer(self, answer: str) -> str:
        """Robustly extracts and normalizes the choice character (A-H)."""
        if not isinstance(answer, str): 
            return ""
        answer = answer.strip()
        
        boxed_matches = list(re.finditer(self.BOXED_PATTERN, answer, re.IGNORECASE))
        if boxed_matches: return boxed_matches[-1].group(1).upper()
        
        answer_matches = list(re.finditer(self.ANSWER_PATTERN, answer, re.IGNORECASE))
        if answer_matches: return answer_matches[0].group(1).upper()
        
        dot_matches = list(re.finditer(self.SIMPLE_DOT_PATTERN, answer, re.IGNORECASE))
        if dot_matches: return dot_matches[-1].group(1).upper()
        
        simple_matches = list(re.finditer(self.SIMPLE_PATTERN, answer, re.IGNORECASE))
        if simple_matches: return simple_matches[-1].group(1).upper()
        
        match = re.search(r"<answer>(.*?)</answer>", answer, re.DOTALL)
        if match: return self.normalize_answer(match.group(1).strip())
        
        return answer.upper()

    @staticmethod
    def extract_think(text: str) -> str | None:
        """Extracts the internal reasoning process enclosed in <think> tags."""
        if not isinstance(text, str): 
            return None
        # Prioritize fully closed tags
        match = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
        if match: 
            return match.group(1).strip()
        # Fallback for truncated outputs
        match_start = re.search(r"<think>(.*)", text, re.DOTALL | re.IGNORECASE)
        if match_start: 
            return match_start.group(1).strip()
        return None

    @staticmethod
    def jaccard_similarity(text1: str, text2: str) -> float:
        """Calculates token-level Jaccard similarity between two strings."""
        if not text1 or not text2: return 0.0
        try:
            set1 = set(re.findall(r'\b\w+\b', text1.lower()))
            set2 = set(re.findall(r'\b\w+\b', text2.lower()))
            if not set1 and not set2: return 1.0 
            if not set1 or not set2: return 0.0  
            intersection = set1.intersection(set2)
            union = set1.union(set2)
            return len(intersection) / len(union) if union else 0.0
        except Exception:
            return 0.0

    def __call__(self, completions: List[str], solution: List[str] = None, **kwargs) -> List[float]:
        """Evaluates intra-group consistency and assigns scaled rewards."""
        num_completions = len(completions)
        stability_rewards = [0.0] * num_completions
        
        if solution is None: 
            return stability_rewards

        # Dynamically infer group size from GRPO kwargs, default to 4
        group_size = kwargs.get("group_size", 4) 

        # Process completions group by group
        for g_start in range(0, num_completions, group_size):
            g_end = min(g_start + group_size, num_completions)
            
            group_correct_indices = []
            group_correct_thinks = {}
            skipped_short_count = 0
            
            # --- Phase A: Qualification Filtering ---
            for i in range(g_start, g_end):
                sol = solution[i] if i < len(solution) else solution[-1]
                
                norm_comp = self.normalize_answer(completions[i])
                norm_sol = self.normalize_answer(sol)
                
                # Prerequisite: The trajectory must yield the correct answer
                if norm_comp == norm_sol and norm_comp != "":
                    think = self.extract_think(completions[i])
                    
                    if think:
                        if len(think) < self.min_len:
                            skipped_short_count += 1
                        else:
                            group_correct_indices.append(i)
                            group_correct_thinks[i] = think

            valid_count = len(group_correct_indices)
            if valid_count > 0 or skipped_short_count > 0:
                log.debug(
                    f"[StabilityReward] Group {g_start // group_size}: "
                    f"{valid_count} valid trajectories. (Skipped {skipped_short_count} < {self.min_len} chars)"
                )

            # --- Phase B: Pairwise Cross-Validation & Reward Scaling ---
            if valid_count >= 2:
                indices = list(group_correct_thinks.keys())
                
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        idx_a = indices[i]
                        idx_b = indices[j]
                        
                        text_a = group_correct_thinks[idx_a]
                        text_b = group_correct_thinks[idx_b]
                        
                        sim = self.jaccard_similarity(text_a, text_b)
                        
                        if sim > self.similarity_threshold:
                            # Length Incentive Mechanism: Prevents policy collapse into short answers
                            avg_len = (len(text_a) + len(text_b)) / 2.0
                            len_factor = min(1.0, avg_len / self.target_len)
                            
                            bonus = sim * self.reward_value * len_factor
                            
                            # Assign the max bonus achieved in any pairwise comparison within the group
                            stability_rewards[idx_a] = max(stability_rewards[idx_a], bonus)
                            stability_rewards[idx_b] = max(stability_rewards[idx_b], bonus)
                            
                            log.debug(f" -> Pair ({idx_a}, {idx_b}) | Sim: {sim:.2f} | LenFactor: {len_factor:.2f} | Bonus: {bonus:.4f}")
                        else:
                            log.debug(f" -> Pair ({idx_a}, {idx_b}) | Sim: {sim:.2f} (Below threshold)")

        return stability_rewards

# Register the plugin with the ms-swift ORM registry
orms['stability'] = StabilityReward