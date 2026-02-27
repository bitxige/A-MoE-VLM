# -*- coding: utf-8 -*-
"""
Logic Consistency Arbiter API Server
This module runs a FastAPI server hosting a high-capacity VLM/LLM (e.g., Qwen-32B).
It evaluates the logical consistency of an agent's reasoning chain by performing a 
forward pass, extracting raw logits, and calculating the Shannon Entropy of the target 
options (A-H) to quantify epistemic uncertainty.
"""

import argparse
import json
import logging
import math
import os
import re
import sys
import traceback
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================
# Logging Configuration
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ConsistencyArbiter")

# ==========================================
# Data Models
# ==========================================
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "default-model"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.1 
    max_tokens: Optional[int] = 10

class PurifiedResponse(BaseModel):
    choice: Optional[str] = None
    entropy: float = -1.0
    error: Optional[str] = None

# ==========================================
# Global Application State
# ==========================================
class AppState:
    model: AutoModelForCausalLM = None
    tokenizer: AutoTokenizer = None
    option_token_map: Dict[int, str] = {}
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

state = AppState()

# ==========================================
# Utility Functions
# ==========================================
def build_option_token_map(tokenizer: AutoTokenizer, options: List[str] = list('ABCDEFGH')) -> Dict[int, str]:
    """
    Constructs a robust mapping from Token IDs to Option Characters.
    Crucially handles the BPE prefix space issue (e.g., 'A' vs ' A').
    """
    logger.info("Constructing robust Token-to-Option mapping...")
    option_map = {}
    for option in options:
        # 1. Standard token without prefix space
        ids = tokenizer.encode(option, add_special_tokens=False)
        if len(ids) == 1: 
            option_map[ids[0]] = option
            
        # 2. Token with prefix space (common in Chat Templates)
        ids_sp = tokenizer.encode(" " + option, add_special_tokens=False)
        if len(ids_sp) == 1: 
            option_map[ids_sp[0]] = option 
            
    logger.info(f"Token Map Initialized with {len(option_map)} variants.")
    return option_map

def extract_answer(text: str) -> Optional[str]:
    """Extracts the final alphabetical choice from the generated text."""
    if not text: return None
    matches = re.findall(r'([A-H])', text.strip().upper())
    if matches: return matches[-1]
    return None

# ==========================================
# Core Epistemic Logic (Forward Pass & Entropy)
# ==========================================
def get_raw_logits_and_entropy(inputs_dict: Dict[str, torch.Tensor]) -> tuple[float, str]:
    """
    Executes a pure forward pass to extract raw logits for the next token.
    Calculates the Shannon Entropy across the valid option tokens to quantify 
    the model's confidence in its logical deduction.
    """
    logger.debug("Executing forward pass (model.forward) for logit extraction...")
    
    with torch.no_grad():
        outputs = state.model(**inputs_dict)
        # Extract logits for the very last token in the sequence
        raw_logits = outputs.logits[0, -1, :] 
        
    probs = F.softmax(raw_logits, dim=-1) 
    entropy_val = 0.0
    
    # Track the maximum probability among variants for each semantic option (A-H)
    option_max_prob = {char: 0.0 for char in set(state.option_token_map.values())}
    for tid, char in state.option_token_map.items():
        if tid < len(probs):
            p = probs[tid].item()
            if p > option_max_prob[char]:
                option_max_prob[char] = p
                
    # Normalize probabilities relative ONLY to the valid choices (A-H)
    target_probs = [option_max_prob[char] for char in sorted(option_max_prob.keys())]
    sum_prob = sum(target_probs)
    
    # Epsilon protection against floating-point underflow
    epsilon = 1e-9
    
    if sum_prob > epsilon:
        norm_probs = [p / sum_prob for p in target_probs]
        for p in norm_probs:
            if p > epsilon:
                entropy_val -= p * math.log2(p)
    else:
        # If the model is completely derailed and assigns ~0 probability to A-H
        entropy_val = -1.0 

    # Identify the top choice implicitly from the probability distribution
    best_choice = max(option_max_prob, key=option_max_prob.get)
    
    logger.debug(f"Target Sub-Distribution Sum: {sum_prob:.4f} | Entropy: {entropy_val:.4f}")
    return float(entropy_val), best_choice

# ==========================================
# FastAPI Lifecycle & Endpoints
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern FastAPI lifecycle manager for safe memory initialization/teardown."""
    model_path = app.state.model_path
    logger.info(f"🚀 Initializing Arbiter Model from: {model_path}")
    try:
        state.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16, 
            device_map="auto", 
            trust_remote_code=True
        )
        state.model.eval()
        
        state.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if state.tokenizer.pad_token is None:
            state.tokenizer.pad_token = state.tokenizer.eos_token
            
        state.option_token_map = build_option_token_map(state.tokenizer)
        logger.info("✅ Model loaded and ready for serving.")
        yield
    except Exception as e:
        logger.error(f"❌ Fatal error during model loading: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Graceful cleanup
        logger.info("🛑 Shutting down Arbiter service, clearing VRAM...")
        del state.model
        del state.tokenizer
        torch.cuda.empty_cache()

app = FastAPI(title="Consistency Arbiter API", lifespan=lifespan)

@app.post("/v1/chat/completions", response_model=PurifiedResponse)
async def chat_completion(request: ChatCompletionRequest):
    """OpenAI-compatible endpoint dedicated to consistency verification."""
    if state.model is None: 
        raise HTTPException(status_code=503, detail="Model is currently initializing or unavailable.")
        
    try:
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        prompt = state.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = state.tokenizer(prompt, return_tensors="pt").to(state.device)
        
        # 1. Epistemic Verification: Calculate Entropy via Forward Pass
        entropy, implicit_choice = get_raw_logits_and_entropy(inputs)
        
        # 2. Generative Verification: Let the model physically output the token
        with torch.no_grad():
            gen_outputs = state.model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=max(request.temperature, 0.01), 
                do_sample=False
            )
        
        output_ids = gen_outputs[0][inputs.input_ids.shape[1]:]
        ans_text = state.tokenizer.decode(output_ids, skip_special_tokens=True)
        explicit_choice = extract_answer(ans_text.strip())
        
        # Prefer the explicitly generated token, fallback to implicit logit maximum
        final_choice = explicit_choice if explicit_choice else implicit_choice
        
        logger.info(f"Verdict -> Choice: {final_choice} | Entropy: {entropy:.4f}")

        return PurifiedResponse(choice=final_choice, entropy=entropy)

    except Exception as e:
        logger.error(f"API Request crashed: {e}")
        traceback.print_exc()
        return PurifiedResponse(error="Internal Server Exception")

# ==========================================
# Application Entry Point
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the Consistency Arbiter API Server")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the HuggingFace model (e.g., 32B Arbiter)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host IP address")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server")
    args = parser.parse_args()

    # Pass the model path into the FastAPI app state for the lifespan manager
    app.state.model_path = args.model_path

    logger.info(f"Igniting Uvicorn Server on {args.host}:{args.port}...")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")