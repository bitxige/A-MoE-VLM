# -*- coding: utf-8 -*-
"""
Baseline Generative Reward Server (Text-Only Heuristics)
This module implements the standard/baseline method for consistency checking 
often found in prior literature. Unlike the epistemic approach (which calculates 
logit entropy), this server relies solely on text generation to evaluate logical consistency.
"""

import argparse
import logging
import re
import sys
import traceback
from contextlib import asynccontextmanager
from typing import List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
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
logger = logging.getLogger("BaselineRewardServer")

# ==========================================
# Data Models (OpenAI API Compatible)
# ==========================================
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "default-model"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.1
    max_tokens: Optional[int] = 10

class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage

class ChatCompletionResponse(BaseModel):
    choices: List[ChatCompletionChoice]

# ==========================================
# Global Application State
# ==========================================
class AppState:
    model: AutoModelForCausalLM = None
    tokenizer: AutoTokenizer = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

state = AppState()

# ==========================================
# Utility Functions
# ==========================================
def extract_answer(text: str) -> str:
    """
    A basic heuristic answer extractor (A-H).
    Returns "Z" as an invalid marker if no match is found, mirroring baseline protocols.
    """
    if not text: 
        return "Z"
    matches = re.findall(r'([A-H])', text.strip().upper())
    if matches: 
        return matches[-1]
    return "Z"

# ==========================================
# FastAPI Lifecycle & Endpoints
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for model initialization and memory teardown."""
    model_path = app.state.model_path
    logger.info(f"🚀 Initializing Baseline Generative Reward Model: {model_path}")
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
            
        logger.info("✅ Baseline Reward Model successfully loaded into VRAM.")
        yield
    except Exception as e:
        logger.error(f"❌ Fatal error during model loading: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        logger.info("🛑 Shutting down server, flushing VRAM...")
        del state.model
        del state.tokenizer
        torch.cuda.empty_cache()

app = FastAPI(title="Baseline Generative Reward Server", lifespan=lifespan)

@app.get("/health")
async def health_check():
    """Health check endpoint used by the training orchestrator scripts."""
    return {"status": "ok"}

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completion(request: ChatCompletionRequest):
    """
    Evaluates reasoning purely based on textual generation outputs.
    Optimized for speed by disabling output_scores and dict returns.
    """
    if state.model is None:
        raise HTTPException(status_code=503, detail="Reward Model is not loaded.")
        
    logger.debug("Received arbitration request from Student Policy.")
    
    try:
        # 1. Prompt Construction
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        prompt = state.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        inputs = state.tokenizer(prompt, return_tensors="pt").to(state.device)
        
        # 2. Optimized Inference (Generative-Only)
        with torch.no_grad():
            gen_outputs = state.model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=max(request.temperature, 0.01),
                output_scores=False,           # Memory/Speed optimization
                return_dict_in_generate=False  # Memory/Speed optimization
            )
        
        # 3. Extraction
        output_ids = gen_outputs[0][inputs.input_ids.shape[1]:]
        ans_text = state.tokenizer.decode(output_ids, skip_special_tokens=True)
        
        choice = extract_answer(ans_text.strip())
        logger.info(f"Baseline Verdict: {choice} (Raw text: '{ans_text}')")

        # 4. Standardized Response Packing
        response_message = ChatMessage(role="assistant", content=choice)
        response_choice = ChatCompletionChoice(message=response_message)
        return ChatCompletionResponse(choices=[response_choice])

    except Exception as e:
        logger.error(f"Request processing failed: {e}")
        traceback.print_exc()
        # Fallback to an invalid answer 'Z' so the student policy receives a zero reward
        err_msg = ChatMessage(role="assistant", content="Z")
        err_choice = ChatCompletionChoice(message=err_msg)
        return ChatCompletionResponse(choices=[err_choice])

# ==========================================
# Application Entry Point
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline Text-Only Reward Server")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the Baseline Reward Model (e.g., 7B)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host IP address")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server")
    args = parser.parse_args()

    app.state.model_path = args.model_path

    logger.info(f"Igniting Baseline Server on {args.host}:{args.port}...")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")