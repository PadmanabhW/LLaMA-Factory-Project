#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse
import uvicorn
from contextlib import asynccontextmanager

# Optional imports for system profiling
import psutil
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from fastapi import FastAPI, HTTPException, status, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


# Global model state
chat_model = None
is_mock = True
model_name = "Qwen3-4B-Instruct-Custom"
model_engine = "Mock CPU"


# Pydantic models for OpenAI request/response compatibility
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]


# Mock answers helper (High fidelity)
def get_mock_response(prompt: str) -> str:
    prompt_lower = prompt.lower()
    if "sft" in prompt_lower and "rlhf" in prompt_lower:
        return "Supervised Fine-Tuning (SFT) trains a model on target prompt-response pairs to teach formatting and tone. Reinforcement Learning from Human Feedback (RLHF) optimizes models using preference comparison pairs (chosen/rejected outputs) to align outputs with human values."
    elif "lora" in prompt_lower and "rank" in prompt_lower:
        return "Configure LoRA in LLaMA-Factory by setting 'finetuning_type: lora'. Specify 'lora_rank: 8' (or 16, 32) and target modules via 'lora_target: all' to automatically apply adapters to all linear projection layers of the model."
    elif "templates" in prompt_lower:
        return "The templates system formats raw conversational blocks (e.g. system, user, assistant markers) into exact token structures expected by specific model architectures (e.g. Llama-3, Qwen, DeepSeek)."
    elif "memory" in prompt_lower or "gpu" in prompt_lower:
        return "To optimize memory usage, enable deepspeed (ZeRO 2/3), use QLoRA ('quantization_bit: 4'), activate gradient checkpointing ('gradient_checkpointing: true'), or reduce sequence length ('cutoff_len: 1024')."
    elif "merge" in prompt_lower:
        return "Merge adapters using the CLI export tool:\n\nllamafactory-cli export examples/merge_lora/custom_merge.yaml\n\nEnsure that 'model_name_or_path' points to the base model and 'adapter_name_or_path' points to the LoRA weights directory."
    elif "serving" in prompt_lower or "fastapi" in prompt_lower:
        return "You can deploy the model server via llamafactory-cli api or python src/llamafactory/api/custom_server.py. It hosts standard REST endpoints and an embedded interactive Web Sandbox interface."
    elif "metrics" in prompt_lower:
        return "Standard SFT metrics are BLEU-4 and ROUGE-1/2/L for text overlap assessment, alongside Exact Match (EM) ratios."
    elif "wandb" in prompt_lower or "weights" in prompt_lower:
        return "Weights & Biases (W&B) handles real-time logging of loss values, learning rates, system resource usage, and final evaluation comparison tables."
    elif "gradient" in prompt_lower:
        return "Gradient accumulation accumulates gradients over multiple forward steps before updating weights, allowing you to train with large virtual batches on memory-constrained GPUs."
    elif "register" in prompt_lower:
        return "Register custom datasets in 'data/dataset_info.json' by defining a new dictionary with 'file_name' pointing to your dataset and configuring column names."
    else:
        return f"Hello! This is a mock response from the LLaMA-Factory API service. I received your prompt: \"{prompt[:100]}...\". Start fine-tuning or supply actual model weights to enable real-time inference!"


# Lifespan events
@asynccontextmanager
async def lifespan(app: FastAPI):
    global chat_model, is_mock, model_engine
    
    # Try loading LLaMA-Factory ChatModel
    model_path = os.getenv("API_MODEL_PATH", None)
    if model_path and model_path != "mock":
        try:
            print(f"Loading real LLaMA-Factory ChatModel from: {model_path}...")
            from llamafactory.chat import ChatModel
            chat_model = ChatModel(dict(model_name_or_path=model_path, template="qwen3_nothink"))
            is_mock = False
            model_engine = "LLaMA-Factory ChatModel"
            print("Successfully loaded real model engine.")
        except Exception as e:
            print(f"Failed to load real LLaMA-Factory ChatModel: {e}")
            print("Falling back to CPU Mock/Simulation engine.")
    else:
        print("Starting FastAPI server with high-fidelity Mock/Simulation engine.")
        
    yield
    
    # Shutdown / GC
    if HAS_TORCH:
        torch.cuda.empty_cache()


app = FastAPI(
    title="LLaMA-Factory Custom API Service",
    description="Custom serving layer with Weights & Biases reporting and interactive chat sandbox",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request latency logger middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# 1. Web Sandbox Root UI
@app.get("/", response_class=HTMLResponse)
async def serve_web_ui():
    # Attempt to read index.html from templates directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(script_dir, "templates", "index.html")
    
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        raise HTTPException(status_code=404, detail="Web Sandbox templates/index.html not found.")


# 2. Health check endpoint
@app.get("/health")
async def health_check():
    # CPU/Memory Stats
    cpu_percent = psutil.cpu_percent()
    mem = psutil.virtual_memory()
    mem_allocated = f"{mem.used / (1024**3):.1f} GB / {mem.total / (1024**3):.1f} GB"
    
    gpu_allocated = "0.0 GB"
    if HAS_TORCH and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        gpu_allocated = f"{allocated:.1f} GB (Reserved: {reserved:.1f} GB)"

    return {
        "status": "online",
        "timestamp": time.time(),
        "model_name": model_name,
        "model_engine": model_engine,
        "cpu_usage": f"{cpu_percent}%",
        "ram_usage": mem_allocated,
        "gpu_allocated": gpu_allocated,
        "is_mock": is_mock
    }


# 3. Chat completion endpoint
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    global chat_model, is_mock
    
    # Get prompt text (usually last user message)
    last_user_msg = ""
    for msg in reversed(request.messages):
        if msg.role == "user":
            last_user_msg = msg.content
            break
            
    if not last_user_msg:
        raise HTTPException(status_code=400, detail="Request must contain at least one user message.")
        
    reply_content = ""
    if is_mock:
        # Simulate slight latency
        time.sleep(0.3)
        reply_content = get_mock_response(last_user_msg)
    else:
        try:
            # Transform pydantic messages list into standard python format for ChatModel
            history = [{"role": m.role, "content": m.content} for m in request.messages]
            responses = chat_model.chat(
                history,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            reply_content = responses[0].response_text
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference Engine Error: {str(e)}")

    response_choice = ChatChoice(
        index=0,
        message=ChatMessage(role="assistant", content=reply_content),
        finish_reason="stop"
    )
    
    return ChatCompletionResponse(
        id=f"chatcmpl-{int(time.time())}",
        created=int(time.time()),
        model=model_name,
        choices=[response_choice]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLaMA-Factory Custom API Service")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host binding address")
    parser.add_argument("--port", type=int, default=8000, help="Port binding number")
    parser.add_argument("--model_name_or_path", type=str, default="mock", help="Path to base model or adapter weights")
    
    args = parser.parse_args()
    
    # Pass model path via env to lifespan context manager
    os.environ["API_MODEL_PATH"] = args.model_name_or_path
    
    # Launch uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
