
import sys
import os
import argparse
import uvicorn
import json
import time
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional, Any, Union
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

# Initialize FastAPI
app = FastAPI()

# Global model and tokenizer
model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"

class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Any]] # Handle both string and complex vision content

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "default"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 40
    n_predict: Optional[int] = -1
    max_tokens: Optional[int] = 2048
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    stream_options: Optional[dict] = None

    # For Pydantic v2
    model_config = {"extra": "allow"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    global model, tokenizer
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Raw body logging for robustness debugging
    body_bytes = await request.body()
    body_str = body_bytes.decode()
    print(f"Incoming Request: {body_str}")
    
    try:
        data = json.loads(body_str)
        req = ChatCompletionRequest(**data)
    except Exception as e:
        print(f"Request validation error: {e}")
        # Try to proceed with raw data if validation fails but keys exist
        try:
             # Manual mapping if Pydantic fails
             messages_data = data.get("messages", [])
             messages = [ChatMessage(**m) if isinstance(m, dict) else m for m in messages_data]
             class DummyReq: pass
             req = DummyReq()
             req.messages = messages
             req.temperature = data.get("temperature", 0.7)
             req.top_p = data.get("top_p", 0.9)
             req.top_k = data.get("top_k", 40)
             req.n_predict = data.get("n_predict", -1)
             req.max_tokens = data.get("max_tokens", 2048)
             req.stream = data.get("stream", False)
             req.model = data.get("model", "default")
        except:
            raise HTTPException(status_code=422, detail=f"Invalid request format: {e}")

    # Construct prompt
    try:
        # Construct message list for template
        chat_messages = []
        for m in req.messages:
            content = m.content
            if isinstance(content, list):
                # Extract text from vision content list for now
                text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                content = " ".join(text_parts)
            chat_messages.append({"role": m.role, "content": content})

        if tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)
        else:
            # Fallback simple chat format
            prompt = ""
            for msg in chat_messages:
                prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            prompt += "<|im_start|>assistant\n"
    except Exception as e:
        print(f"Prompt construction error: {e}")
        prompt = ""
        for msg in req.messages:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            prompt += f"{msg.role}: {content}\n"
        prompt += "assistant: "

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    stop_sequences = ["<|im_start|>", "<|im_end|>", "\nuser:", "\nassistant:", "###", "</s>"]
    if tokenizer.eos_token:
        stop_sequences.append(tokenizer.eos_token)

    # Map n_predict to max_new_tokens
    max_new_tokens = 2048
    if hasattr(req, 'n_predict') and req.n_predict > 0:
        max_new_tokens = req.n_predict
    elif hasattr(req, 'max_tokens') and req.max_tokens > 0:
        max_new_tokens = req.max_tokens

    generate_kwargs = dict(
        input_ids=inputs.input_ids,
        max_new_tokens=max_new_tokens,
        temperature=req.temperature if req.temperature > 0 else 1.0,
        top_p=req.top_p,
        top_k=req.top_k,
        do_sample=True if req.temperature > 0 else False,
        pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0,
        stop_strings=stop_sequences,
        tokenizer=tokenizer
    )

    if req.stream:
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, timeout=10)
        generate_kwargs["streamer"] = streamer
        
        thread = Thread(target=model.generate, kwargs=generate_kwargs)
        thread.start()

        from fastapi.responses import StreamingResponse
        
        def stream_generator():
            try:
                for new_text in streamer:
                    chunk = {
                        "id": f"chatcmpl-{int(time.time())}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": req.model,
                        "choices": [{"index": 0, "delta": {"content": new_text}, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                print(f"Streaming error: {e}")
                error_chunk = {"error": str(e)}
                yield f"data: {json.dumps(error_chunk)}\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        with torch.no_grad():
            outputs = model.generate(**generate_kwargs)
        
        response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(inputs.input_ids[0]),
                "completion_tokens": len(outputs[0]) - len(inputs.input_ids[0]),
                "total_tokens": len(outputs[0])
            }
        }

def load_model(model_path: str):
    global model, tokenizer
    print(f"Loading model from {model_path} on {device}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="auto", 
            dtype=torch.float16 if device == "cuda" else torch.float32,
            local_files_only=True,
            trust_remote_code=True
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    load_model(args.model)
    print(f"Starting Transformers Server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
