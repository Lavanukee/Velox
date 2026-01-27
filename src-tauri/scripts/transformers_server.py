import sys
import os
import argparse
import uvicorn
import json
import time
import logging
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional, Any, Union
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, AutoProcessor, AutoConfig, AutoModel, AutoModelForVision2Seq
from peft import PeftModel
from threading import Thread
import base64
from io import BytesIO
from PIL import Image

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("TF_SERVER")

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

    # Raw body logging
    try:
        body_bytes = await request.body()
        body_str = body_bytes.decode()
        logger.info(f"Incoming Request: {body_str[:500]}...") # Truncate for sanity
    except Exception as e:
        logger.error(f"Failed to read request body: {e}")

    try:
        data = json.loads(body_str)
        req = ChatCompletionRequest(**data)
    except Exception as e:
        logger.warning(f"Request validation error: {e}. Attempting manual parsing.")
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

    # Multimodal processing
    images = []
    has_images = False
    
    try:
        # Construct message list for template
        chat_messages = []
        for m in req.messages:
            content = m.content
            if isinstance(content, list):
                # Multimodal content
                text_content = ""
                for part in content:
                    if part.get("type") == "text":
                        text_content += part.get("text", "")
                    elif part.get("type") == "image_url":
                        img_data = part.get("image_url", {}).get("url", "")
                        if img_data.startswith("data:image"):
                            # Extract base64
                            try:
                                import base64
                                from io import BytesIO
                                from PIL import Image
                                base64_data = img_data.split(",")[1]
                                # Some models need explicit placeholders if the chat template doesn't add them
                                # Qwen3-VL usually handles it, but let's be safe if no image tokens found
                                if "<|vision_start|>" not in text_content and "<image>" not in text_content:
                                     text_content = "<|vision_start|><|image_pad|><|vision_end|>" + text_content
                                images.append(image)
                                has_images = True
                            except Exception as img_err:
                                logger.error(f"Failed to decode image: {img_err}")
                content = text_content
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
        logger.error(f"Prompt construction error: {e}")
        prompt = ""
        for msg in req.messages:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            prompt += f"{msg.role}: {content}\n"
        prompt += "assistant: "

    logger.info(f"Generated Prompt (len={len(prompt)}): {prompt[:100]}...")

    try:
        # Check if tokenizer is actually a processor
        if hasattr(tokenizer, "tokenizer"):
            # It's a processor
            if has_images:
                logger.info(f"Processing message with {len(images)} images")
                inputs = tokenizer(text=prompt, images=images, return_tensors="pt").to(device)
            else:
                inputs = tokenizer(text=prompt, return_tensors="pt").to(device)
        else:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
    except Exception as e:
        logger.error(f"Tokenization error: {e}")
        # Final fallback: try just calling it with generic args
        try:
             inputs = tokenizer(prompt, return_tensors="pt").to(device)
        except Exception as e2:
             logger.error(f"Double tokenization error: {e2}")
             raise HTTPException(status_code=500, detail=f"Tokenization failed: {e}")

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
                logger.info(f"Starting stream generator")
                for new_text in streamer:
                    logger.debug(f"Generated chunk: '{new_text}'")
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
                logger.error(f"Streaming error: {e}")
                error_chunk = {"error": str(e)}
                yield f"data: {json.dumps(error_chunk)}\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        with torch.no_grad():
            outputs = model.generate(**generate_kwargs)
        
        response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        # CRITICAL FIX: Ensure we never return the string "None" to the frontend
        if response_text is None or response_text == "None":
            response_text = ""
        
        logger.info(f"Generated full text (len={len(response_text)})")
        
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

def load_model(path_arg: str):
    global model, tokenizer
    
    # Check for LoRA (split by |)
    if "|" in path_arg:
        parts = path_arg.split("|")
        model_path = parts[0]
        lora_path = parts[1]
        logger.info(f"Detected Base Model + LoRA configuration.")
        logger.info(f"Base Model: {model_path}")
        logger.info(f"LoRA Adapter: {lora_path}")
    else:
        model_path = path_arg
        lora_path = None
        logger.info(f"Loading Base Model: {model_path}")

    logger.info(f"Loading on device: {device}")
    
    try:
        trust_remote_code = True
        
        # Load config to detect model type
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        model_type = getattr(config, "model_type", "").lower()
        architectures = getattr(config, "architectures", [])
        
        is_multimodal = any("ConditionalGeneration" in arch for arch in architectures) or \
                        any("Vision" in arch for arch in architectures) or \
                        "vl" in model_type or \
                        hasattr(config, "vision_config") or \
                        hasattr(config, "visual")

        logger.info(f"Model Type: {model_type}, Architectures: {architectures}, Is Multimodal: {is_multimodal}")

        if is_multimodal:
            logger.info(f"Detected multimodal model type: {model_type}. Using AutoModelForImageTextToText/AutoProcessor.")
            
            # Use AutoModelForImageTextToText if available (Transformers 4.45+)
            try:
                from transformers import AutoModelForImageTextToText
                loader_class = AutoModelForImageTextToText
            except ImportError:
                loader_class = AutoModelForVision2Seq

            try:
                # Try specific loader
                model = loader_class.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    device_map="auto",
                    trust_remote_code=trust_remote_code
                )
            except Exception as e:
                logger.warning(f"{loader_class.__name__} failed ({e}), trying generic AutoModel...")
                model = AutoModel.from_pretrained(
                    model_path,
                    device_map="auto",
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    trust_remote_code=trust_remote_code
                )
            
            # Multimodal models usually need a processor
            tokenizer = AutoProcessor.from_pretrained(model_path, trust_remote_code=trust_remote_code)
            
            # For chat template support, processors sometimes lack it but their internal tokenizer has it
            if not hasattr(tokenizer, 'chat_template') or not tokenizer.chat_template:
                try:
                    temp_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
                    tokenizer.chat_template = temp_tokenizer.chat_template
                    logger.info("Successfully copied chat_template from AutoTokenizer to AutoProcessor")
                except:
                    pass
        else:
            # Standard CausalLM Loading
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.padding_side = "left"

            model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                device_map="auto", 
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                trust_remote_code=trust_remote_code
            )
        
        # Ensure model config matches tokenizer
        # For multimodal processors, pad_token_id is on the inner tokenizer
        if hasattr(model, 'config') and hasattr(model.config, 'pad_token_id') and model.config.pad_token_id is None:
            # Try to get pad_token_id from tokenizer or its inner tokenizer
            pad_id = None
            if hasattr(tokenizer, 'pad_token_id'):
                pad_id = tokenizer.pad_token_id
            elif hasattr(tokenizer, 'tokenizer') and hasattr(tokenizer.tokenizer, 'pad_token_id'):
                pad_id = tokenizer.tokenizer.pad_token_id
                logger.info(f"Using inner tokenizer's pad_token_id: {pad_id}")
            
            if pad_id is not None:
                model.config.pad_token_id = pad_id
            else:
                logger.warning("Could not find pad_token_id, using eos_token_id as fallback")
                if hasattr(tokenizer, 'eos_token_id'):
                    model.config.pad_token_id = tokenizer.eos_token_id
                elif hasattr(tokenizer, 'tokenizer') and hasattr(tokenizer.tokenizer, 'eos_token_id'):
                    model.config.pad_token_id = tokenizer.tokenizer.eos_token_id
        
        # Load LoRA if requested
        if lora_path:
            logger.info(f"Loading LoRA adapter from {lora_path}...")
            # Resolve lora_path if it's relative
            if not os.path.isabs(lora_path):
                # Try common locations
                possible_paths = [
                    lora_path,
                    os.path.join(os.getcwd(), lora_path),
                ]
                for p in possible_paths:
                    if os.path.exists(p):
                        lora_path = p
                        break
            
            if not os.path.exists(lora_path):
                logger.error(f"LoRA adapter path does not exist: {lora_path}")
                raise FileNotFoundError(f"LoRA adapter not found: {lora_path}")
            
            logger.info(f"Resolved LoRA path: {lora_path}")
            model = PeftModel.from_pretrained(model, lora_path)
            logger.info("LoRA adapter loaded successfully.")

        logger.info("Model System Ready.")
    except Exception as e:
        logger.critical(f"FATAL: Error loading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    load_model(args.model)
    logger.info(f"Starting Transformers Server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
