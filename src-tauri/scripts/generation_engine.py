import argparse
import json
import os
import sys
import time
import re
import random
from typing import Dict, List, Any, Union

# Ensure UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

def log(msg: str):
    print(json.dumps({"type": "log", "message": msg}), flush=True)

def progress(current: int, total: int, msg: str = ""):
    print(json.dumps({"type": "progress", "current": current, "total": total, "message": msg}), flush=True)

class GenerationEngine:
    def __init__(self, recipe: Dict[str, Any], output_path: str):
        self.recipe = recipe
        self.output_path = output_path
        self.steps = recipe.get("steps", [])
        self.target_count = recipe.get("targetCount", 10)
        self.sources = recipe.get("sources", [])
        self.model = None
        self.model_path = None
        
        # Source State for sequential iteration
        # Map source_id -> { items: [], index: 0, mode: 'sequential'|'random' }
        self.source_state = {} 
        self.initialize_sources()

        # Initialize model if needed
        self.initialize_model()

    def initialize_sources(self):
        """Pre-load sources or set up iterators."""
        for src in self.sources:
            src_id = src.get("id")
            mode = src.get("iterationMode", "sequential")
            path = src.get("path")
            src_type = src.get("type")
            
            items = []
            if src_type == "folder" and os.path.exists(path):
                # Load file paths
                exts = tuple(src.get("filter") or ['jpg', 'png', 'jpeg', 'webp', 'txt', 'md', 'json'])
                try:
                    all_files = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(exts)]
                    # Sort for deterministic sequential
                    all_files.sort()
                    items = all_files
                except Exception as e:
                    log(f"Error loading source {src.get('name')}: {e}")
            elif src_type == "file" and os.path.exists(path):
                # Load lines
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        items = [line.strip() for line in f if line.strip()]
                except Exception as e:
                    log(f"Error reading source file {src.get('name')}: {e}")

            if mode == 'shuffled':
                random.shuffle(items)

            self.source_state[src_id] = {
                "items": items,
                "index": 0,
                "mode": mode,
                "allowRepetition": src.get("allowRepetition", False),
                "name": src.get("name")
            }
            log(f"Initialized source '{src.get('name')}' with {len(items)} items ({mode})")

    def initialize_model(self):
        # Scan steps for model requirements
        for step in self.steps:
             # Basic logic: use first found local model
             config = step.get("config", {})
             m = config.get("model")
             if m and not m.startswith("http") and not m in ["gpt-4o", "claude-3-opus"]:
                 self.model_path = m
                 break
        
        if self.model_path:
            try:
                from llama_cpp import Llama
                log(f"Loading local model: {self.model_path}")
                self.model = Llama(
                    model_path=self.model_path,
                    n_ctx=2048, # Increase context
                    n_gpu_layers=-1, 
                    verbose=False
                )
                log("Model loaded.")
            except ImportError:
                log("llama-cpp-python not installed.")
            except Exception as e:
                log(f"Failed to load model: {e}")

    def get_source_item(self, source_id: str) -> str:
        state = self.source_state.get(source_id)
        if not state or not state["items"]:
            return f"[Missing Source: {source_id}]"
        
        items = state["items"]
        if state["mode"] == "random":
            if state.get("allowRepetition", True):
                return random.choice(items)
            else:
                # Random without replacement logic? 
                # For now just random with replacement is safest for long runs
                return random.choice(items)
        
        # Sequential / Shuffled (pre-shuffled)
        idx = state["index"]
        item = items[idx % len(items)] # Loop around
        # Only increment index once per generation cycle? 
        # Actually this method might be called multiple times per row generator.
        # If we want "per row" consistency, we need to pass a context index.
        # For now, let's just cycle.
        # Better: Start of 'run_row', we snapshot the current indices? 
        # Or just increment access counters. Side effect: multiple blocks using same source = consecutive items.
        state["index"] += 1 
        return item

    def resolve_template(self, template: List[Union[str, Dict]], context: Dict) -> str:
        """
        Resolve a list of strings and blocks into a final string.
        Recursive resolution for generators.
        """
        result = ""
        for part in template:
            if isinstance(part, str):
                result += part
            elif isinstance(part, dict):
                # It's a block
                b_type = part.get("type")
                if b_type == "text": 
                    result += part.get("content", "")
                elif b_type == "source_data":
                    # Get content from source
                    src_id = part.get("sourceId")
                    item = self.get_source_item(src_id)
                    # If item is file path (image), we might need special handling?
                    # "Text" resolution implies we want text. 
                    # If it's an image path, return path or base64? 
                    # If the block is inside a text prompt, commonly we want the PATH or Description.
                    # Standard: Return string representation.
                    result += str(item)
                elif b_type == "generator":
                    # Recursive generation
                    prompt_template = part.get("prompt", [])
                    prompt_text = self.resolve_template(prompt_template, context)
                    
                    # Call LLM
                    if self.model:
                        # Simple completion
                        resp = self.model.create_completion(
                            prompt=f"User: {prompt_text}\nAssistant:",
                            max_tokens=128,
                            stop=["User:", "\n"]
                        )
                        gen_text = resp["choices"][0]["text"].strip()
                        result += gen_text
                    else:
                        result += f"[Generated: {prompt_text[:20]}...]"
        return result

    def generate_chat(self, messages: List[Dict]) -> str:
        if self.model:
            output = self.model.create_chat_completion(
                messages=messages,
                max_tokens=512,
                temperature=0.7
            )
            return output['choices'][0]['message']['content']
        else:
            return "[Mock Response]"

    def run(self):
        log(f"Starting generation for recipe: {self.recipe.get('name')}")
        
        # Determine count
        # If any source is "Sequential", we might limit by its length if targetCount is 0?
        # Use targetCount from recipe
        total = self.target_count
        
        results = []
        
        for i in range(total):
            row = {}
            row_messages = []
            
            # Context for this row
            context = {"index": i}
            
            for step in self.steps:
                try:
                    config = step.get("config", {})
                    # role = config.get("role", "user") # Not strictly in config but implied by structure?
                    # The UI saves role in config.
                    
                    # Resolve Templates
                    user_tmpl = config.get("userTemplate", [])
                    asst_tmpl = config.get("assistantTemplate", [])
                    
                    if user_tmpl:
                        user_content = self.resolve_template(user_tmpl, context)
                        # Check if user_content contains image path? 
                        # Simple heuristic for now: we are text-only unless we specifically enable multimodal blocks.
                        row_messages.append({"role": "user", "content": user_content})
                        
                    if asst_tmpl:
                        # Assistant template might be purely generative (just a generator block)
                        # OR it might be "Generate textual response based on history"
                        
                        # If template is empty/just text, we use it. 
                        # If it contains Generator, we use it.
                        # BUT, standard "Assistant" step usually implies "Ask Model to Reply to History".
                        # If template is empty, do we auto-generate?
                        # New Logic: Explicit blocks ONLY.
                        
                        asst_content = self.resolve_template(asst_tmpl, context)
                        
                        # If the resolved content is empty, SHOULD we auto-generate from history?
                        # Users usually drag "Generator" block if they want generation.
                        # But "Conversation Turn" implies generation.
                        # Let's say if template is strictly empty string [''], we generate from history.
                        if not asst_content and self.model:
                             asst_content = self.generate_chat(row_messages) # Generate based on current history
                             
                        row_messages.append({"role": "assistant", "content": asst_content})
                        
                except Exception as e:
                    log(f"Error in step {step.get('name')}: {e}")
            
            row["conversations"] = row_messages
            results.append(row)
            progress(i + 1, total, f"Generated row {i+1}/{total}")
            
            # Simple rate limit/yield
            time.sleep(0.05)

        self.save_results(results)
        log("Generation complete.")

    def save_results(self, rows: List[Dict]):
        os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
        with open(self.output_path, 'w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    try:
        recipe_data = json.loads(args.recipe)
        engine = GenerationEngine(recipe_data, args.output)
        engine.run()
    except Exception as e:
        import traceback
        print(json.dumps({"type": "error", "message": f"{str(e)}\n{traceback.format_exc()}"}), flush=True)
