import cv2
import numpy as np
import mss
import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import time
from concurrent.futures import ThreadPoolExecutor
import threading
import base64
import requests
from PIL import Image
from io import BytesIO
import argparse # Added for command-line arguments

# ============================================================================
# LLM INTEGRATION
# ============================================================================

class LLMDescriber:
    """Uses VLM to describe UI elements"""
    
    def __init__(self, args: argparse.Namespace): # Accepts args instead of config dict
        self.server_url = args.llm_server
        self.timeout = args.llm_timeout
        self.description_file = Path(args.output_dir) / args.description_file
        self.description_cache = {}  # Cache by template name
        self.args = args # Store args for later access
        
        self._load_cache()
        
        # Test connection
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            print(f"✅ LLM server connected: {self.server_url}")
        except:
            print(f"⚠️  Warning: Could not connect to LLM server at {self.server_url}")
            
    def _load_cache(self):
        """Load descriptions from the persistent file."""
        if not self.description_file or not self.description_file.exists():
            return
        
        print(f"📖 Loading descriptions from {self.description_file}...")
        try:
            with open(self.description_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Expected format: template_name: {"description": "...", "prompt": "..."}
                    if ':' in line:
                        template_name, json_str = line.split(':', 1)
                        template_name = template_name.strip()
                        
                        try:
                            data = json.loads(json_str.strip())
                            if "description" in data and "prompt" in data:
                                self.description_cache[template_name] = data
                        except json.JSONDecodeError as e:
                            print(f"  ⚠️  Skipping malformed line in description file: {line[:50]}...")
                            
            print(f"✅ Loaded {len(self.description_cache)} descriptions.")
        except Exception as e:
            print(f"❌ Error loading description file: {e}")

    def _save_cache_entry(self, template_name: str, data: Dict):
        """Append a new description entry to the persistent file."""
        if not self.description_file or not self.args.cache_descriptions: # Use args
            return
        
        line = f"{template_name}: {json.dumps(data, ensure_ascii=False)}\n"
        
        try:
            with open(self.description_file, 'a', encoding='utf-8') as f:
                f.write(line)
        except Exception as e:
            print(f"❌ Error writing description entry for {template_name}: {e}")
    
    def resize_image_for_context(self, img: np.ndarray, max_size: int = 1024) -> np.ndarray:
        """Resizes the image to fit within max_size on the longest side."""
        h, w = img.shape[:2]
        if max(h, w) <= max_size:
            return img
        
        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Use PIL for high-quality resizing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Convert back to numpy array (BGR format for consistency)
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def crop_element(self, screenshot: np.ndarray, x: int, y: int,
                     template_width: int, template_height: int, padding: int = 20) -> np.ndarray:
        """Crop element from screenshot with padding"""
        h, w = screenshot.shape[:2]
        
        # Calculate crop bounds with padding
        left = max(0, x - template_width // 2 - padding)
        right = min(w, x + template_width // 2 + padding)
        top = max(0, y - template_height // 2 - padding)
        bottom = min(h, y + template_height // 2 + padding)
        
        return screenshot[top:bottom, left:right]
    
    def numpy_to_base64(self, img: np.ndarray) -> str:
        """Convert numpy array to base64"""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        buffer = BytesIO()
        pil_img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def describe_element(self, screenshot: np.ndarray, x: int, y: int,
                        template_name: str, template_width: int,
                        template_height: int, max_retries: int = 3) -> Dict[str, str]:
        """
        Ask LLM to describe the UI element and generate a click prompt
        
        Returns:
            {
                "description": "what the element is",
                "prompt": "natural language instruction to click it"
            }
        """
        
        # Check cache first
        if template_name in self.description_cache:
            return self.description_cache[template_name]
        
        # If caching is disabled, we still proceed to generate, but we won't save to file/memory
        if not self.args.cache_descriptions: # Use args
            print(f"  ⚠️  Cache disabled, generating description for {template_name}")
        
        # Crop element with context
        cropped = self.crop_element(
            screenshot, x, y, template_width, template_height,
            padding=self.args.crop_padding # Use args
        )
        
        # Check if the cropped image is valid (not empty)
        if cropped.size == 0:
            print(f"  ❌ Cropped image for {template_name} is empty. Skipping LLM.")
            return self._fallback_description(template_name)
 
        # Convert images to base64
        resized_screenshot = self.resize_image_for_context(screenshot)
        img_b64_full = self.numpy_to_base64(resized_screenshot)
        img_b64_cropped = self.numpy_to_base64(cropped)
        
        # Log image sizes for debugging
        print(f"  🖼️ Full image size: {resized_screenshot.shape[1]}x{resized_screenshot.shape[0]}. Cropped size: {cropped.shape[1]}x{cropped.shape[0]}")
        
        # Build LLM prompt
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a UI/UX expert. Describe UI elements concisely and naturally. "
                    "Format your response as JSON with two fields:\n"
                    "1. 'description': What the element is (e.g., 'search button', 'close icon', 'menu item')\n"
                    "2. 'prompt': A natural instruction to click it (e.g., 'Click the search button', 'Close this window')\n\n"
                    "The first image is the full screen for context. The second image is a zoomed-in view of the element of interest. "
                    "Use the full screen context to refine your description of the element. "
                    "Respond ONLY with JSON in the specified format. Do not include any other text or markdown fences."
                )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64_full}"}
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64_cropped}"}
                    },
                    {
                        "type": "text",
                        "text": (
                            "Analyze the element shown in the zoomed image, using the full screen image for context. "
                            "What UI element is this? "
                            "Respond ONLY with JSON in this exact format:\n"
                            '{"description": "brief description", "prompt": "click instruction"}'
                        )
                    }
                ]
            }
        ]
        
        for attempt in range(max_retries):
            try:
                # Query LLM
                payload = {
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 200,
                    "stream": False,
                }
                
                response = requests.post(
                    f"{self.server_url}/v1/chat/completions",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    content = response.json()['choices'][0]['message']['content']
                    
                    # Attempt to parse JSON response
                    content = content.strip()
                    
                    # Aggressively strip markdown fences if present, but prioritize raw JSON
                    if content.startswith("```"):
                        if "```json" in content:
                            content = content.split("```json")[1].split("```")[0].strip()
                        else:
                            content = content.split("```")[1].split("```")[0].strip()
                    
                    try:
                        result = json.loads(content)
                        
                        # Validate response
                        if "description" in result and "prompt" in result:
                            # Cache result and persist to file
                            if self.args.cache_descriptions: # Use args
                                self.description_cache[template_name] = result
                                self._save_cache_entry(template_name, result)
                            return result
                        else:
                            print(f"  ⚠️  LLM response validation failed (missing keys). Raw content: {content}")
                            time.sleep(1) # Wait before retry
                            continue
                            
                    except json.JSONDecodeError as e:
                        print(f"  ⚠️  LLM response JSON decode failed (Attempt {attempt+1}/{max_retries}). Error: {e}. Raw content: {content}")
                        time.sleep(1) # Wait before retry
                        continue
                else:
                    print(f"  ⚠️  LLM HTTP request failed with status {response.status_code} (Attempt {attempt+1}/{max_retries}). Response: {response.text[:100]}...")
                    time.sleep(1) # Wait before retry
                    continue
                    
            except requests.exceptions.RequestException as e:
                print(f"  ⚠️  LLM connection error (Attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(1) # Wait before retry
                continue
            except Exception as e:
                print(f"  ⚠️  Unexpected error during LLM query (Attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(1) # Wait before retry
                continue
        
        # Fallback description after all retries fail
        print(f"  ⚠️  LLM failed for {template_name} after {max_retries} attempts, using fallback")
        return self._fallback_description(template_name)
        
    def _fallback_description(self, template_name: str) -> Dict[str, str]:
        """Generates a fallback description."""
        readable_name = template_name.replace('_', ' ').replace('-', ' ')
        return {
            "description": f"{readable_name} element",
            "prompt": f"Click the {readable_name}"
        }

# ============================================================================
# PARALLEL TEMPLATE MATCHER
# ============================================================================

def match_single_template(args_tuple): # Renamed config to args_tuple
    """Worker function for parallel template matching"""
    screenshot, template_name, template_data, args = args_tuple # Unpack args
    
    template_img = template_data["image"]
    
    if (template_img.shape[0] > screenshot.shape[0] or 
        template_img.shape[1] > screenshot.shape[1]):
        return {
            "template_name": template_name,
            "match": None,
            "found": False,
            "confidence": 0.0
        }
    
    result = cv2.matchTemplate(screenshot, template_img, args.match_method) # Use args.match_method
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    confidence = max_val
    
    if confidence >= args.match_threshold: # Use args.match_threshold
        center_x = max_loc[0] + template_img.shape[1] // 2
        center_y = max_loc[1] + template_img.shape[0] // 2
        
        return {
            "template_name": template_name,
            "match": (center_x, center_y, confidence),
            "found": True,
            "confidence": confidence
        }
    else:
        return {
            "template_name": template_name,
            "match": None,
            "found": False,
            "confidence": 0.0
        }


class ParallelCPUMatcher:
    """Parallel CPU-based template matcher"""
    
    def __init__(self, templates: Dict, args: argparse.Namespace): # Accepts args
        self.args = args # Store args
        self.templates = templates
        self.num_workers = args.num_workers # Use args
        
        print(f"\n⚡ Parallel CPU Matcher Initialized")
        print(f"   Workers: {self.num_workers}")
        print(f"   Templates: {len(templates)}")
    
    def match_all_templates_parallel(self, screenshot: np.ndarray) -> List[Dict]:
        """Match all templates in parallel using thread pool"""
        worker_args = [
            (screenshot, name, data, self.args) # Pass args
            for name, data in self.templates.items()
        ]
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(match_single_template, worker_args))
        
        return results


# ============================================================================
# DATASET GENERATOR
# ============================================================================

class LLMPoweredDatasetGenerator:
    """Dataset generator with LLM-based element descriptions"""
    
    def __init__(self, args: argparse.Namespace): # Accepts args
        self.args = args # Store args
        self.templates = {}
        self.screen_capture = mss.mss()
        self.dataset_buffer = []
        self.write_lock = threading.Lock()
        
        self._setup_output_dirs()
        self._load_templates()
        
        # Initialize matcher and LLM describer
        description_file_path = Path(args.output_dir) / args.description_file # Use args
        
        self.matcher = ParallelCPUMatcher(self.templates, args) # Pass args
        self.llm_describer = LLMDescriber(args) # Pass args
        
        print(f"\n🚀 LLM-POWERED DATASET GENERATOR READY")
    
    def _setup_output_dirs(self):
        """Create dataset/images/ structure"""
        dataset_dir = Path(self.args.output_dir) # Use args
        images_dir = dataset_dir / self.args.screenshots_dir # Use args
        
        dataset_dir.mkdir(exist_ok=True)
        images_dir.mkdir(exist_ok=True)
        
        print(f"✅ Output: {dataset_dir}")
        print(f"   Images: {images_dir}")
    
    def _load_templates(self):
        templates_path = Path(self.args.templates_dir) # Use args
        
        if not templates_path.exists():
            print(f"⚠️ Templates directory not found: {templates_path}. Creating it.")
            templates_path.mkdir(parents=True, exist_ok=True)
        
        supported_formats = {'.png', '.jpg', '.jpeg', '.bmp'}
        # Recursively search for template files in the directory and its subdirectories
        template_files = [
            f for f in templates_path.rglob('*')
            if f.is_file() and f.suffix.lower() in supported_formats
        ]
        
        print(f"\n📂 Loading {len(template_files)} templates...")
        
        for template_file in template_files:
            template_img = cv2.imread(str(template_file))
            if template_img is None:
                continue
            
            template_name = template_file.stem
            self.templates[template_name] = {
                "name": template_name,
                "image": template_img,
                "width": template_img.shape[1],
                "height": template_img.shape[0],
                "filename": template_file.name,
            }
        
        print(f"✅ Loaded {len(self.templates)} templates")
    
    def capture_screenshot(self) -> Tuple[np.ndarray, Tuple[int, int]]:
        monitor = self.screen_capture.monitors[self.args.monitor_index] # Use args
        screenshot = self.screen_capture.grab(monitor)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        screen_size = (monitor['width'], monitor['height'])
        return img, screen_size
    
    def _flush_dataset_buffer(self):
        """Write buffered dataset entries to disk"""
        if not self.dataset_buffer:
            return
        
        dataset_path = Path(self.args.output_dir) / self.args.dataset_file # Use args
        
        with self.write_lock:
            with open(dataset_path, 'a', encoding='utf-8') as f:
                for entry in self.dataset_buffer:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
            self.dataset_buffer.clear()
    
    def save_dataset_entry(self, image_filename: str, element_info: Dict,
                          match: Tuple[int, int, float], screen_size: Tuple[int, int]):
        """
        Save entry in the exact format the agent expects
        Matches the format from your agent script
        """
        x, y, confidence = match
        screen_width, screen_height = screen_size
        
        # Normalize coordinates to 1-1000 range (as agent expects)
        norm_x = max(1, min(1000, int((x / screen_width) * 1000)))
        norm_y = max(1, min(1000, int((y / screen_height) * 1000)))
        
        # Format exactly like the agent's training data
        entry = {
            "image": f"{self.args.screenshots_dir}/{image_filename}", # Use args
            "conversations": [
                {
                    "role": "user", 
                    "content": f"\n{element_info['prompt']}"
                },
                {
                    "role": "assistant", 
                    "content": f"<tool_call>{{\"name\": \"computer_use\", \"arguments\": {{\"action\": \"left_click\", \"coordinate\": [{norm_x}, {norm_y}]}}}}</tool_call>"
                }
            ],
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "template_name": element_info['template_name'],
                "description": element_info['description'],
                "coordinates_normalized": [norm_x, norm_y],
                "coordinates_raw": [x, y],
                "screen_size": screen_size,
                "confidence": float(confidence)
            }
        }
        
        self.dataset_buffer.append(entry)
        
        if len(self.dataset_buffer) >= self.args.dataset_write_buffer: # Use args
            self._flush_dataset_buffer()
    
    def process_screenshot_with_llm(self, iteration: int = 0, verbose: bool = True) -> Dict:
        """Process screenshot and generate LLM-powered descriptions"""
        
        # Capture
        t0 = time.perf_counter()
        screenshot, screen_size = self.capture_screenshot()
        t_capture = time.perf_counter() - t0
        
        # Save image
        t1 = time.perf_counter()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"screenshot_{timestamp}_{iteration:04d}.png"
        image_path = Path(self.args.output_dir) / self.args.screenshots_dir / image_filename # Use args
        if not self.args.skip_image_save: # Use args
            cv2.imwrite(str(image_path), screenshot)
        t_save = time.perf_counter() - t1
        
        # Match templates
        t2 = time.perf_counter()
        results = self.matcher.match_all_templates_parallel(screenshot)
        t_match = time.perf_counter() - t2
        
        # Process results with LLM
        t3 = time.perf_counter()
        stats = {"found": 0, "not_found": 0, "llm_described": 0}
        
        for result in results:
            if result["found"]:
                template_name = result["template_name"]
                template_data = self.templates[template_name]
                x, y, confidence = result["match"]
                
                # Get LLM description
                if verbose:
                    print(f"  🤖 Describing: {template_name}...", end=" ")
                
                element_info = self.llm_describer.describe_element(
                    screenshot, x, y,
                    template_name,
                    template_data["width"],
                    template_data["height"]
                )
                
                if element_info is None:
                    # This should ideally not happen based on describe_element implementation,
                    # but we handle it defensively.
                    print(f"  ❌ Failed to get element info for {template_name}. Skipping.")
                    stats["not_found"] += 1
                    continue
                    
                element_info["template_name"] = template_name
                
                if verbose:
                    print(f"✓ '{element_info['prompt']}'")
                
                # Save to dataset
                self.save_dataset_entry(
                    image_filename,
                    element_info,
                    result["match"],
                    screen_size
                )
                
                stats["found"] += 1
                stats["llm_described"] += 1
            else:
                stats["not_found"] += 1
        
        t_process = time.perf_counter() - t3
        total_time = time.perf_counter() - t0
        
        return {
            "total": total_time,
            "capture": t_capture,
            "save": t_save,
            "match": t_match,
            "process": t_process,
            "templates_per_sec": len(results) / t_match if t_match > 0 else 0,
            **stats
        }
    
    def run_continuous_collection(self, num_iterations: int = 100):
        print(f"\n{'='*60}")
        print(f"🚀 CONTINUOUS COLLECTION WITH LLM DESCRIPTIONS")
        print(f"{'='*60}")
        
        total_stats = []
        
        try:
            for i in range(num_iterations):
                print(f"\n[Iteration {i+1}/{num_iterations}]")
                
                stats = self.process_screenshot_with_llm(iteration=i, verbose=True)
                total_stats.append(stats)
                
                print(f"  ⏱️  Time: {stats['total']:.2f}s | "
                      f"Found: {stats['found']} | "
                      f"LLM: {stats['llm_described']}")
                
                # Maintain framerate
                if i < num_iterations - 1:
                    time.sleep(max(0, self.args.screenshot_delay - stats["total"])) # Use args
        
        finally:
            if self.args.batch_write_dataset: # Use args
                self._flush_dataset_buffer()
        
        # Final stats
        total_found = sum(s["found"] for s in total_stats)
        total_llm = sum(s["llm_described"] for s in total_stats)
        
        print(f"\n{'='*60}")
        print(f"🏁 COMPLETE")
        print(f"{'='*60}")
        print(f"Iterations: {num_iterations}")
        print(f"Total matches: {total_found}")
        print(f"LLM descriptions: {total_llm}")
        print(f"Dataset: {Path(self.args.output_dir) / self.args.dataset_file}") # Use args
        print(f"{'='*60}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-Powered Template Matching Dataset Generator")

    # Paths
    parser.add_argument("--templates_dir", type=str, default="./unity/editor",
                        help="Directory containing UI element templates (cropped images).")
    parser.add_argument("--output_dir", type=str, default="./dataset",
                        help="Root directory for generated dataset (contains dataset.jsonl and images/).")
    parser.add_argument("--dataset_file", type=str, default="dataset.jsonl",
                        help="Filename for the JSONL dataset.")
    parser.add_argument("--screenshots_dir", type=str, default="images",
                        help="Subdirectory within output_dir to save screenshots.")

    # LLM Server
    parser.add_argument("--llm_server", type=str, default="http://localhost:8080",
                        help="URL of the LLM inference server.")
    parser.add_argument("--llm_timeout", type=int, default=30,
                        help="Timeout for LLM requests in seconds.")
    
    # Template matching
    parser.add_argument("--match_threshold", type=float, default=0.9,
                        help="Confidence threshold for template matching (0-1).")
    parser.add_argument("--match_method", type=int, default=cv2.TM_CCOEFF_NORMED,
                        help="OpenCV template matching method (e.g., cv2.TM_CCOEFF_NORMED).")
    
    # Screenshot settings
    parser.add_argument("--monitor_index", type=int, default=1,
                        help="Index of the monitor to capture (0-indexed).")
    parser.add_argument("--screenshot_delay", type=float, default=0.0833,
                        help="Delay between screenshots in continuous collection (seconds).")
    
    # Optimization settings
    parser.add_argument("--use_multiscale", type=bool, default=False,
                        help="Whether to use multiscale template matching (experimental).")
    parser.add_argument("--scale_factors", type=float, nargs='*', default=[1.0],
                        help="List of scale factors for multiscale matching if enabled.")
    parser.add_argument("--num_workers", type=int, default=32,
                        help="Number of parallel workers for template matching.")
    
    # Speed optimizations
    parser.add_argument("--skip_image_save", type=bool, default=False,
                        help="If True, skips saving raw screenshots (only saves dataset entries).")
    parser.add_argument("--batch_write_dataset", type=bool, default=True,
                        help="If True, buffers dataset entries and writes in batches.")
    parser.add_argument("--dataset_write_buffer", type=int, default=50,
                        help="Number of entries to buffer before writing to dataset.jsonl.")
    
    # LLM description settings
    parser.add_argument("--crop_padding", type=int, default=20,
                        help="Pixels of padding around detected element for LLM context crop.")
    parser.add_argument("--cache_descriptions", type=bool, default=True,
                        help="If True, caches LLM descriptions for templates to a file.")
    parser.add_argument("--description_file", type=str, default="image_descriptions.txt",
                        help="File for persistent caching of LLM descriptions.")
    
    # Collection settings
    parser.add_argument("--num_iterations", type=int, default=100,
                        help="Number of continuous collection iterations to run.")
    
    args = parser.parse_args()

    generator = LLMPoweredDatasetGenerator(args) # Pass args to generator
    
    # Run continuous collection
    generator.run_continuous_collection(num_iterations=args.num_iterations) # Use args.num_iterations