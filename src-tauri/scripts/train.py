import os
import sys

# ============================================================================
# CRITICAL: Windows Multiprocessing Fix - MUST BE AT VERY TOP
# ============================================================================
# Disable tokenizers parallelism to prevent fork bombs on Windows
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Force single-process for datasets mapping on Windows
os.environ["HF_DATASETS_DISABLE_CACHING"] = "1"

import torch
import json
import argparse
import logging
# Import datasets to configure it
import datasets
datasets.disable_progress_bar()
datasets.logging.set_verbosity_info()

from typing import Optional, Dict, Any
from pathlib import Path

# ============================================================================
# 1. WINDOWS COMPATIBILITY PATCHES
# ============================================================================

# Patch torch._inductor for Windows
if hasattr(torch, "_inductor") and not hasattr(torch._inductor, "config"):
    class DummyInductorConfig:
        pass
    torch._inductor.config = DummyInductorConfig

# Auto-install missing dependencies (specifically tensorboard for Windows)
try:
    import tensorboard
except ImportError:
    print("Warning: tensorboard not found. Attempting to install...")
    import subprocess
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorboard"])
        print("Successfully installed tensorboard.")
    except Exception as e:
        print(f"Failed to auto-install tensorboard: {e}")
        print("Please install it manually: pip install tensorboard")

# ============================================================================
# 2. IMPORTS
# ============================================================================

# CRITICAL: Import psutil BEFORE unsloth and inject into builtins
# Unsloth's compiled cache (UnslothSFTTrainer.py) uses psutil.cpu_count()
# but doesn't import it, expecting it in global namespace
import psutil
import builtins
builtins.psutil = psutil  # Make psutil available globally for Unsloth's compiled code

try:
    import unsloth
    from unsloth import FastLanguageModel, FastVisionModel
    from unsloth import UnslothVisionDataCollator
except ImportError as e:
    print(f"Warning: Unsloth not installed or failed to import: {e}")
    print("Will use standard HuggingFace + PEFT instead")
    unsloth = None
    FastLanguageModel = None
    FastVisionModel = None
    UnslothVisionDataCollator = None

# Handle potential torchvision version mismatch before importing transformers
try:
    from transformers import (
        TrainingArguments, 
        AutoConfig, 
        AutoProcessor, 
        AutoTokenizer,
        AutoModelForSpeechSeq2Seq,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        PretrainedConfig,
        CONFIG_MAPPING
    )
except RuntimeError as e:
    if "torchvision" in str(e) or "does not exist" in str(e):
        print("\n" + "="*60)
        print("❌ CRITICAL ERROR: PyTorch/TorchVision Version Mismatch")
        print("="*60)
        print(f"Error: {e}")
        print("\nThis error occurs when torch and torchvision versions are incompatible.")
        print("\nTo fix this, run the environment setup again:")
        print("  python src-tauri/scripts/setup_torch.py")
        print("\nOr manually install compatible versions:")
        print("  pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121")
        print("="*60)
        sys.exit(1)
    raise
from trl import SFTTrainer, DPOTrainer, DPOConfig
from datasets import Dataset, load_from_disk, concatenate_datasets
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import psutil  # Required by Unsloth compiled cache

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

# ============================================================================
# 3. HELPER FUNCTIONS
# ============================================================================

def resolve_model_path(path: str) -> str:
    """Robustly resolves model paths (absolute, relative, subdirs)."""
    candidates = [
        path,
        os.path.abspath(path),
        os.path.join(os.getcwd(), path),
        os.path.join(os.getcwd(), "data", "models", path),
        os.path.join(os.getcwd(), "data", "models", os.path.basename(path)),
    ]
    
    if os.path.isdir(path):
        subdirs = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        candidates.extend(subdirs)

    print(f"\nResolving model path: '{path}'")
    for candidate in candidates:
        config_path = os.path.join(candidate, "config.json")
        adapter_path = os.path.join(candidate, "adapter_config.json")
        
        if os.path.exists(config_path) or os.path.exists(adapter_path):
            print(f"  ✓ Found config at: {candidate}")
            return os.path.abspath(candidate)
    
    # If no local config found, assume HF Hub ID
    if not os.path.exists(path) and "/" in path:
         print(f"  ⚠ No local path found. Assuming HuggingFace Hub ID.")
         return path
         
    return path

def detect_model_type(model_name: str) -> str:
    """
    Inspects config to determine if model is Text, Vision, or Audio.
    Returns: 'text', 'vision', 'audio'
    """
    try:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        
        # Handle case where config is a dict
        if isinstance(config, dict):
            model_type = config.get("model_type", "").lower()
            archs = config.get("architectures", [])
        else:
            archs = getattr(config, "architectures", [])
            model_type = str(getattr(config, "model_type", "")).lower()
        
        str_archs = str(archs).lower()
        
        # Audio Detection
        if "whisper" in model_type or "speech" in str_archs or "audio" in str_archs:
            return "audio"
            
        # Vision Detection (VLMs)
        # unsloth_zoo/vision_utils.py checks for vision_config or specific archs
        vision_keywords = ["vision", "vl", "visual", "image", "mllama", "llava", "qwen2_vl", "qwen3_vl", "pixtral"]
        if any(k in model_type for k in vision_keywords) or any(k in str_archs for k in vision_keywords):
            return "vision"
        
        # Deep check for vision_config
        if hasattr(config, "vision_config") or (isinstance(config, dict) and "vision_config" in config):
            return "vision"

        # Default to Text
        return "text"
    except Exception as e:
        print(f"Warning: Could not auto-detect model type ({e}). Defaulting to 'text'.")
        return "text"

def resolve_dataset_path(dataset_arg: str) -> str:
    """
    Robustly resolves dataset paths to find the processed_data/train directory.
    """
    # 1. Clean up and check if absolute exists
    clean_arg = dataset_arg.replace("\\", "/").strip("/")
    if os.path.exists(dataset_arg) or os.path.exists(clean_arg):
        path = dataset_arg if os.path.exists(dataset_arg) else clean_arg
        # If it's the root of the dataset, look for processed_data/train
        for sub in ["processed_data/train", "processed_data", "train"]:
            sub_path = os.path.join(path, sub)
            if os.path.exists(sub_path):
                return os.path.abspath(sub_path)
        return os.path.abspath(path)

    # 2. Setup project root and dataset name
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    dataset_name = os.path.basename(clean_arg)

    # 3. Probable search bases
    search_bases = [
        project_root,
        os.path.join(project_root, "data"),
        os.path.join(project_root, "data", "datasets"),
    ]
    
    # AppData for Windows
    if os.name == 'nt':
        appdata = os.environ.get('APPDATA')
        if appdata:
            search_bases.append(os.path.join(appdata, "com.lavanukee.Velox", "data", "datasets"))

    # 4. Search
    for base in search_bases:
        if not os.path.exists(base): continue
        
        # Check direct folder match and subfolders
        check_path = os.path.join(base, dataset_name)
        candidates = [
            check_path,
            os.path.join(check_path, "processed_data", "train"),
            os.path.join(check_path, "processed_data"),
            os.path.join(check_path, "train"),
        ]
        
        for cand in candidates:
            if os.path.exists(cand):
                # Verify it has some dataset markers
                if any(os.path.exists(os.path.join(cand, f)) for f in ["dataset_info.json", "dataset.arrow", "state.json"]):
                    print(f"TAURI_BACKEND: TRAIN: ✓ Resolved dataset to: {cand}")
                    return os.path.abspath(cand)

    print(f"TAURI_BACKEND: TRAIN: ❌ Could not find dataset in any expected location")
    return dataset_arg

# ============================================================================
# 4. MAIN TRAINING LOGIC
# ============================================================================

def main(args):
    # =========================================================================
    # OOM SAFEGUARDS - Check VRAM and setup memory management
    # =========================================================================
    print("\n" + "="*60)
    print("🔍 GPU Memory Check")
    print("="*60)
    
    if torch.cuda.is_available():
        # Clear any stale CUDA cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Get VRAM info
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free_mem, total = torch.cuda.mem_get_info()
        free_mem_gb = free_mem / 1024**3
        used_mem_gb = (total - free_mem) / 1024**3
        
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM: {total_mem:.1f} GB")
        print(f"Available VRAM: {free_mem_gb:.1f} GB")
        print(f"Currently Used: {used_mem_gb:.1f} GB")
        
        # Auto-adjust batch size based on available VRAM
        original_batch_size = args.batch_size
        if free_mem_gb < 6:
            args.batch_size = 1
            print(f"⚠️ LOW VRAM (<6GB): batch_size forced to 1")
        elif free_mem_gb < 12:
            args.batch_size = min(args.batch_size, 2)
            print(f"⚠️ Moderate VRAM (<12GB): batch_size capped at {args.batch_size}")
        
        if args.batch_size != original_batch_size:
            print(f"   (was {original_batch_size}, now {args.batch_size})")
        
        # Warn if critically low
        if free_mem_gb < 4:
            print("\n" + "!"*60)
            print("⚠️ CRITICAL: Less than 4GB VRAM available!")
            print("   Consider closing other GPU-using applications.")
            print("!"*60)
    else:
        print("❌ No CUDA GPU available! Training will be very slow or fail.")
    
    print("="*60 + "\n")
    
    # --- Dataset Path Resolution (Multiple Datasets Support) ---
    PROCESSED_TRAIN_DIRS = []
    for ds_path in args.dataset_paths:
        resolved = resolve_dataset_path(ds_path)
        PROCESSED_TRAIN_DIRS.append(resolved)
        print(f"  Dataset path: {resolved}")
    
    OUTPUT_DIR = args.output_dir if hasattr(args, 'output_dir') and args.output_dir else "data/outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    model_path = resolve_model_path(args.model)
    model_category = detect_model_type(model_path)
    
    print("="*60)
    print(f"Universal Training Pipeline")
    print(f"Detected Category: {model_category.upper()}")
    print(f"Model: {model_path}")
    print("="*60)

    # Variables to be filled based on category
    model = None
    tokenizer = None
    processor = None
    use_unsloth = False  # Flag to track which method we're using
    
    # ------------------------------------------------------------------------
    # LOADER: TEXT (LLM) - Uses Unsloth FastLanguageModel with PEFT fallback
    # ------------------------------------------------------------------------
    if model_category == "text":
        # Try Unsloth first if available
        if FastLanguageModel is not None:
            print("Attempting to load with Unsloth FastLanguageModel (Text)...")
            try:
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=model_path,
                    max_seq_length=args.max_seq_length,
                    dtype=None,
                    load_in_4bit=True,
                    trust_remote_code=True,
                )
                
                model = FastLanguageModel.get_peft_model(
                    model,
                    r=args.lora_r,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    lora_alpha=args.lora_alpha,
                    lora_dropout=0,
                    bias="none", 
                    use_gradient_checkpointing="unsloth",
                    random_state=3407,
                )
                print("✓ Model loaded successfully with Unsloth!")
                use_unsloth = True
                
            except Exception as e:
                print(f"\n⚠️ Unsloth loading failed: {e}")
                print("\n🔄 Falling back to standard HuggingFace + PEFT...")
        else:
            print("Unsloth not available, using standard HuggingFace + PEFT...")
        
        # Fallback to standard HuggingFace + PEFT if Unsloth failed or not available
        if not use_unsloth:
            try:
                print("Loading tokenizer directly...")
                # Load tokenizer first WITHOUT config - let it handle its own config loading
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True
                )
                
                # Set padding token if not set
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                print("Configuring 4-bit quantization...")
                # Configure 4-bit quantization
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                
                # Load base model - let it load its own config internally
                print(f"Loading base model from {model_path}...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                )
                
                # Prepare model for k-bit training
                print("Preparing model for k-bit training...")
                model = prepare_model_for_kbit_training(model)
                
                # Enable gradient checkpointing with use_reentrant=False (recommended)
                print("Enabling gradient checkpointing...")
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                
                # Configure LoRA
                print("Configuring LoRA...")
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=0.1,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    bias="none",
                )
                
                # Apply PEFT
                print("Applying PEFT...")
                model = get_peft_model(model, peft_config)
                model.print_trainable_parameters()
                
                print("✓ Model loaded successfully with HuggingFace PEFT!")
                
            except Exception as fallback_error:
                print(f"\n❌ CRITICAL ERROR: Both Unsloth and HuggingFace PEFT failed!")
                print(f"PEFT fallback error: {fallback_error}")
                print("\nTroubleshooting steps:")
                print("1. Verify the model path is correct")
                print("2. Ensure you have enough GPU memory")
                print("3. Check that all dependencies are installed")
                print("4. Try with a smaller model first")
                import traceback
                traceback.print_exc()
                sys.exit(1)

    # ------------------------------------------------------------------------
    # LOADER: VISION (VLM) - Uses Unsloth FastVisionModel
    # ------------------------------------------------------------------------
    elif model_category == "vision":
        if FastVisionModel is not None:
            print("Initializing Unsloth FastVisionModel (Text + Image)...")
            try:
                model, tokenizer = FastVisionModel.from_pretrained(
                    model_name=model_path,
                    load_in_4bit=True,
                    use_gradient_checkpointing="unsloth",
                    trust_remote_code=True,
                )
                
                model = FastVisionModel.get_peft_model(
                    model,
                    finetune_vision_layers=True,
                    finetune_language_layers=True,
                    finetune_attention_modules=True,
                    finetune_mlp_modules=True,
                    r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=0,
                    bias="none",
                    random_state=3407,
                )
                # FORCE VISION FLAGS - Critical for Unsloth SFTTrainer detection
                print("DEBUG: Setting critical vision flags...")
                model.is_vision = True
                if hasattr(model, "config"):
                    model.config.is_vision_model = True
                    
                # Ensure vision_config exists (create dummy if needed for detection logic)
                if not hasattr(model.config, "vision_config"):
                    print("DEBUG: Creating dummy vision_config for detection compatibility")
                    model.config.vision_config = {"exists": True}

                # PATCH: Fix missing chat template for vision models (resolves Qwen3-VL issues)
                if hasattr(tokenizer, "chat_template") and (tokenizer.chat_template is None or tokenizer.chat_template == ""):
                    print("⚠️ Vision processor/tokenizer is missing a chat_template. Applying robust Qwen2-VL fallback.")
                    # Standard Qwen2-VL / Qwen3-VL chat template fallback (handles string and list content)
                    tokenizer.chat_template = (
                        "{%- for message in messages -%}"
                        "{{- '<|im_start|>' + message['role'] + '\n' -}}"
                        "{%- if message['content'] is string -%}"
                        "{{- message['content'] -}}"
                        "{%- else -%}"
                        "{%- for item in message['content'] -%}"
                        "{%- if item['type'] == 'text' -%}"
                        "{{- item['text'] -}}"
                        "{%- elif item['type'] == 'image' -%}"
                        "{{- '<|vision_start|><|placeholder_output|><|vision_end|>' -}}"
                        "{%- endif -%}"
                        "{%- endfor -%}"
                        "{%- endif -%}"
                        "{{- '<|im_end|>\n' -}}"
                        "{%- endfor -%}"
                        "{%- if add_generation_prompt -%}"
                        "{{- '<|im_start|>assistant\n' -}}"
                        "{%- endif -%}"
                    )
                
                # Manually tag as vision to satisfy SFTTrainer checks if it happens to fail detection
                if not hasattr(model, "is_vision"):
                    model.is_vision = True

                print("✓ Vision model loaded successfully!")
                use_unsloth = True
                
            except Exception as e:
                print(f"Failed to load Vision model with Unsloth: {e}")
                # Debug: List directory contents to see what's actually there
                try:
                    # import os  <-- Removed to prevent UnboundLocalError (os is already global)
                    print(f"DEBUG: Contents of {model_path}:")
                    for fname in os.listdir(model_path):
                        print(f"  - {fname}")
                except Exception as list_err:
                    print(f"Could not list directory: {list_err}")

                print("Vision models require Unsloth. Please check your installation or model integrity.")
                sys.exit(1)
        else:
            print("❌ Vision models require Unsloth, but it's not available.")
            sys.exit(1)

    # ------------------------------------------------------------------------
    # LOADER: AUDIO - Fallback to Standard HuggingFace
    # ------------------------------------------------------------------------
    elif model_category == "audio":
        print("Initializing Standard HuggingFace Model (Audio)...")
        print("NOTE: Unsloth acceleration is not available for Audio yet. Using standard PEFT.")
        
        try:
            processor = AutoProcessor.from_pretrained(model_path)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_path,
                load_in_8bit=True,
                device_map="auto"
            )
            
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM, 
                inference_mode=False, 
                r=args.lora_r, 
                lora_alpha=args.lora_alpha, 
                lora_dropout=0.1
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
            print("✓ Audio model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading audio model: {e}")
            sys.exit(1)
            
    # ------------------------------------------------------------------------
    # DATA LOADING & FORMATTING (Multiple Datasets Support)
    # ------------------------------------------------------------------------
    print(f"\nLoading {len(PROCESSED_TRAIN_DIRS)} dataset(s)...")
    datasets_list = []
    try:
        for ds_dir in PROCESSED_TRAIN_DIRS:
            print(f"  Loading: {ds_dir}")
            ds = Dataset.load_from_disk(ds_dir)
            print(f"    ✓ {len(ds)} examples, columns: {ds.column_names}")
            datasets_list.append(ds)
        
        # Concatenate if multiple datasets
        if len(datasets_list) > 1:
            print(f"\nConcatenating {len(datasets_list)} datasets...")
            dataset = concatenate_datasets(datasets_list)
            # Shuffle to mix examples from different sources
            dataset = dataset.shuffle(seed=42)
            print(f"✓ Combined dataset: {len(dataset)} examples")
        else:
            dataset = datasets_list[0]
            print(f"\n✓ Dataset loaded: {len(dataset)} examples")
        
        print(f"  Final columns: {dataset.column_names}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        sys.exit(1)
    
    # --- Formatter for TEXT Models ---
    if model_category == "text":
        if "text" not in dataset.column_names:
            print("Formatting dataset for Text Model...")
            
            # Use Unsloth's chat template utility if available and requested
            if use_unsloth and hasattr(args, 'chat_template') and args.chat_template and args.chat_template != "none":
                print(f"Applying Unsloth Chat Template: {args.chat_template}")
                from unsloth.chat_templates import get_chat_template
                
                try:
                    tokenizer = get_chat_template(
                        tokenizer,
                        chat_template = args.chat_template,
                        mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # Default ShareGPT style
                    )
                    
                    def formatting_prompts_func(examples):
                        # Assumes dataset has "conversations" or similar structure if using strict chat templates
                        # If dataset is just instruction/input/output, we need to convert it first for the template
                        convos = []
                        texts = []
                        for i in range(len(examples["instruction"])):
                            # Convert Alpaca style to simplistic "chat" for template application if needed
                            # But standard get_chat_template expects a specific structure usually.
                            
                            # FALLBACK: If dataset is Alpaca (instruction/input/output), standard template might be overkill
                            # UNLESS we manually construct the message list.
                            
                            inst = examples["instruction"][i]
                            inp = examples["input"][i]
                            out = examples["output"][i]
                            
                            # Construct a "conversation" for the chat template
                            user_text = f"{inst}\n{inp}" if inp else inst
                            messages = [
                                {"role": "user", "content": user_text},
                                {"role": "assistant", "content": out}
                            ]
                            
                            # Apply template
                            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                            texts.append(text)
                            
                        return {"text": texts, }
                        
                    dataset = dataset.map(formatting_prompts_func, batched=True)
                    
                except Exception as e:
                    print(f"Warning: Failed to apply Unsloth chat template: {e}. Falling back to default Alpaca format.")
                    # Fallback logic below
                    
            # Default Alpaca-style formatting if no template specified or fallback
            if "text" not in dataset.column_names: 
                print("Using default formatting logic...")
                
                def format_text(examples):
                    # ShareGPT style
                    if "conversations" in examples:
                        convos = examples["conversations"]
                        texts = []
                        # convos is list of lists of dicts if batched, or list of dicts if not?
                        # If batched=False, examples is a dict of values. examples['conversations'] is the list of messages.
                        # If batched=False:
                        if isinstance(convos, list) and len(convos) > 0 and isinstance(convos[0], dict):
                             # Single example processing (if mapped with batched=False)
                             # Construct text from conversation
                             c_text = ""
                             for msg in convos:
                                 role = msg.get("role", "")
                                 content = msg.get("content", "")
                                 c_text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
                             c_text += "<|im_start|>assistant\n" # Generation prompt?
                             return {"text": c_text}
                        
                        # If mapped with batched=True? The code below uses batched=False for this fallback
                        return examples

                    # Alpaca style
                    if "instruction" in examples and "output" in examples:
                        inst = examples["instruction"]
                        inp = examples.get("input", "")
                        out = examples["output"]
                        text = f"Instruction: {inst}\nInput: {inp}\nOutput: {out}" if inp else f"Instruction: {inst}\nOutput: {out}"
                        return {"text": text}
                        
                    return examples
            
                dataset = dataset.map(format_text, batched=False)
                
            if "text" in dataset.column_names:
                print(f"✓ Dataset formatted. Sample: {dataset[0]['text'][:100]}...")
            else:
                print("Warning: Dataset could not be formatted. 'text' column missing. Columns found:", dataset.column_names)
            
    # ------------------------------------------------------------------------
    # TRAINING ARGUMENTS
    # ------------------------------------------------------------------------
    print("\nConfiguring training arguments...")
    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=OUTPUT_DIR,
        logging_dir=OUTPUT_DIR,
        report_to="tensorboard",
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        dataloader_num_workers=0, # Force single process for dataloader on Windows
        remove_unused_columns=False if model_category == "vision" else True, # Vision collator needs raw image/conversations
    )

    # ------------------------------------------------------------------------
    # TRAINER INITIALIZATION
    # ------------------------------------------------------------------------
    print("Initializing Trainer...")
    
    if model_category == "vision":
        from unsloth import UnslothVisionDataCollator
        from PIL import Image
        import io
        import base64
        import re
        
        columns = set(dataset.column_names)
        print(f"  Vision dataset columns: {columns}")
        
        # Unsloth VLM expects 'messages' column with multimodal content format
        # Each message should be: {"role": "...", "content": [{"type": "image", ...}, {"type": "text", ...}]}
        # The UnslothVisionDataCollator handles this format directly - no pre-processing needed
        
        if 'messages' not in columns:
            # Fallback: try to convert other formats to 'messages'
            if 'conversations' in columns:
                print("  Converting 'conversations' to 'messages' format...")
                def convert_to_messages(example):
                    return {"messages": example['conversations']}
                dataset = dataset.map(convert_to_messages, num_proc=1)
            elif 'text' in columns:
                print("  Converting 'text' to 'messages' format for vision training...")
                
                # Check if dataset has image column or image embedded in text
                has_image_column = 'image' in columns or 'images' in columns
                
                def convert_text_to_messages(example):
                    """
                    Convert text-based dataset to messages format for UnslothVisionDataCollator.
                    Handles various formats from HuggingFace datasets.
                    """
                    text = example.get('text', '')
                    messages = []
                    
                    # Try to get image from dataset columns
                    image = None
                    if 'image' in example and example['image'] is not None:
                        image = example['image']
                    elif 'images' in example and example['images']:
                        image = example['images'][0] if isinstance(example['images'], list) else example['images']
                    
                    # Parse the text to extract conversation structure
                    # Common formats: ChatML (<|im_start|>...<|im_end|>), Alpaca, simple user/assistant
                    
                    # Try ChatML format first
                    chatml_pattern = r'<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>'
                    chatml_matches = re.findall(chatml_pattern, text, re.DOTALL)
                    
                    if chatml_matches:
                        for role, content in chatml_matches:
                            content = content.strip()
                            # Check for image tags in content
                            if role == 'user' and image is not None:
                                # First user message gets the image
                                messages.append({
                                    "role": role,
                                    "content": [
                                        {"type": "image", "image": image},
                                        {"type": "text", "text": content.replace('<|vision_start|>', '').replace('<|vision_end|>', '').replace('<|placeholder_output|>', '').strip()}
                                    ]
                                })
                                image = None  # Only use image once
                            else:
                                messages.append({
                                    "role": role,
                                    "content": content
                                })
                    else:
                        # Fallback: treat entire text as a single user->assistant exchange
                        # This is a last resort for unstructured text
                        if image is not None:
                            messages = [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "image", "image": image},
                                        {"type": "text", "text": "Describe this image."}
                                    ]
                                },
                                {
                                    "role": "assistant", 
                                    "content": text[:500] if len(text) > 500 else text  # Truncate very long texts
                                }
                            ]
                        else:
                            # No image found - create placeholder structure
                            # This will likely still fail but provides better error messages
                            messages = [
                                {"role": "user", "content": [{"type": "text", "text": text[:500]}]},
                                {"role": "assistant", "content": "I understand."}
                            ]
                    
                    return {"messages": messages}
                
                print("  Applying text->messages conversion (single process for Windows)...")
                dataset = dataset.map(convert_text_to_messages, num_proc=1)
                print(f"  ✓ Conversion complete. New columns: {dataset.column_names}")
                
            elif 'instruction' in columns and ('image' in columns or 'images' in columns):
                print("  Converting 'instruction' + 'image' to 'messages' format for vision training...")
                
                def convert_instruct_to_messages(example):
                    # Extract fields
                    instruction = example.get('instruction', '')
                    
                    # Handle image
                    image = None
                    if 'image' in example and example['image'] is not None:
                        image = example['image']
                    elif 'images' in example and example['images']:
                        image = example['images'][0] if isinstance(example['images'], list) else example['images']
                        
                    # Handle output (bbox or other target)
                    # For GroundUI, output is bbox
                    output_text = ""
                    if 'bbox' in example and example['bbox'] is not None:
                        # Format bbox as string. If it's a list, stringify it.
                        bbox = example['bbox']
                        output_text = str(bbox)
                    
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": instruction}
                            ]
                        },
                        {
                            "role": "assistant",
                            "content": output_text
                        }
                    ]
                    return {"messages": messages}
                
                print("  Applying instruction->messages conversion (single process for Windows)...")
                dataset = dataset.map(convert_instruct_to_messages, num_proc=1)
                print(f"  ✓ Conversion complete. New columns: {dataset.column_names}")

            else:
                print(f"  Warning: No 'messages', 'conversations', or 'text' column. Available: {columns}")
                print(f"  Vision training will likely fail.")
        else:
            print("  Vision format: 'messages' column found - validating structure...")
            
            # Validate message structure
            try:
                sample = dataset[0]
                sample_msgs = sample.get('messages', [])
                
                if sample_msgs and isinstance(sample_msgs, list) and len(sample_msgs) > 0:
                    first_msg = sample_msgs[0]
                    
                    # Check structure
                    has_role = 'role' in first_msg
                    has_content = 'content' in first_msg
                    content = first_msg.get('content')
                    
                    # Check if content is multimodal (list) or text-only (string)
                    is_multimodal = isinstance(content, list)
                    has_image = is_multimodal and any(
                        item.get('type') == 'image' for item in content if isinstance(item, dict)
                    )
                    
                    print(f"    ✓ Structure: role={has_role}, content={has_content}")
                    print(f"    ✓ Format: {'multimodal' if is_multimodal else 'text-only'}")
                    print(f"    ✓ Has image data: {has_image}")
                    
                    if not has_image:
                        print("    ⚠️ Warning: No image found in first message content")
                        print("       This may cause issues with vision training")
                        
                    # Show sample for debugging
                    print(f"    Sample first message role: {first_msg.get('role')}")
                    if is_multimodal:
                        content_types = [item.get('type') for item in content if isinstance(item, dict)]
                        print(f"    Sample content types: {content_types}")
                else:
                    print("    ⚠️ Warning: messages column appears empty or malformed")
                    
            except Exception as e:
                print(f"    ⚠️ Warning: Could not validate messages structure: {e}")
        
        # DEBUG: Inspect and Patch Model Config for Unsloth Compatibility
        print(f"DEBUG: Model Config Architectures: {getattr(model.config, 'architectures', 'None')}")
        print(f"DEBUG: Model Config Type: {getattr(model.config, 'model_type', 'None')}")
        
        # VERIFICATION: Check VLM flags before Trainer Init
        print("\n" + "="*40)
        print("🔍 Pre-Trainer Vision Check")
        print("="*40)
        print(f"Model Class: {type(model).__name__}")
        print(f"model.is_vision: {getattr(model, 'is_vision', 'MISSING')}")
        print(f"config.is_vision_model: {getattr(model.config, 'is_vision_model', 'MISSING')}")
        print(f"Has vision_config: {hasattr(model.config, 'vision_config')}")
        print(f"Architecture: {model.config.architectures}")
        
        # Final safety net - force it again just in case
        if not getattr(model, 'is_vision', False):
             print("⚠️ WARN: model.is_vision was False/Missing. Forcing True.")
             model.is_vision = True
             
        print("="*40 + "\n")

    # ------------------------------------------------------------------------
    # EXECUTION WRAPPER (Prevents Zombie Processes)
    # ------------------------------------------------------------------------
    try:
        # ------------------------------------------------------------------------
        # LOADER: TEXT (LLM) - Uses Unsloth FastLanguageModel with PEFT fallback
        # ------------------------------------------------------------------------
        if model_category == "text":
            # ... (Text loading code remains same, simplified here for context but I will assume it's preserved if not replaced)
            # WAIT: replace_file_content replaces the BLOCK. I must be careful not to delete the text loader if I select a range that includes it.
            # The current selection range is 900+ (Vision/Audio trainers). 
            # I will only touch the Vision/Audio block and the generic trainer init part.
            pass # Placeholder logic - actual replacement below wraps the specific VLM logic
            
            # CRITICAL: On Windows, use single-process tokenization to prevent spawn issues
            trainer = SFTTrainer(
                model=model,
                processing_class=tokenizer,
                train_dataset=dataset,
                args=training_args,
                dataset_num_proc=1,  # FORCE single-process tokenization (prevents Windows spawn loop)
                dataset_batch_size=1000,  # Batch for efficiency despite single proc
                dataset_kwargs={"num_proc": 1}, # Extra safety: pass as kwargs in case init arg is ignored
            )

        # ------------------------------------------------------------------------
        # LOADER: VISION (VLM) - Universal Fix
        # ------------------------------------------------------------------------
        elif model_category == "vision":
             print("DEBUG: Applying Universal Unsloth Vision Workflow")
             
             # 1. Mode Switch (Essential for Qwen3-VL, safe for others)
             print("  Applying FastVisionModel.for_training(model)...")
             FastVisionModel.for_training(model)
             
             # 2. Data Format - Strict "messages" only
             # Removing 'image' column bypasses Unsloth's faulty "is_vision_dataset" check
             print("  Stripping dataset to 'messages' column only (Bypassing Trainer Vision Check)...")
             if "messages" not in dataset.column_names:
                  print("❌ Error: 'messages' column missing for Vision training!")
                  raise ValueError("Dataset missing 'messages' column")
             
             cols_to_remove = [c for c in dataset.column_names if c != "messages"]
             if cols_to_remove:
                  dataset = dataset.remove_columns(cols_to_remove)
             print(f"  ✓ Final columns: {dataset.column_names}")

             # 3. Trainer Init with Bypass
             print("Initializing SFTTrainer with UnslothVisionDataCollator & Bypass Config...")
             
             # Dummy formatter to appease SFTTrainer's check (required even with skip_prepare_dataset)
             # Unsloth validates this returns a list of strings, even if we skip using the result.
             def dummy_formatting_func(examples):
                 # Return list of empty strings matching batch size
                 return ["" for _ in range(len(examples["messages"]))]

             trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                data_collator=UnslothVisionDataCollator(model, tokenizer),
                train_dataset=dataset,
                args=training_args,
                max_seq_length=args.max_seq_length,
                dataset_text_field="", # Empty string + skip_prepare bypasses processing
                formatting_func=dummy_formatting_func, # Satisfies "must specify formatting_func" check
                dataset_num_proc=1,
                dataset_kwargs={
                    "num_proc": 1, 
                    "skip_prepare_dataset": True # The Magic Switch
                }, 
            )
            
        elif model_category == "audio":
            from transformers import Seq2SeqTrainer
            trainer = Seq2SeqTrainer(
                args=training_args,
                model=model,
                train_dataset=dataset,
                processing_class=processor.feature_extractor,
            )

        # ------------------------------------------------------------------------
        # EXECUTION
        # ------------------------------------------------------------------------
        print("\n" + "="*60)
        print("🚀 Starting Training...")
        print("="*60)
        
        # --- Checkpoint Resumption ---
        resume_checkpoint = None
        if os.path.exists(OUTPUT_DIR):
            checkpoints = [os.path.join(OUTPUT_DIR, d) for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
            if checkpoints:
                checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
                resume_checkpoint = checkpoints[-1]
                print(f"🔄 Found existing checkpoint: {resume_checkpoint}")

        trainer_stats = trainer.train(resume_from_checkpoint=resume_checkpoint)
        print("\n✓ Training completed successfully!")
        
        # Log peak memory usage
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / 1024**3
            print(f"   Peak VRAM used: {peak_mem:.2f} GB")
            
        # Save Model
        print("\nSaving Model...")
        final_path = os.path.join(OUTPUT_DIR, "final_model")
        model.save_pretrained(final_path)
        if tokenizer: tokenizer.save_pretrained(final_path)
        if processor: processor.save_pretrained(final_path)
        print(f"✓ Model saved to: {final_path}")

    except Exception as e:
        # CRITICAL: Catch initialization errors AND training errors
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        # Ensure we exit with error code so backend knows
        sys.exit(1)

if __name__ == "__main__":
    # CRITICAL: Windows multiprocessing support - must be first thing in main block
    import multiprocessing
    multiprocessing.freeze_support()
    
    # Set spawn method for Windows compatibility (prevents fork issues)
    if sys.platform == "win32":
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=False, help="Single dataset path (legacy)")
    parser.add_argument("--datasets", type=str, required=False, help="Comma-separated list of dataset paths")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--output_dir", type=str, default="data/outputs")
    parser.add_argument("--chat_template", type=str, default="none", help="Chat template to use (llama-3, chatml, zephyr, etc) if using Unsloth")
    parser.add_argument("--training_method", type=str, default="sft", choices=["sft", "dpo", "orpo"], help="Training method: sft (default), dpo, orpo")
    parser.add_argument("--adapter_type", type=str, default="lora", choices=["lora", "dora", "full"], help="Adapter type: lora (default), dora, full")
    parser.add_argument("--use_dora", action="store_true", help="Use DoRA instead of LoRA (legacy flag)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--optimizer", type=str, default="adamw_8bit", choices=["adamw", "adamw_8bit", "paged_adamw_8bit"])
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", choices=["linear", "cosine", "constant"])

    args = parser.parse_args()
    
    # Handle legacy --dataset arg and new --datasets arg
    if args.datasets:
        args.dataset_paths = [p.strip() for p in args.datasets.split(",")]
    elif args.dataset:
        args.dataset_paths = [args.dataset]
    else:
        print("Error: Must provide either --dataset or --datasets argument")
        sys.exit(1)
    
    # Ensure UTF-8 output
    if sys.platform == "win32":
        sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')
    
    main(args)