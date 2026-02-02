import os
import sys

# ============================================================================
# ============================================================================
# CRITICAL: Windows Multiprocessing Fix - MUST BE AT VERY TOP
# ============================================================================
if os.name == 'nt':
    # Disable tokenizers parallelism to prevent fork bombs on Windows
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Force single-process for datasets mapping on Windows
    os.environ["HF_DATASETS_DISABLE_CACHING"] = "1"
    
    # Prevent memory fragmentation - Critical for Windows/limited VRAM
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
    
    # DISBALE MULTIPROCESSING GLOBALLY FOR DATASETS ON WINDOWS
    datasets.config.MAX_CORES = 1
    datasets.config.IN_MEMORY_MAX_SIZE = 0 # Force disk-based if memory is tight
else:
    # Linux/Mac Setup
    # Enable parallelism for better performance
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    # Allow caching
    if "HF_DATASETS_DISABLE_CACHING" in os.environ:
        del os.environ["HF_DATASETS_DISABLE_CACHING"]

datasets.disable_progress_bar()
datasets.logging.set_verbosity_error() # Use error level to reduce sprawl logs

# ============================================================================
# CRITICAL: Windows Multiprocessing Fix - Monkeypatch Dataset.map
# ============================================================================
if os.name == 'nt':
    try:
        # Save original method to avoid recursion if patched multiple times
        if not hasattr(datasets.Dataset, '_original_map'):
            datasets.Dataset._original_map = datasets.Dataset.map
            
        def windows_safe_map(self, *args, **kwargs):
            # Force single process to prevent spawn crashes
            # valid_args = self.map.__code__.co_varnames
            
            # Helper to update kwargs
            if 'num_proc' in kwargs:
                if kwargs['num_proc'] is not None and kwargs['num_proc'] > 1:
                    pass # silent override or log if debug
                kwargs['num_proc'] = 1
            
            return datasets.Dataset._original_map(self, *args, **kwargs)
            
        datasets.Dataset.map = windows_safe_map
        print("✓ Applied Windows safety patch for datasets.map")
    except Exception as e:
        print(f"Warning: Failed to patch datasets.map: {e}")

# Suppress Unsloth/HF spam
import logging
logging.getLogger("unsloth").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)

from typing import Optional, Dict, Any
from pathlib import Path

# ============================================================================
# 1. WINDOWS COMPATIBILITY PATCHES
# ============================================================================

def create_deepspeed_config(output_dir: str, offload_optimizer: bool = False):
    """
    Creates a ZeRO-2 DeepSpeed config for memory conservation.
    ZeRO-2 is best for single-GPU hybrid setups.
    """
    config = {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {
                "device": "cpu" if offload_optimizer else "none",
                "pin_memory": True
            },
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True
        },
        "gradient_clipping": "auto",
        "steps_per_print": 2000,
        "wall_clock_breakdown": False
    }
    
    config_path = os.path.join(output_dir, "ds_config.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    return config_path
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
# 2. IMPORTS AND AUTO-REPAIR
# ============================================================================

import subprocess

def check_and_install_packages(packages):
    """Checks if packages are installed and installs them if missing."""
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            print(f"⚠️ Missing dependency '{package}'. Auto-installing...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✓ Successfully installed '{package}'.")
                # Invalidate cache to allow import
                import importlib
                importlib.invalidate_caches()
            except Exception as e:
                print(f"❌ Failed to auto-install '{package}': {e}")
                print(f"   Please install it manually: pip install {package}")

# CRITICAL: Ensure 'perceptron' is installed for Isaac models
# Also ensure xformers is installed for memory efficient attention on Windows
check_and_install_packages(["perceptron", "xformers"])

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
from huggingface_hub import hf_hub_download

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

# ============================================================================
# 3. HELPER FUNCTIONS
# ============================================================================

def ensure_model_files(model_path: str):
    """
    Checks if the model directory is missing required custom code files 
    referenced in config.json (e.g., modular_isaac.py) and attempts to download them.
    """
    if not os.path.exists(model_path) or not os.path.isdir(model_path):
        return

    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        return

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # Check auto_map for custom files
        auto_map = config.get("auto_map", {})
        required_files = set()
        
        for key, value in auto_map.items():
            # Value can be "filename.class"
            if isinstance(value, str):
                filename = value.split(".")[0] + ".py"
                required_files.add(filename)

        # Attempt recovery
        for filename in required_files:
            file_path = os.path.join(model_path, filename)
            if not os.path.exists(file_path):
                print(f"⚠️ Missing required custom file '{filename}'. Attempting to download...")
                
                # Infer Repo ID: usually the parent directory name if using standard structure
                # Or assume path ends with "Author--ModelName" or similar
                # Just loop through plausible repo IDs or use a heuristic
                # Simple heuristic: Use the directory name if it looks like Author--Model
                dir_name = os.path.basename(model_path)
                repo_id = dir_name.replace("--", "/")
                
                print(f"   Inferring Repo ID: {repo_id}")
                try:
                    hf_hub_download(repo_id=repo_id, filename=filename, local_dir=model_path)
                    print("   ✓ Download successful.")
                except Exception as dl_err:
                     # Try simpler name (Isaac-0.2-2B-Preview) if -- format check failed
                     print(f"   First attempt failed ({dl_err}). Trying fallback IDs...")
                     pass

    except Exception as e:
        print(f"Warning: Failed to check/recover model files: {e}")

def get_bnb_config():
    """Returns a standardized 4-bit config with CPU offload enabled."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )

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
            abs_path = os.path.abspath(candidate)
            # Auto-repair custom files if needed
            ensure_model_files(abs_path)
            return abs_path
    
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

def dummy_formatting_func(examples):
    """
    Dummy formatting function for Vision models.
    Required by SFTTrainer but UnslothVisionDataCollator handles the actual formatting.
    Must be global for Windows multiprocessing pickling.
    """
    # Return list of empty strings safely
    count = len(examples["messages"]) if isinstance(examples["messages"], list) else 1
    return ["" for _ in range(count)]

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
                    quantization_config=get_bnb_config(),
                    trust_remote_code=True,
                    device_map="auto" if args.use_cpu_offload else None,
                    offload_folder=os.path.join(OUTPUT_DIR, "offload"),
                    llm_int8_enable_fp32_cpu_offload=True,
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
                # ROBUST MEASURE: Reserve VRAM
                total_ram_gb = psutil.virtual_memory().total / (1024**3)
                safe_ram_gb = int(total_ram_gb - min(8, total_ram_gb * 0.1))
                max_mem = {0: "18GiB", "cpu": f"{safe_ram_gb}GiB"}
                
                if torch.cuda.is_available():
                    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    reserved_vram = 6 if total_vram > 12 else 2
                    max_mem[0] = f"{int(total_vram - reserved_vram)}GiB"

                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    quantization_config=get_bnb_config(),
                    device_map="auto",
                    max_memory=max_mem,
                    trust_remote_code=True,
                    offload_folder=os.path.join(OUTPUT_DIR, "offload"),
                    llm_int8_enable_fp32_cpu_offload=True,
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
            print("\n" + "="*60)
            print("🖼️  VISION MODEL LOADER (VLM)")
            print("="*60)
            
            # =========================================================
            # PRE-LOAD: Estimate model size to determine strategy
            # =========================================================
            try:
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                
                # Get layer count
                num_layers = getattr(config, "num_hidden_layers", 32)
                if hasattr(config, "text_config"): 
                    num_layers = getattr(config.text_config, "num_hidden_layers", num_layers)
                
                # Get hidden size for estimation
                hidden_size = getattr(config, "hidden_size", 4096)
                if hasattr(config, "text_config"):
                    hidden_size = getattr(config.text_config, "hidden_size", hidden_size)
                
                # =====================================================
                # ROBUST MoE DETECTION: Check multiple sources
                # =====================================================
                num_experts = None
                is_moe = False
                
                # Method 1: Check config attributes
                for attr_name in ["num_local_experts", "num_experts", "num_moe_experts"]:
                    if num_experts is None:
                        num_experts = getattr(config, attr_name, None)
                    if num_experts is None and hasattr(config, "text_config"):
                        num_experts = getattr(config.text_config, attr_name, None)
                
                # Method 2: Check architectures list for "MoE" or "Moe"
                arch_list = getattr(config, "architectures", [])
                if any("moe" in str(a).lower() for a in arch_list):
                    is_moe = True
                    print(f"📊 Detected MoE from architecture: {arch_list}")
                
                # Method 3: Check model path for MoE patterns
                model_path_lower = model_path.lower()
                if "moe" in model_path_lower or "-a3b" in model_path_lower or "-a2b" in model_path_lower:
                    is_moe = True
                    print(f"📊 Detected MoE from model path pattern")
                
                # Method 4: Extract size from model path (e.g., "30B-A3B" -> 30)
                import re
                size_match = re.search(r'(\d+\.?\d*)b', model_path_lower)
                path_size_b = float(size_match.group(1)) if size_match else None
                
                if num_experts and num_experts > 1:
                    is_moe = True
                
                # =====================================================
                # CALCULATE MODEL SIZE
                # =====================================================
                if is_moe:
                    # For MoE models, use path-extracted size if available (more reliable)
                    if path_size_b and path_size_b > 5:  # Sanity check
                        params_b = path_size_b
                        print(f"📊 MoE model size from path: {params_b}B")
                    else:
                        # Fallback calculation
                        intermediate = getattr(config, "moe_intermediate_size", 
                                     getattr(config, "intermediate_size", hidden_size * 4))
                        if hasattr(config, "text_config"):
                            intermediate = getattr(config.text_config, "moe_intermediate_size", intermediate)
                        experts = num_experts if num_experts else 8  # Default for most MoE
                        params_b = (num_layers * (4 * hidden_size**2 + experts * 3 * hidden_size * intermediate)) / 1e9
                        print(f"📊 MoE model size calculated: {params_b:.1f}B (experts={experts})")
                else:
                    # Standard model
                    params_b = (num_layers * 12 * hidden_size**2) / 1e9
                
                print(f"📊 Final estimated model size: ~{params_b:.1f}B parameters")
                print(f"📊 Layers: {num_layers}, Hidden: {hidden_size}, MoE: {is_moe}")
                
            except Exception as e:
                print(f"⚠️ Could not estimate model size: {e}")
                # Try to extract from path as last resort
                import re
                size_match = re.search(r'(\d+\.?\d*)b', model_path.lower())
                params_b = float(size_match.group(1)) if size_match else 7
                print(f"📊 Fallback size from path: {params_b}B")
            
            # =========================================================
            # CRITICAL: Check if model is too large for Unsloth 4-bit
            # Qwen3-VL-MoE models don't properly support 4-bit in Unsloth!
            # =========================================================
            if params_b > 20:
                print(f"\n" + "="*60)
                print(f"⚠️ LARGE MODEL DETECTED: {params_b}B parameters")
                print(f"   Unsloth may not properly quantize MoE vision models.")
                print(f"   Will verify after loading and abort if needed.")
                print(f"="*60)
            
            # =========================================================
            # DECIDE STRATEGY: Auto-select based on model size vs VRAM
            # =========================================================
            if torch.cuda.is_available():
                total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
                torch.cuda.empty_cache()
                free_vram = torch.cuda.mem_get_info()[0] / 1024**3
            else:
                total_vram = 0
                free_vram = 0
            
            total_ram_gb = psutil.virtual_memory().total / (1024**3)
            
            # 4-bit model size = params_b * 0.5 GB (rough)
            # Add 50% overhead for activations, LoRA, optimizer states
            estimated_4bit_vram = params_b * 0.5 * 1.5
            
            print(f"\n📊 VRAM Analysis:")
            print(f"   Available: {free_vram:.1f} / {total_vram:.1f} GB")
            print(f"   Estimated 4-bit requirement: ~{estimated_4bit_vram:.1f} GB")
            
            # Force CPU offload for large models that won't fit
            if args.vram_strategy == "auto":
                if estimated_4bit_vram > free_vram * 0.9:
                    forced_strategy = "cpu"
                    print(f"   ⚠️ Model too large for GPU-only, forcing CPU offload strategy")
                else:
                    forced_strategy = "gpu"
                    print(f"   ✓ Model should fit in VRAM")
            else:
                forced_strategy = args.vram_strategy
            
            # =========================================================
            # LOAD MODEL: Use Unsloth's native 4-bit parameter
            # =========================================================
            def attempt_load_model(strategy):
                print(f"\n{'='*60}")
                print(f"⚡ LOADING IN 4-BIT QUANTIZED MODE (NF4)")
                print(f"   Strategy: {strategy.upper()}")
                print(f"{'='*60}")
                
                # Calculate memory limits
                safe_ram_gb = int(total_ram_gb - min(8, total_ram_gb * 0.1))
                
                # =====================================================
                # STRATEGY SELECTION
                # 30B Model @ 4-bit is ~16GB. Fits on 24GB GPU.
                # We should prioritize keeping it fully on GPU to avoid
                # 4-bit CPU offload issues.
                # =====================================================
                dev_map = "auto"
                
                if params_b > 25 and total_vram < 20: 
                     # Only force offload if we strictly don't have space
                     # (e.g. 30B model on 16GB GPU)
                     print(f"   ⚠️ Model ({params_b}B) larger than VRAM ({total_vram}GB). Enabling CPU offload hooks.")
                
                # Use high utilization to fit model
                gpu_limit = int(total_vram - 2) # Leave 2GB buffer
                max_mem = {0: f"{gpu_limit}GiB"}
                
                # If using CPU strategy, allow more offload but prefer GPU
                if strategy == "cpu":
                     max_mem["cpu"] = f"{safe_ram_gb}GiB"
                
                print(f"   device_map: {dev_map}")
                print(f"   max_memory: {max_mem}")
                print(f"{'='*60}\n")
                
                # =====================================================
                # CREATE EXPLICIT 4-BIT CONFIG with CPU offload enabled
                # =====================================================
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    llm_int8_enable_fp32_cpu_offload=True,  # CRITICAL for CPU offload
                )
                
                print("📦 BitsAndBytesConfig:")
                print(f"   load_in_4bit: True")
                print(f"   llm_int8_enable_fp32_cpu_offload: True")
                
                # =====================================================
                # ATTEMPT: Unsloth with custom device map
                # =====================================================
                model = None
                tokenizer = None
                
                try:
                    print("\n🔄 Attempt 1: Unsloth with explicit quantization_config...")
                    model, tokenizer = FastVisionModel.from_pretrained(
                        model_name=model_path,
                        load_in_4bit=True,
                        quantization_config=bnb_config,
                        use_gradient_checkpointing="unsloth",
                        trust_remote_code=True,
                        device_map=dev_map,
                        max_memory=max_mem,
                        offload_folder=os.path.join(OUTPUT_DIR, "offload"),
                        low_cpu_mem_usage=True,
                    )
                except Exception as e:
                    print(f"   ⚠️ Unsloth load failed: {e}")
                    error_str = str(e).lower()
                    
                    # Special retry for TypeError regarding quantization_config
                    if "quantization_config" in error_str and isinstance(e, TypeError):
                        try:
                            print(f"   ⚠️ quantization_config not accepted, trying load_in_4bit only...")
                            model, tokenizer = FastVisionModel.from_pretrained(
                                model_name=model_path,
                                load_in_4bit=True,
                                use_gradient_checkpointing="unsloth",
                                trust_remote_code=True,
                                device_map="auto",
                                max_memory=max_mem,
                                offload_folder=os.path.join(OUTPUT_DIR, "offload"),
                                low_cpu_mem_usage=True,
                            )
                        except Exception as e2:
                            print(f"   ⚠️ Retry failed: {e2}")
                            model = None
                    else:
                        model = None
                
                # =====================================================
                # VERIFY & FALLBACK
                # =====================================================
                print("\n🔍 Verifying quantization...")
                should_fallback = False
                
                if model is None:
                    print("   ⚠️ Model failed to load in Attempt 1.")
                    should_fallback = True
                elif torch.cuda.is_available():
                    torch.cuda.synchronize()
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    print(f"   Post-load VRAM: {allocated:.1f} GB allocated")
                    expected_4bit = params_b * 0.5
                    
                    # If memory usage indicates FP16 loading (too high), force fallback
                    if allocated > expected_4bit * 2 and allocated > 20:
                        print(f"\n   ⚠️ Memory too high ({allocated:.1f} GB > expected {expected_4bit:.1f} GB)")
                        print(f"   🔄 Unsloth quantization appears to have failed!")
                        should_fallback = True
                
                if should_fallback:
                    print(f"   Attempting fallback to standard HuggingFace + BitsAndBytes...")
                    
                    # Clean up failed model
                    if model is not None: del model
                    if tokenizer is not None: del tokenizer
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    # =====================================================
                    # ATTEMPT 2: Standard HuggingFace AutoModelForVision2Seq
                    # with BitsAndBytes - may work better for MoE
                    # =====================================================
                    print("\n🔄 Attempt 2: HuggingFace AutoModel with BitsAndBytes...")
                    from transformers import AutoProcessor, AutoModelForVision2Seq, Qwen2VLForConditionalGeneration
                    
                    # Try to use the appropriate model class
                    try:
                        model = AutoModelForVision2Seq.from_pretrained(
                            model_path,
                            quantization_config=bnb_config,
                            device_map="auto",
                            max_memory=max_mem,
                            trust_remote_code=True,
                            low_cpu_mem_usage=True,
                            offload_folder=os.path.join(OUTPUT_DIR, "offload"),
                        )
                        tokenizer = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
                    except Exception as hf_error:
                        print(f"   ❌ HuggingFace fallback failed: {hf_error}")
                        raise RuntimeError(
                            f"Both Unsloth and HuggingFace failed to load model in 4-bit. "
                            f"This model ({params_b}B) may not support 4-bit quantization. "
                            f"Please use a smaller model (7B-8B) instead."
                        )
                    
                    # Check if HuggingFace version worked
                    torch.cuda.synchronize()
                    allocated_hf = torch.cuda.memory_allocated() / 1024**3
                    print(f"   HuggingFace post-load VRAM: {allocated_hf:.1f} GB")
                    
                    if allocated_hf > expected_4bit * 2:
                        raise RuntimeError(
                            f"Quantization failed with both Unsloth and HuggingFace. "
                            f"Model using {allocated_hf:.1f}GB (expected ~{expected_4bit:.1f}GB). "
                            f"Qwen3-VL-MoE architecture may not support 4-bit. Use a 7B model."
                        )
                    
                    print(f"   ✓ HuggingFace 4-bit loading successful!")
                    # Note: We'll need to apply LoRA differently for HF models
                    return model, tokenizer

                
                return model, tokenizer

            # --- EXECUTE LOADING ---
            try:
                # Try forced strategy first
                strategies_to_try = [forced_strategy]
                if forced_strategy == "gpu":
                    strategies_to_try.append("cpu")  # Fallback to CPU offload
                
                for i, strategy in enumerate(strategies_to_try):
                    try:
                        # Clear cache before attempt
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                            import gc
                            gc.collect()
                        
                        model, tokenizer = attempt_load_model(strategy)
                        use_unsloth = True
                        print(f"\n✓ Model loaded successfully with {strategy.upper()} strategy!")
                        break
                    except Exception as e:
                        error_str = str(e).lower()
                        print(f"\n❌ Failed to load with {strategy.upper()}: {e}")
                        
                        if i < len(strategies_to_try) - 1:
                            print("   Retrying with fallback strategy...")
                            import gc
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        else:
                            raise e

            except Exception as final_error:
                print(f"❌ All load attempts failed. Last error: {final_error}")
                sys.exit(1)

            # ===================================================================
            # POST-LOAD: Apply LoRA adapters
            # ===================================================================
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
            
            # FORCE VISION FLAGS
            print("DEBUG: Setting critical vision flags...")
            model.is_vision = True
            if hasattr(model, "config"):
                model.config.is_vision_model = True
                
            # Ensure vision_config exists
            if not hasattr(model.config, "vision_config"):
                model.config.vision_config = {"exists": True}

            # PATCH: Fix missing chat template for vision models
            if hasattr(tokenizer, "chat_template") and (tokenizer.chat_template is None or tokenizer.chat_template == ""):
                print("⚠️ Vision processor/tokenizer is missing a chat_template. Applying robust Qwen2-VL fallback.")
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
            
            if not hasattr(model, "is_vision"):
                model.is_vision = True

            print("✓ Vision model loaded and PEFT adapters applied!")
            use_unsloth = True
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
                device_map="auto",
                offload_folder=os.path.join(OUTPUT_DIR, "offload"),
                llm_int8_enable_fp32_cpu_offload=True,
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
    # DATA LOADING & FORMATTING (Multiple Datasets Support with Weighting)
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
            print(f"\nCombining {len(datasets_list)} datasets...")
            
            # Equal weighting (default behavior)
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
    # EVALUATION SPLIT
    # ------------------------------------------------------------------------
    eval_dataset = None
    if args.eval_split and args.eval_split > 0:
        print(f"\nSplitting dataset for evaluation ({args.eval_split*100}%)...")
        # Split logic
        try:
            ds_split = dataset.train_test_split(test_size=args.eval_split, seed=42)
            dataset = ds_split["train"]
            eval_dataset = ds_split["test"]
            print(f"  Train samples: {len(dataset)}")
            print(f"  Eval samples:  {len(eval_dataset)}")
        except Exception as e:
            print(f"⚠️ Failed to split dataset: {e}. Training without eval.")
    else:
        print("\nSkipping evaluation split (using all data for training).")
            
    # ------------------------------------------------------------------------
    # TRAINING ARGUMENTS
    # ------------------------------------------------------------------------
    print("\nConfiguring training arguments...")
    
    # Optimizer selection
    optim_type = "adamw_8bit"
    if args.offload_optimizer:
        optim_type = "adamw_8bit" # Fallback if DS not used, DS handles offload itself
    elif args.use_paged_optimizer:
        optim_type = "paged_adamw_8bit"
    elif args.optimizer:
        optim_type = args.optimizer
        
    print(f"Using optimizer: {optim_type}")

    ds_config = None
    if args.use_deepspeed:
        print("Generating DeepSpeed ZeRO-2 Configuration...")
        ds_config = create_deepspeed_config(args.output_dir, args.offload_optimizer)
    
    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size // 2),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim=optim_type,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        seed=3407,
        output_dir=OUTPUT_DIR,
        logging_dir=OUTPUT_DIR,
        report_to="tensorboard",
        deepspeed=ds_config,
        
        # Eval settings
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=args.eval_steps if eval_dataset else None,
        
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        dataloader_num_workers=0, # Force single process for dataloader on Windows
        remove_unused_columns=False if model_category == "vision" else True, # Vision collator needs raw image/conversations
        
        # Gradient Checkpointing - Force ON for MoE models or if VRAM is tight
        gradient_checkpointing=True if (args.use_gradient_checkpointing or "moe" in model_path.lower()) else args.use_gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False}, # Always false for Unsloth compatibility
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
        
        def finalize_vision_dataset(ds):
            """Ensures dataset is strictly 'messages' only for Unsloth."""
            columns = ds.column_names
            if 'messages' not in columns:
                print("❌ Error: Dataset formatting failed to create 'messages' column.")
                return ds
            
            # Keep ONLY 'messages' column to prevent collator confusion
            to_remove = [c for c in columns if c != 'messages']
            if to_remove:
                print(f"  Cleaning dataset columns: removing {to_remove}")
                ds = ds.remove_columns(to_remove)
            
            # Filter out empty entries which cause IndexError/KeyError in collators
            print("  Filtering out empty or invalid conversations...")
            ds = ds.filter(lambda x: isinstance(x.get('messages'), list) and len(x['messages']) > 0, num_proc=1)
            
            # Final validation of a sample
            if len(ds) == 0:
                print("❌ Fatal: Dataset is empty after filtering!")
                return ds

            try:
                sample = ds[0]['messages']
                print(f"  ✓ Validated sample: {len(sample)} messages in first entry.")
            except Exception as e:
                print(f"⚠️ Validation error: {e}")
                
            return ds

        if 'messages' not in columns:
            # Fallback: try to convert other formats to 'messages'
            if 'conversations' in columns:
                print("  Converting 'conversations' -> 'messages'...")
                dataset = dataset.map(lambda x: {"messages": x['conversations']}, num_proc=1)
                dataset = finalize_vision_dataset(dataset)
            elif 'text' in columns:
                print("  Converting 'text' -> 'messages' format...")
                dataset = dataset.map(convert_text_to_messages, num_proc=1)
                dataset = finalize_vision_dataset(dataset)
            elif 'instruction' in columns:
                print("  Converting 'instruction' -> 'messages' format...")
                dataset = dataset.map(convert_instruct_to_messages, num_proc=1)
                dataset = finalize_vision_dataset(dataset)
            else:
                print(f"  Warning: No recognizable columns. Found: {columns}")
        else:
            print("  ✓ 'messages' column found. Ensuring strict schema & validating structure...")
            dataset = finalize_vision_dataset(dataset)
            
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
                eval_dataset=eval_dataset, # Pass eval dataset
                args=training_args,
                dataset_num_proc=1,  # FORCE single-process tokenization (prevents Windows spawn loop)
                dataset_batch_size=1000,  # Batch for efficiency despite single proc
                dataset_kwargs={"num_proc": 1}, # Extra safety: pass as kwargs in case init arg is ignored
            )

        # ------------------------------------------------------------------------
        # LOADER: VISION (VLM) - Universal Fix
        # ------------------------------------------------------------------------
        elif model_category == "vision":
             print("DEBUG: Finalizing Unsloth Vision setup...")
             
             # 1. Mode Switch (Essential for Qwen3-VL)
             FastVisionModel.for_training(model)
             
             # 2. Strict Dataset Validation (Last chance)
             if "messages" not in dataset.column_names:
                  raise ValueError("❌ Fatal: Dataset passed to Vision Trainer missing 'messages' column!")
             
             # Double check columns
             if len(dataset.column_names) > 1:
                  dataset = dataset.remove_columns([c for c in dataset.column_names if c != "messages"])
                  
             print(f"  ✓ Trainer Dataset Columns: {dataset.column_names}")

             # 3. Trainer Init
             print("Initializing SFTTrainer with UnslothVisionDataCollator...")
             
             # 3. Trainer Init
             print("Initializing SFTTrainer with UnslothVisionDataCollator...")


             trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                data_collator=UnslothVisionDataCollator(model, tokenizer),
                train_dataset=dataset,
                eval_dataset=eval_dataset,
                args=training_args,
                max_seq_length=args.max_seq_length,
                dataset_text_field="messages", # Explicitly set to messages
                formatting_func=dummy_formatting_func,
                dataset_num_proc=1,
                dataset_kwargs={
                    "num_proc": 1, 
                    "skip_prepare_dataset": True # Bypasses SFTTrainer's messy vision checks
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
        
        # --- Checkpoint Resumption with LoRA Rank Compatibility Check ---
        resume_checkpoint = None
        if os.path.exists(OUTPUT_DIR):
            checkpoints = [os.path.join(OUTPUT_DIR, d) for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
            if checkpoints:
                checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
                latest_checkpoint = checkpoints[-1]
                print(f"🔄 Found existing checkpoint: {latest_checkpoint}")
                
                # Check LoRA rank compatibility to prevent dimension mismatch errors
                adapter_config_path = os.path.join(latest_checkpoint, "adapter_config.json")
                if os.path.exists(adapter_config_path):
                    try:
                        with open(adapter_config_path, 'r') as f:
                            checkpoint_config = json.load(f)
                        checkpoint_r = checkpoint_config.get("r", None)
                        
                        if checkpoint_r is not None and checkpoint_r != args.lora_r:
                            print(f"⚠️ WARNING: Checkpoint LoRA rank mismatch!")
                            print(f"   Checkpoint rank: {checkpoint_r}")
                            print(f"   Current config rank: {args.lora_r}")
                            print(f"   Skipping checkpoint resume to avoid dimension errors.")
                            print(f"   Training will start from scratch with rank {args.lora_r}.")
                            resume_checkpoint = None
                        else:
                            print(f"✓ LoRA rank compatible (r={args.lora_r}). Resuming from checkpoint.")
                            resume_checkpoint = latest_checkpoint
                    except Exception as e:
                        print(f"⚠️ Could not read checkpoint config: {e}")
                        print(f"   Proceeding with caution, attempting resume...")
                        resume_checkpoint = latest_checkpoint
                else:
                    # No adapter config found, might be full fine-tune checkpoint
                    print(f"   No adapter_config.json found. Assuming compatible.")
                    resume_checkpoint = latest_checkpoint

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
    parser.add_argument("--weights", type=str, required=False, help="Comma-separated list of weights (0-1) for each dataset")
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
    
    # New Arguments for Hardware/Eval
    parser.add_argument("--eval_split", type=float, default=0.0, help="Fraction of data to use for evaluation (0-1)")
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluate every N steps")
    parser.add_argument("--use_cpu_offload", action="store_true", help="Offload model to CPU to save VRAM")
    parser.add_argument("--use_paged_optimizer", action="store_true", help="Use paged optimizer to save VRAM")
    parser.add_argument("--use_gradient_checkpointing", action="store_true", help="Use gradient checkpointing")
    
    # Hybrid Training Arguments
    parser.add_argument("--hybrid_training", action="store_true", help="Enable hybrid GPU+CPU training")
    parser.add_argument("--gpu_layers", type=int, default=None, help="Number of layers to keep on GPU")
    parser.add_argument("--offload_optimizer", action="store_true", help="Offload optimizer states to CPU RAM")
    parser.add_argument("--use_deepspeed", action="store_true", help="Use DeepSpeed ZeRO-2 for advanced offloading")
    parser.add_argument("--vram_strategy", type=str, default="auto", choices=["auto", "gpu", "cpu"], help="Memory strategy: auto, gpu, cpu")

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