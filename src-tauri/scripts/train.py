import os
import sys
import torch
import json
import argparse
import logging
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

# ============================================================================
# 2. IMPORTS
# ============================================================================
try:
    import unsloth
    from unsloth import FastLanguageModel, FastVisionModel
except ImportError as e:
    print(f"Warning: Unsloth not installed or failed to import: {e}")
    print("Will use standard HuggingFace + PEFT instead")
    unsloth = None
    FastLanguageModel = None
    FastVisionModel = None

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
from trl import SFTTrainer
from datasets import Dataset, load_from_disk
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

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
            model_type = config.get("model_type", "")
            archs = config.get("architectures", [])
        else:
            archs = getattr(config, "architectures", [])
            model_type = getattr(config, "model_type", "")
        
        str_archs = str(archs).lower()
        
        # Audio Detection
        if "whisper" in model_type or "speech" in str_archs or "audio" in str_archs:
            return "audio"
            
        # Vision Detection (VLMs)
        if "vision" in str_archs or "vl" in str_archs or "visual" in str_archs or "image" in model_type:
            return "vision"
        
        # Default to Text
        return "text"
    except Exception as e:
        print(f"Warning: Could not auto-detect model type ({e}). Defaulting to 'text'.")
        return "text"

def resolve_dataset_path(dataset_arg: str) -> str:
    """
    Robustly resolves dataset paths to find the processed_data/train directory.
    
    Checks in order:
    1. Exact path as given
    2. data/datasets/{dataset_arg}/processed_data/train
    3. data/datasets/{dataset_arg}/processed_data
    4. {dataset_arg}/processed_data/train
    5. {dataset_arg}/processed_data
    """
    # Clean up the path
    dataset_arg = dataset_arg.replace("src-tauri", "").replace("scripts", "").replace("\\", "/").strip("/")
    
    # Get the project root (parent of src-tauri/scripts)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    
    # Extract just the dataset name if it's a path
    dataset_name = os.path.basename(dataset_arg.rstrip("/"))
    
    # List of candidate paths to check
    candidates = [
        dataset_arg,  # As-is
        os.path.join(dataset_arg, "train"),  # Add train subdirectory
        os.path.join(project_root, "data", "datasets", dataset_name, "processed_data", "train"),
        os.path.join(project_root, "data", "datasets", dataset_name, "processed_data"),
        os.path.join(project_root, "data", "datasets", dataset_arg, "processed_data", "train"),
        os.path.join(project_root, "data", "datasets", dataset_arg, "processed_data"),
        os.path.join(dataset_arg, "processed_data", "train"),
        os.path.join(dataset_arg, "processed_data"),
    ]
    
    print(f"\nResolving dataset path: '{dataset_arg}'")
    print(f"Project root: {project_root}")
    print(f"Dataset name: {dataset_name}")
    
    for candidate in candidates:
        abs_candidate = os.path.abspath(candidate)
        print(f"  Checking: {abs_candidate}")
        
        # Check if this path exists and has dataset_info.json
        dataset_info_path = os.path.join(abs_candidate, "dataset_info.json")
        if os.path.exists(dataset_info_path):
            print(f"  ✓ Found dataset_info.json at: {abs_candidate}")
            return abs_candidate
        
        # Also check if it's a valid HuggingFace dataset directory (has data files)
        if os.path.isdir(abs_candidate):
            # Check for common dataset files
            has_data = any(
                os.path.exists(os.path.join(abs_candidate, f))
                for f in ["data-00000-of-00001.arrow", "dataset.arrow", "state.json"]
            )
            if has_data:
                print(f"  ✓ Found dataset files at: {abs_candidate}")
                return abs_candidate
    
    # If nothing found, return the original and let it fail with a clear error
    print(f"  ❌ Could not find dataset in any expected location")
    print(f"  Please ensure the dataset is at: {os.path.join(project_root, 'data', 'datasets', dataset_name, 'processed_data', 'train')}")
    return dataset_arg

# ============================================================================
# 4. MAIN TRAINING LOGIC
# ============================================================================

def main(args):
    # --- Dataset Path Resolution ---
    PROCESSED_TRAIN_DIR = resolve_dataset_path(args.dataset)
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
                print("✓ Vision model loaded successfully!")
                use_unsloth = True
                
            except Exception as e:
                print(f"Failed to load Vision model with Unsloth: {e}")
                print("Vision models require Unsloth. Please check your installation.")
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
    # DATA LOADING & FORMATTING
    # ------------------------------------------------------------------------
    print(f"\nLoading dataset from {PROCESSED_TRAIN_DIR}...")
    try:
        dataset = Dataset.load_from_disk(PROCESSED_TRAIN_DIR)
        print(f"✓ Dataset loaded: {len(dataset)} examples")
        print(f"  Columns: {dataset.column_names}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        sys.exit(1)
    
    # --- Formatter for TEXT Models ---
    if model_category == "text":
        if "text" not in dataset.column_names:
            print("Formatting dataset for Text Model...")
            def format_text(examples):
                if "instruction" in examples and "output" in examples:
                    inst = examples["instruction"]
                    inp = examples.get("input", "")
                    out = examples["output"]
                    text = f"Instruction: {inst}\nInput: {inp}\nOutput: {out}" if inp else f"Instruction: {inst}\nOutput: {out}"
                    return {"text": text}
                return examples
            
            dataset = dataset.map(format_text, batched=False)
            print(f"✓ Dataset formatted. Sample: {dataset[0]['text'][:100]}...")
            
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
        report_to="none",
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
    )

    # ------------------------------------------------------------------------
    # TRAINER INITIALIZATION
    # ------------------------------------------------------------------------
    print("Initializing Trainer...")
    
    if model_category == "vision":
        from unsloth import UnslothVisionDataCollator
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            data_collator=UnslothVisionDataCollator(model, tokenizer),
            train_dataset=dataset,
            args=training_args,
        )
        
    elif model_category == "audio":
        from transformers import Seq2SeqTrainer
        trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            train_dataset=dataset,
            processing_class=processor.feature_extractor,
        )
        
    else:  # TEXT
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            args=training_args,
        )

    # ------------------------------------------------------------------------
    # EXECUTION
    # ------------------------------------------------------------------------
    print("\n" + "="*60)
    print("🚀 Starting Training...")
    print("="*60)
    
    try:
        trainer_stats = trainer.train()
        print("\n✓ Training completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\nSaving Model...")
    final_path = os.path.join(OUTPUT_DIR, "final_model")
    
    try:
        model.save_pretrained(final_path)
        if tokenizer:
            tokenizer.save_pretrained(final_path)
        if processor:
            processor.save_pretrained(final_path)
        print(f"✓ Model saved to: {final_path}")
        
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--output_dir", type=str, default="data/outputs")
    
    args = parser.parse_args()
    
    # Ensure UTF-8 output
    if sys.platform == "win32":
        sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')
    
    main(args)