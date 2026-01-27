"""
Unsloth Dynamic GGUF 2.0 Converter
Uses Unsloth's native save_pretrained_gguf for fast conversion.
"""
import os
import sys
import argparse
import torch
import gc
import subprocess
import json
from huggingface_hub import hf_hub_download

# Force UTF-8 encoding for stdout/stderr on Windows to prevent encoding errors with emojis
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

# Suppress verbose warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1" # Bypass SSL errors on Windows
os.environ["UNSLOTH_LOCAL_ONLY"] = "1"

# ============================================================================
# AUTO-REPAIR LOGIC
# ============================================================================

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
                import importlib
                importlib.invalidate_caches()
            except Exception as e:
                print(f"❌ Failed to auto-install '{package}': {e}")

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
                
                # Infer Repo ID heuristic
                dir_name = os.path.basename(model_path)
                repo_id = dir_name.replace("--", "/")
                
                print(f"   Inferring Repo ID: {repo_id}")
                try:
                    hf_hub_download(repo_id=repo_id, filename=filename, local_dir=model_path)
                    print("   ✓ Download successful.")
                except Exception as dl_err:
                     print(f"   First attempt failed ({dl_err}).")
                     pass

    except Exception as e:
        print(f"Warning: Failed to check/recover model files: {e}")

def monkeypatch_unsloth():
    """
    Monkeypatch Unsloth to skip telemetry/statistics checks that fail on Windows
    due to missing Unix-specific features (SIGALRM/setitimer).
    """
    try:
        import unsloth.models._utils as utils
        def dummy_stats(*args, **kwargs):
            return None
        utils.get_statistics = dummy_stats
        print("🔧 Monkeypatched Unsloth get_statistics (Windows Safety)")
    except Exception as e:
        print(f"⚠️ Failed to monkeypatch unsloth: {e}")

def main():
    parser = argparse.ArgumentParser(description="Convert HuggingFace models to GGUF using Unsloth")
    parser.add_argument("--model", type=str, required=True, help="Path to HuggingFace model or model ID")
    parser.add_argument("--output", type=str, required=True, help="Output GGUF file path")
    parser.add_argument("--quant", type=str, default="q8_0", 
                       choices=["q4_k_m", "q5_k_m", "q8_0", "f16", "bf16", "not_quantized", "auto"],
                       help="Quantization method")
    parser.add_argument("--lora", type=str, default=None, help="Optional LoRA adapter path to merge")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 Unsloth Dynamic GGUF 2.0 Converter")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print(f"Quantization: {args.quant}")
    if args.lora:
        print(f"LoRA: {args.lora}")
    
    # Ensure 'perceptron' is installed
    check_and_install_packages(["perceptron"])

    # Dynamic Hardware Detection
    vram_gb = 0
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"📊 GPU Detected: {torch.cuda.get_device_name(0)} ({vram_gb:.1f} GB VRAM)")
    else:
        print("⚠️ No CUDA GPU detected!")
    
    # Hardware-based choices
    load_in_4bit = True
    if vram_gb >= 20:
        print("💎 High VRAM (3090 Ti level): Forcing float16 loading for clean merge.")
        load_in_4bit = False
    else:
        print("📉 Moderate VRAM: Loading in 4-bit to save memory.")
        load_in_4bit = True

    try:
        monkeypatch_unsloth()
        from unsloth import FastLanguageModel
    except ImportError as e:
        print(f"❌ Error: Unsloth not installed. {e}")
        sys.exit(1)
    
    print("\nPROGRESS: 10% - Loading model...")
    try:
        # Auto-repair model files if needed
        ensure_model_files(args.model)
        
        # Determine strict or permissive mode
        try:
             model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=args.model,
                max_seq_length=4096,
                dtype=torch.float16 if vram_gb >= 12 else torch.float32,
                load_in_4bit=load_in_4bit,
                local_files_only=True
             )
        except Exception as unsloth_err:
             print(f"Unsloth loading failed: {unsloth_err}")
             print("Attempting fallback to direct llama.cpp conversion (convert_hf_to_gguf.py)...")
             
             # Fallback logic
             script_path = os.path.join(os.path.dirname(__file__), "convert_hf_to_gguf.py")
             
             # Map quantization types
             out_type = args.quant
             if out_type == "auto":
                 out_type = "q8_0" if vram_gb >= 16 else "q4_k_m"
             elif out_type == "not_quantized":
                 out_type = "f16"

             cmd = [
                 sys.executable, script_path, args.model,
                 "--outfile", args.output,
                 "--outtype", out_type
             ]
             print(f"Executing Fallback: {' '.join(cmd)}")
             subprocess.run(cmd, check=True)
             print("\n✅ Fallback conversion complete.")
             sys.exit(0)

        print(f"✓ Model loaded successfully (4-bit={load_in_4bit})")
        
        # Merge LoRA if provided
        if args.lora and os.path.exists(args.lora):
            print("\nPROGRESS: 40% - Merging LoRA adapter...")
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.lora)
            model = model.merge_and_unload()
            print("✓ LoRA merged successfully")
        
        # Ensure output directory exists
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print("\nPROGRESS: 60% - Saving GGUF (this may take a few minutes)...")
        
        if sys.platform == "win32":
            print("\n💾 Windows detected: Using memory-optimized merge-and-convert flow...")
            
            import tempfile
            import subprocess
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Remove bitsandbytes quantization config for clean GGUF export
                if hasattr(model, 'config') and hasattr(model.config, 'quantization_config'):
                    model.config.quantization_config = None
                
                # GC and cache clear before casting
                gc.collect()
                torch.cuda.empty_cache()
                
                try:
                    print("🔧 Casting model to float16 for export...")
                    model = model.to(torch.float16)
                    gc.collect()
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"⚠️ Casting warning: {e}")
                
                # Save the model and tokenizer with SMALLER shards to avoid MemoryError on Windows
                print(f"📦 Saving HF shards (1GB chunks) to: {temp_dir}")
                model.save_pretrained(temp_dir, max_shard_size="1GB", safe_serialization=True)
                tokenizer.save_pretrained(temp_dir)
                
                # Determine target quantization based on VRAM
                # "if the full model fits ALWAYS Q8_0, if it doesn't fit at q8, then we drop to q4 k m"
                out_type = args.quant
                if out_type in ["f16", "bf16", "not_quantized", "auto"]:
                    if vram_gb >= 20:
                        print("💎 VRAM fits Q8_0: Exporting high quality 8-bit GGUF.")
                        out_type = "q8_0"
                    else:
                        print("📉 Limited VRAM: Exporting auto/Q4 GGUF.")
                        out_type = "auto"
                
                # Final check for hf-to-gguf compatibility
                if out_type not in ["f32", "f16", "bf16", "q8_0", "auto"]:
                    print(f"⚠️  '{out_type}' not natively supported, falling back to q8_0 or auto based on VRAM.")
                    out_type = "q8_0" if vram_gb >= 16 else "auto"

                print(f"\nPROGRESS: 80% - Converting HF to GGUF {out_type}...")
                script_path = os.path.join(os.path.dirname(__file__), "convert_hf_to_gguf.py")
                
                try:
                    cmd = [
                        sys.executable, script_path, temp_dir,
                        "--outfile", args.output,
                        "--outtype", out_type
                    ]
                    print(f"Executing: {' '.join(cmd)}")
                    subprocess.run(cmd, check=True)
                    print("\n✓ GGUF conversion complete")
                except subprocess.CalledProcessError as e:
                    print(f"\n❌ HF to GGUF conversion failed: {e}")
                    raise
        else:
            # Native Unsloth GGUF saving for Linux
            model.save_pretrained_gguf(
                args.output,
                tokenizer,
                quantization_method=args.quant,
            )
        
        # Success verification
        found_output = None
        possible_outputs = [args.output, f"{args.output}.gguf", f"{args.output}-q8_0.gguf"]
        for p in possible_outputs:
            if os.path.exists(p):
                found_output = p
                break
        
        if found_output:
            print("\nPROGRESS: 100% - Success!")
            print(f"✅ Saved: {found_output}")
        else:
            print("\n⚠️ Conversion finished but output file not found in expected locations.")
        
    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
