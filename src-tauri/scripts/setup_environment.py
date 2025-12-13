"""
Complete environment setup script for Velox training pipeline.
Follows strict installation order to avoid dependency conflicts.
"""
import sys
import subprocess
import os

def run_pip(args, description):
    """Run pip command and handle errors."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    cmd = [sys.executable, "-m", "pip"] + args
    print(f"Running: {' '.join(cmd)}\n")
    try:
        subprocess.check_call(cmd)
        print(f"✓ {description} completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with exit code {e.returncode}")
        return False

def run_python_script(script_path, description):
    """Run a Python script."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    cmd = [sys.executable, script_path]
    print(f"Running: {' '.join(cmd)}\n")
    try:
        subprocess.check_call(cmd)
        print(f"✓ {description} completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with exit code {e.returncode}")
        return False

def main():
    print("\n" + "="*60)
    print("Velox Training Environment Setup")
    print("="*60)
    
    # Step 1: Install base packages
    base_packages = [
        "huggingface_hub",
        "requests",
        "mistral_common",
        "sentencepiece",
        "datasets",
        "accelerate",
        "peft",
        "trl",
        "bitsandbytes",
        "packaging",
    ]
    
    if not run_pip(["install"] + base_packages, "Installing base packages"):
        print("\n⚠️ Base packages installation failed. Continuing anyway...")

    # Step 2: Install Triton separately with version constraint
    # IMPORTANT: triton-windows 3.2.x is required for PyTorch 2.6 compatibility
    # version 3.3+ has breaking changes with AttrsDescriptor
    if not run_pip(["install", "-U", "triton-windows<3.3"], "Installing Triton"):
        print("\n⚠️ Triton installation failed. Continuing anyway...")
    
    # Step 3: Run setup_torch.py to install PyTorch with CUDA
    script_dir = os.path.dirname(os.path.abspath(__file__))
    torch_setup_script = os.path.join(script_dir, "setup_torch.py")
    
    if not run_python_script(torch_setup_script, "Setting up PyTorch with CUDA support"):
        print("\n❌ PyTorch setup failed. Cannot continue.")
        return False
    
    # Step 4: Install Unsloth
    unsloth_packages = [
        "unsloth[windows] @ git+https://github.com/unslothai/unsloth.git",
        "unsloth-zoo"
    ]
    
    if not run_pip(["install"] + unsloth_packages, "Installing Unsloth"):
        print("\n⚠️ Unsloth installation failed. Continuing anyway...")
    
    # Step 5: CRITICAL - Reinstall PyTorch with CUDA (Unsloth may pull in CPU version)
    # This ensures we have the GPU-accelerated version
    print("\n⚠️ Reinstalling PyTorch with CUDA to ensure Unsloth didn't override it...")
    if not run_python_script(torch_setup_script, "Reinstalling PyTorch CUDA (Unsloth may have overwritten)"):
        print("\n⚠️ PyTorch CUDA reinstall failed. You may have CPU-only PyTorch.")
    
    # Step 6: Install specific transformers version and gguf
    # Note: xformers is installed by Unsloth with correct version
    final_packages = [
        "transformers>=4.45.0",  # Let Unsloth manage specific version
        "git+https://github.com/ggml-org/llama.cpp.git#subdirectory=gguf-py",
    ]
    
    if not run_pip(["install"] + final_packages, "Installing transformers and gguf"):
        print("\n⚠️ Final packages installation failed.")
    
    # Verification
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    
    verify_script = """
import torch
import transformers

print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

print(f"Transformers: {transformers.__version__}")

try:
    import unsloth
    print("Unsloth: ✓ Installed")
except ImportError as e:
    print(f"Unsloth: ✗ Failed - {e}")

try:
    import gguf
    print("GGUF: ✓ Installed")
except ImportError as e:
    print(f"GGUF: ✗ Failed - {e}")
"""
    
    print("\nRunning verification...")
    subprocess.run([sys.executable, "-c", verify_script])
    
    print("\n" + "="*60)
    print("Setup Complete!")
    print("="*60)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
