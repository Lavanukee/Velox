import sys
import subprocess
import platform
import os

def get_cuda_version():
    """
    Attempts to detect CUDA version via nvidia-smi.
    Returns '118' or '121' or '124' etc. (e.g. 12.1 -> 121)
    Defaulting to 121 if detection fails but GPU exists.
    """
    try:
        # Run nvidia-smi to get driver version / cuda version
        output = subprocess.check_output(["nvidia-smi"], encoding="utf-8")
        if "CUDA Version: 12." in output:
            return "121" # Prefer 12.4 for newer setups if supported, or 12.1
        elif "CUDA Version: 11." in output:
            return "118"
    except:
        pass
    
    # Fallback or default for 3090 Ti (Ampere) which supports 12.x
    return "121"

def install_torch():
    print("Detecting hardware for PyTorch installation...")
    
    system = platform.system()
    if system != "Windows":
        print("This script is designed for Windows.")
        return

    # Check for NVIDIA GPU
    try:
        subprocess.check_call(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        has_gpu = True
        print("NVIDIA GPU detected.")
    except:
        has_gpu = False
        print("No NVIDIA GPU detected via nvidia-smi.")

    if not has_gpu:
        print("Installing CPU-only PyTorch (Unsloth will NOT work)...")
        cmd = [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"]
    else:
        cuda_version = get_cuda_version()
        print(f"Targeting CUDA version: {cuda_version}")
        
        # IMPORTANT: Pin specific versions for Unsloth compatibility
        # torch 2.6.0 + torchvision 0.21.0 + torchaudio 2.6.0 are verified compatible
        # Using CUDA 12.4 as it has the best package availability for torch 2.6
        
        # Override detected version to 124 for torch 2.6 compatibility
        cuda_version = "124"  # Force cu124 for torch 2.6.x availability
        index_url = f"https://download.pytorch.org/whl/cu{cuda_version}"
        
        # Pin versions to avoid version mismatch between torch and torchvision
        # torch 2.6.0 is required for Unsloth-zoo's int1 dtype support
        torch_version = "2.6.0"
        torchvision_version = "0.21.0"
        torchaudio_version = "2.6.0"
        
        print(f"Installing PyTorch {torch_version} with CUDA {cuda_version} support...")
        print("(Pinned versions for Unsloth compatibility)")
        cmd = [
            sys.executable, "-m", "pip", "install", 
            "--upgrade", "--force-reinstall",
            f"torch=={torch_version}+cu{cuda_version}",
            f"torchvision=={torchvision_version}+cu{cuda_version}",
            f"torchaudio=={torchaudio_version}+cu{cuda_version}",
            "perceptron", # Required for Isaac models
            "--index-url", index_url
        ]

    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd)
        print("\nPyTorch installation complete.")
        
        # Verify
        print("\nVerifying installation...")
        verify_cmd = [sys.executable, "-c", "import torch; import perceptron; print(f'Torch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"]
        subprocess.check_call(verify_cmd)
        
    except subprocess.CalledProcessError as e:
        print(f"Installation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    install_torch()
