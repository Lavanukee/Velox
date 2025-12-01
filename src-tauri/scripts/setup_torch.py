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
        
        # Unsloth recommends Torch 2.4.0 or 2.5.0 + CUDA 12.1 or 12.4
        # We will force a reinstall of a compatible version
        
        # Construct pip install command
        # Using the stable index for cu124 or cu121
        index_url = f"https://download.pytorch.org/whl/cu{cuda_version}"
        
        print(f"Installing PyTorch with CUDA {cuda_version} support...")
        cmd = [
            sys.executable, "-m", "pip", "install", 
            "--upgrade", "--force-reinstall",
            "torch", "torchvision", "torchaudio",
            "--index-url", index_url
        ]

    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd)
        print("\nPyTorch installation complete.")
        
        # Verify
        print("\nVerifying installation...")
        verify_cmd = [sys.executable, "-c", "import torch; print(f'Torch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"]
        subprocess.check_call(verify_cmd)
        
    except subprocess.CalledProcessError as e:
        print(f"Installation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    install_torch()
