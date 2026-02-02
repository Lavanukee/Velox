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
    
    # Fallback
    return "121"

def install_torch():
    print("Detecting hardware for PyTorch installation...")
    
    system = platform.system()
    
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
        # Add cpu logic for linux if different index needed? usually fine via pypi for pure cpu
        if system == "Linux":
             cmd.append("--index-url")
             cmd.append("https://download.pytorch.org/whl/cpu")
    else:
        cuda_version = get_cuda_version()
        print(f"Targeting CUDA version: {cuda_version}")
        
        # IMPORTANT: Pin specific versions for Unsloth compatibility
        # torch 2.6.0 + torchvision 0.21.0 + torchaudio 2.6.0 are verified compatible
        
        # Override detected version to 124 for torch 2.6 compatibility if on Windows (or Linux generally)
        cuda_version = "124" 
        
        # Linux and Windows have slightly different index URLs sometimes, 
        # but for PyTorch WHL they are consistent: https://download.pytorch.org/whl/cu124
        index_url = f"https://download.pytorch.org/whl/cu{cuda_version}"
        
        # Pin versions to avoid version mismatch between torch and torchvision
        torch_version = "2.6.0"
        torchvision_version = "0.21.0"
        torchaudio_version = "2.6.0"
        
        if system == "Linux":
             # Linux specific version string? Usually just same.
             print(f"Installing PyTorch {torch_version} with CUDA {cuda_version} support (Linux)...")
        else:
             print(f"Installing PyTorch {torch_version} with CUDA {cuda_version} support (Windows)...")
             
        print("(Pinned versions for Unsloth compatibility)")
        cmd = [
            sys.executable, "-m", "pip", "install", 
            "--upgrade", "--force-reinstall",
            f"torch=={torch_version}", # +cu not always needed in package name if index-url is set correctly, but harmless
            f"torchvision=={torchvision_version}",
            f"torchaudio=={torchaudio_version}",
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
