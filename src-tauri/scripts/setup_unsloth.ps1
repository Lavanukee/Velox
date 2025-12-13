# Check if running as administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")
if (-not $isAdmin) {
    Write-Host "Please run as Administrator to ensure all packages install correctly." -ForegroundColor Yellow
}

Write-Host "Setting up Unsloth Environment for Windows..." -ForegroundColor Cyan

# Check for Python
$python_cmd = "python"
if (Get-Command "python3" -ErrorAction SilentlyContinue) {
    $python_cmd = "python3"
}

Write-Host "Using Python: $python_cmd"

# 1. Install/Update Pytorch for Cuda 12.1 (recommended for Unsloth)
Write-Host "Installing PyTorch 2.4.0 with CUDA 12.1..."
& $python_cmd -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. Install xformers and trl
Write-Host "Installing xformers and trl..."
& $python_cmd -m pip install --upgrade xformers trl peft accelerate bitsandbytes tensorboard

# 3. Install Unsloth
# For Windows, we often need to install from specific wheels or git
Write-Host "Installing Unsloth..."
& $python_cmd -m pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# 4. Install other dependencies
Write-Host "Installing other dependencies..."
& $python_cmd -m pip install --no-deps "trl<0.9.0" peft accelerate bitsandbytes

Write-Host "Environment Setup Complete!" -ForegroundColor Green
Write-Host "You can now run training with Unsloth optimization."
