# Configuration
$LlamaDir = "C:\AI\llama.cpp"
$RepoUrl = "https://github.com/ggml-org/llama.cpp.git"

# --- GPU SETTINGS ---
$UseCUDA = $true

# --------------------

# 1. Check if we need to Clone
if (-not (Test-Path $LlamaDir)) {
    Write-Host "Cloning llama.cpp..." -ForegroundColor Cyan
    git clone $RepoUrl $LlamaDir
}
Set-Location $LlamaDir

# 2. Pull latest changes
Write-Host "Pulling latest code..." -ForegroundColor Cyan
git pull origin master

# 3. Check for NVCC (CUDA Compiler)
if ($UseCUDA) {
    try {
        nvcc --version | Out-Null
        Write-Host "CUDA compiler found." -ForegroundColor Green
    } catch {
        Write-Host "Error: NVCC not found." -ForegroundColor Red
        exit 1
    }
}

# 4. Configure with CMake
Write-Host "Configuring Build..." -ForegroundColor Cyan

$CmakeArgs = @("-B", "build", "-G", "Visual Studio 17 2022", "-A", "x64")
$CmakeArgs += "-DLLAMA_CURL=OFF"

if ($UseCUDA) {
    $CmakeArgs += "-DGGML_CUDA=ON"
    $CmakeArgs += "-DCMAKE_CUDA_ARCHITECTURES=86" 
}

# Run configuration
cmake @CmakeArgs

# 5. Compile (Release mode)
Write-Host "Compiling with Visual Studio..." -ForegroundColor Cyan
cmake --build build --config Release -j $Env:NUMBER_OF_PROCESSORS --target llama-server llama-cli

# 6. Move Binaries AND DLLs (Crucial Fix)
Write-Host "Moving binaries and DLLs..." -ForegroundColor Cyan
Copy-Item -Path "build\bin\Release\*.exe" -Destination "." -Force
Copy-Item -Path "build\bin\Release\*.dll" -Destination "." -Force

Write-Host "Update Complete!" -ForegroundColor Green
Get-Item ".\llama-server.exe"