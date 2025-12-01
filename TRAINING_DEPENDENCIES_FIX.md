# Training Dependencies Fix - Critical Updates

## Problem Summary
After successful dependency installation, training failed with three critical errors:
1. **NumPy 2.x Incompatibility**: PyTorch was compiled with NumPy 1.x but NumPy 2.3.5 was installed
2. **Missing AutoProcessor**: `transformers` package didn't export `AutoProcessor` 
3. **torchao Version Warning**: Torch 2.4.0+cu121 vs torchao 0.14.1 mismatch

## Root Cause
The dependency installation was incomplete and used wrong versions:
- NumPy was installed without version constraint, getting latest (2.3.5)
- Transformers was outdated and missing AutoProcessor export
- huggingface_hub was not explicitly installed

## Solution Implemented

### Updated Package Installation Order (`src-tauri/src/lib.rs` lines 2474-2510)

```rust
let commands = vec![
    // 1. NumPy (MUST be <2 for PyTorch compatibility)
    vec!["-m", "pip", "install", "numpy<2"],
    
    // 2. Torch + Xformers (Combined for dependency resolution)
    vec![
        "-m", "pip", "install",
        "torch", "torchvision", "torchaudio", "xformers",
        "--index-url", "https://download.pytorch.org/whl/cu121",
    ],
    
    // 3. Transformers (updated version with AutoProcessor)
    vec![
        "-m", "pip", "install",
        "--upgrade", "transformers", "huggingface_hub",
    ],
    
    // 4. Bitsandbytes
    vec!["-m", "pip", "install", "bitsandbytes"],
    
    // 5. Triton (Windows) - May fail, non-critical
    vec!["-m", "pip", "install", "triton-windows"],
    
    // 6. Unsloth
    vec![
        "-m", "pip", "install",
        "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
    ],
    
    // 7. TRL and other deps
    vec![
        "-m", "pip", "install",
        "trl", "peft", "accelerate", "datasets", "tensorboard",
    ],
];
```

## Key Changes

### 1. NumPy Version Constraint ✅
**Before**: No constraint → installed NumPy 2.3.5  
**After**: `numpy<2` → installs latest 1.x (e.g., 1.26.4)

**Why**: PyTorch binaries are compiled against NumPy 1.x ABI. NumPy 2.x has breaking ABI changes that cause:
```
UserWarning: Failed to initialize NumPy: _ARRAY_API not found
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.3.5
```

### 2. Transformers Upgrade ✅
**Before**: Transformers installed as dependency (older version)  
**After**: `--upgrade transformers huggingface_hub`

**Why**: Older transformers versions don't export `AutoProcessor`:
```python
from transformers import AutoProcessor  # ImportError in old versions
```

Modern transformers (4.30+) export AutoProcessor for multimodal models.

### 3. Explicit huggingface_hub ✅
**Before**: Not explicitly installed  
**After**: Installed alongside transformers

**Why**: Required for model downloading, authentication, and repo file listing functionality already implemented in the app.

### 4. Installation Order Changed
**New Order**:
1. **NumPy first** - Establishes correct version before PyTorch
2. **PyTorch stack** - torch, torchvision, torchaudio, xformers
3. **Transformers** - upgraded to latest with AutoProcessor
4. **Training tools** - bitsandbytes, triton
5. **Unsloth** - efficient fine-tuning
6. **Training framework** - TRL, PEFT, Accelerate, datasets, TensorBoard

**Why**: NumPy must be correct version before anything that depends on it. Transformers upgrade ensures AutoProcessor is available.

## Testing Requirements

Users who already ran the old setup should:

### Option 1: Re-run Setup (Recommended)
1. Clear localStorage: Open DevTools → Application → Local Storage → Delete `pythonEnvSetup`
2. Refresh app
3. Click "Install Dependencies" again
4. Packages will reinstall with correct versions

### Option 2: Manual Fix (Advanced)
```bash
python -m pip uninstall numpy -y
python -m pip install "numpy<2"
python -m pip install --upgrade transformers huggingface_hub
```

## Verification

After re-running setup, check versions:
```python
import numpy as np
import transformers
print(f"NumPy: {np.__version__}")  # Should be 1.x
print(f"Transformers: {transformers.__version__}")  # Should be 4.40+

from transformers import AutoProcessor  # Should work
print("✅ AutoProcessor available")
```

## Impact on Training

With these fixes:
- ✅ NumPy compatibility resolved
- ✅ AutoProcessor imports successfully  
- ✅ Vision models can be loaded (via AutoProcessor)
- ✅ Training can start without import errors
- ⚠️ torchao warning persists but is non-critical (skipped C++ extensions)

## Future Improvements

1. **Version Pinning**: Consider pinning all package versions for reproducibility
2. **Pre-flight Check**: Add version validation before training starts
3. **Dependency Manifest**: Create requirements.txt for transparency
4. **Rollback Support**: Allow users to revert to previous environment state

## Related Files Modified
- `src-tauri/src/lib.rs` - setup_python_env_command (lines 2474-2510)

## Error Messages Resolved
✅ `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.3.5`  
✅ `ImportError: cannot import name 'AutoProcessor' from 'transformers'`  
⚠️ `Skipping import of cpp extensions due to incompatible torch version` (non-critical)
