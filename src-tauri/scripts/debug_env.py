import sys
import os

print(f"Python Executable: {sys.executable}", flush=True)

print("--- Importing torch ---", flush=True)
try:
    import torch
    print(f"Torch Version: {torch.__version__}", flush=True)
except Exception as e:
    print(f"Torch Import Failed: {e}", flush=True)

print("--- Importing transformers ---", flush=True)
try:
    import transformers
    print(f"Transformers Version: {transformers.__version__}", flush=True)
    print(f"Transformers File: {transformers.__file__}", flush=True)
except Exception as e:
    print(f"Transformers Import Failed: {e}", flush=True)

print("--- Importing AutoProcessor ---", flush=True)
try:
    from transformers import AutoProcessor
    print("AutoProcessor imported successfully", flush=True)
except Exception as e:
    print(f"AutoProcessor Import Failed: {e}", flush=True)

print("--- Importing unsloth ---", flush=True)
try:
    import unsloth
    print(f"Unsloth Version: {unsloth.__version__}", flush=True)
except Exception as e:
    print(f"Unsloth Import Failed: {e}", flush=True)
