import torch
try:
    import unsloth
    print("Unsloth imported successfully.")
except ImportError as e:
    print(f"Failed to import unsloth: {e}")

try:
    from unsloth import FastLanguageModel
    print("FastLanguageModel imported successfully.")
except ImportError as e:
    print(f"Failed to import FastLanguageModel: {e}")

print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
