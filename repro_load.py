import os
import sys
from transformers import AutoConfig, AutoModelForCausalLM

model_path = r"C:\Users\Jedd\AppData\Roaming\com.lavanukee.Velox\data\models\PerceptronAI--Isaac-0.2-2B-Preview"

print(f"Testing model load from: {model_path}")
print(f"Exists: {os.path.exists(model_path)}")
print(f"Files in dir: {os.listdir(model_path)}")

try:
    print("Attempting AutoConfig.from_pretrained...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    print("Config loaded successfully.")
    print(f"Config class: {config.__class__}")
except Exception as e:
    print(f"Config load failed: {e}")
    import traceback
    traceback.print_exc()

print("-" * 20)

try:
    print("Attempting AutoModelForCausalLM.from_pretrained...")
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Model load failed: {e}")
