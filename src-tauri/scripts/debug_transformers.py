import sys
import os

print(f"Python Executable: {sys.executable}", flush=True)

try:
    import transformers
    print(f"Transformers Version: {transformers.__version__}", flush=True)
    print(f"Transformers File: {transformers.__file__}", flush=True)
    print(f"Transformers Dir: {os.path.dirname(transformers.__file__)}", flush=True)
except ImportError as e:
    print(f"Transformers Import Failed: {e}", flush=True)
    sys.exit(1)

print("--- Checking AutoTokenizer ---", flush=True)
try:
    from transformers import AutoTokenizer
    print("AutoTokenizer imported successfully", flush=True)
except ImportError as e:
    print(f"AutoTokenizer Import Failed: {e}", flush=True)

print("--- Checking AutoProcessor ---", flush=True)
try:
    from transformers import AutoProcessor
    print("AutoProcessor imported successfully", flush=True)
except ImportError as e:
    print(f"AutoProcessor Import Failed: {e}", flush=True)
    
    # Check if it's available in submodule
    print("Checking transformers.models.auto...", flush=True)
    try:
        from transformers.models.auto import AutoProcessor
        print("AutoProcessor found in transformers.models.auto", flush=True)
    except ImportError as e2:
        print(f"AutoProcessor NOT found in transformers.models.auto: {e2}", flush=True)

    # List attributes of transformers
    print("Transformers attributes starting with Auto:", flush=True)
    print([x for x in dir(transformers) if x.startswith('Auto')], flush=True)
