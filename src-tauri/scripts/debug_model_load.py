import sys
import os
from transformers import AutoTokenizer, AutoConfig

def test_load(model_path):
    print(f"Testing load for: {model_path}")
    
    print("1. Loading Config...")
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        print(f"   Config type: {type(config)}")
        # print(f"   Config content: {config}") # content might be too large
        
        if isinstance(config, dict):
            print("   WARNING: Config is a dictionary!")
            if "model_type" in config:
                print(f"   model_type in dict: {config['model_type']}")
            else:
                print("   model_type NOT in dict")
        else:
            print("   Config is an object.")
            if hasattr(config, "model_type"):
                print(f"   model_type attribute: {config.model_type}")
            else:
                print("   model_type attribute MISSING")

    except Exception as e:
        print(f"   Failed to load config: {e}")
        return

    print("\n2. Loading Tokenizer...")
    try:
        # Try passing the config object directly if we have it
        if 'config' in locals():
            print("   Attempting to pass loaded config to tokenizer...")
            
            # Fix for dict config
            if isinstance(config, dict):
                print("   Converting dict config to PretrainedConfig...")
                from transformers import PretrainedConfig
                # We need to know the model type to create the right config, 
                # but PretrainedConfig.from_dict might work if model_type is in the dict
                if "model_type" in config:
                    # This is a bit hacky, usually AutoConfig should handle this
                    # Let's try to reload it forcing it to be an object if possible, 
                    # or just wrap it.
                    # Actually, let's try to use the dict to initialize a generic config
                    config_obj = PretrainedConfig.from_dict(config)
                    print(f"   Converted config type: {type(config_obj)}")
                    tokenizer = AutoTokenizer.from_pretrained(model_path, config=config_obj, trust_remote_code=True)
                else:
                    print("   Cannot convert dict config: missing 'model_type'")
                    tokenizer = AutoTokenizer.from_pretrained(model_path, config=config, trust_remote_code=True)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, config=config, trust_remote_code=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            
        print(f"   Tokenizer loaded successfully: {type(tokenizer)}")
    except Exception as e:
        print(f"   Failed to load tokenizer: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Hardcoded path based on the error log
    model_path = r"C:\Users\Jedd\Desktop\TuningPipeline\Velox\data\models\Qwen--Qwen3-0.6B"
    test_load(model_path)
