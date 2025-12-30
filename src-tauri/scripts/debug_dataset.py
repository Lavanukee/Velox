
from datasets import load_from_disk
import sys
import os

dataset_path = r"C:\Users\Jedd\AppData\Roaming\com.lavanukee.Velox\data\datasets\agent-studio--GroundUI-1K\processed_data"
output_file = r"src-tauri/scripts/debug_result.log"

with open(output_file, "w", encoding="utf-8") as f:
    try:
        f.write(f"Loading dataset from {dataset_path}...\n")
        if not os.path.exists(dataset_path):
             f.write("Dataset path does not exist.\n")
             sys.exit(1)
             
        ds = load_from_disk(dataset_path)
        f.write("Dataset loaded successfully.\n")
        f.write(f"Features: {ds.features}\n")
        
        if len(ds) == 0:
            f.write("Dataset is empty.\n")
        else:
            example = ds[0]
            f.write("\n--- Example 0 ---\n")
            f.write(f"Keys: {list(example.keys())}\n")
            
            if 'conversations' in example:
                conv = example['conversations']
                f.write(f"Type of 'conversations': {type(conv)}\n")
                f.write(f"Content of 'conversations': {conv}\n")
                
                if isinstance(conv, list):
                    f.write("conversations IS a list.\n")
                    if len(conv) > 0:
                        f.write(f"Type of conversations[0]: {type(conv[0])}\n")
                    else:
                        f.write("conversations list is empty\n")
                else:
                    f.write("conversations is NOT a list (THIS IS THE ISSUE if found).\n")
            
            elif 'messages' in example:
                 f.write("Found 'messages' column instead of conversations.\n")
            else:
                 f.write("No 'conversations' or 'messages' column found.\n")

            if 'text' in example:
                f.write(f"\n'text' column snippet: {str(example['text'])[:100]}...\n")

    except Exception as e:
        f.write(f"Error inspecting dataset: {e}\n")
