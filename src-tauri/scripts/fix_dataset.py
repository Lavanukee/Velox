import os
import json
import argparse
from pathlib import Path
from PIL import Image
import shutil

def validate_image(image_path, root_dir):
    """
    Checks if image exists and is valid.
    Returns (is_valid, abs_path, error_msg)
    """
    # handle cases where path starts with images/ already
    clean_path = image_path
    if clean_path.startswith("images/") or clean_path.startswith("images\\"):
        clean_path = clean_path[7:] # strip images/
        
    candidates = [
        os.path.join(root_dir, image_path),
        os.path.join(root_dir, clean_path),
        os.path.join(root_dir, "images", clean_path),
        os.path.abspath(image_path)
    ]
    
    final_path = None
    for cand in candidates:
        if os.path.exists(cand):
            final_path = cand
            break
            
    if not final_path:
        return False, None, f"Image not found: {image_path}"
        
    try:
        with Image.open(final_path) as img:
            img.verify() # check file integrity
        return True, final_path, None
    except Exception as e:
        return False, final_path, f"Corrupt image: {e}"

def fix_dataset(dataset_dir, output_file=None):
    """
    Scans dataset.jsonl in dataset_dir.
    Fixes:
    - Missing/Broken images (removes entry)
    - Missing conversations (converts instruction->conversations)
    - JSON syntax errors (skips line)
    """
    dataset_path = os.path.join(dataset_dir, "dataset.jsonl")
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} does not exist.")
        return

    print(f"Scanning {dataset_path}...")
    
    fixed_data = []
    stats = {"total": 0, "kept": 0, "fixed_structure": 0, "removed_image": 0, "removed_json": 0}
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            stats["total"] += 1
            line = line.strip()
            if not line: continue
            
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                print(f"Line {line_num}: Invalid JSON. Skipping.")
                stats["removed_json"] += 1
                continue
                
            # 1. Structure Fixes
            if "conversations" not in entry:
                if "instruction" in entry and "output" in entry:
                    # Convert Alpaca to ShareGPT
                    entry["conversations"] = [
                        {"role": "user", "content": entry["instruction"] + ("\n" + entry["input"] if entry.get("input") else "")},
                        {"role": "assistant", "content": entry["output"]}
                    ]
                    stats["fixed_structure"] += 1
                else:
                    print(f"Line {line_num}: Missing conversations/instruction. Skipping.")
                    stats["removed_json"] += 1
                    continue

            # 2. Image Validation
            if "image" in entry:
                is_valid, abs_path, err = validate_image(entry["image"], dataset_dir)
                if not is_valid:
                    print(f"Line {line_num}: {err}. Skipping.")
                    stats["removed_image"] += 1
                    continue
                # Optional: normalize path in entry?
                # entry["image"] = os.path.relpath(abs_path, dataset_dir) 

            fixed_data.append(entry)
            stats["kept"] += 1

    # Save
    if not output_file:
        output_file = os.path.join(dataset_dir, "dataset_fixed.jsonl")
        
    print(f"Saving {len(fixed_data)} entries to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in fixed_data:
            f.write(json.dumps(entry) + "\n")
            
    print("\nSummary:")
    print(f"  Total Lines: {stats['total']}")
    print(f"  Kept: {stats['kept']}")
    print(f"  Auto-Fixed Structure: {stats['fixed_structure']}")
    print(f"  Removed (Image Errors): {stats['removed_image']}")
    print(f"  Removed (JSON Errors): {stats['removed_json']}")
    
    return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True, help="Directory containing dataset.jsonl")
    parser.add_argument("--output", help="Output JSONL file path (default: dataset_fixed.jsonl)")
    args = parser.parse_args()
    
    fix_dataset(args.dataset_dir, args.output)
