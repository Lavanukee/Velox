import json
from pathlib import Path
from typing import Dict, Any
import argparse

def get_unique_key(entry: Dict[str, Any]) -> str:
    """Generates a unique key based on image path, description, raw coordinates, and screen size."""
    # This key generation needs to be robust. Assuming 'image' field is present or we get 'abc'.
    image_path = entry.get("image", "abc") # Use the image path from the entry itself.
    
    # Metadata fields
    metadata = entry.get("metadata", {})
    description = metadata.get("description", "")
    
    # Raw coordinates (x, y)
    coords_raw = metadata.get("coordinates_raw", [0, 0])
    raw_x = int(coords_raw[0]) if len(coords_raw) >= 1 else 0
    raw_y = int(coords_raw[1]) if len(coords_raw) >= 2 else 0
    
    # Screen size (width, height)
    screen_size = metadata.get("screen_size", [0, 0])
    screen_w = int(screen_size[0]) if len(screen_size) >= 1 else 0
    screen_h = int(screen_size[1]) if len(screen_size) >= 2 else 0
    
    # Combine all fields into a single, strict key
    key_parts = [
        image_path,
        description,
        f"raw:{raw_x},{raw_y}",
        f"screen:{screen_w},{screen_h}"
    ]
    
    return "|".join(key_parts)

def deduplicate_dataset(dataset_path: Path, backup_path: Path):
    """Reads the dataset, removes duplicates, and overwrites the original file."""
    if not dataset_path.exists():
        print(f"Error: Dataset file not found at {dataset_path}")
        return

    print(f"Reading dataset from {dataset_path}...")
    
    unique_entries = {}
    total_entries = 0
    
    # 1. Read all entries and store unique ones
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            total_entries += 1
            try:
                entry = json.loads(line.strip())
                key = get_unique_key(entry)
                
                # Keep the first occurrence (or the latest, depending on how you iterate. 
                # Since we read line by line, we keep the first one encountered.)
                if key not in unique_entries:
                    unique_entries[key] = entry
                
            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed JSON line: {line.strip()[:50]}...")
                
    unique_count = len(unique_entries)
    duplicate_count = total_entries - unique_count
    
    print(f"Total entries read: {total_entries}")
    print(f"Unique entries found: {unique_count}")
    print(f"Duplicates removed: {duplicate_count}")
    
    if unique_count == total_entries:
        print("No duplicates found. Exiting.")
        return

    # 2. Backup the original file
    print(f"Backing up original file to {backup_path}...")
    backup_path.parent.mkdir(parents=True, exist_ok=True) # Ensure backup directory exists
    dataset_path.rename(backup_path)
    
    # 3. Write unique entries back to the original path
    print(f"Writing {unique_count} unique entries back to {dataset_path}...")
    with open(dataset_path, 'w', encoding='utf-8') as f:
        for entry in unique_entries.values():
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    print("Deduplication complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deduplicate a dataset.jsonl file based on image path, description, raw coordinates, and screen size."
    )
    parser.add_argument(
        "--dataset_path",
        type=Path,
        default="./dataset/dataset.jsonl",
        help="Path to the dataset.jsonl file to deduplicate."
    )
    parser.add_argument(
        "--backup_path",
        type=Path,
        default="./dataset/dataset_deduplicated.jsonl.bak", # Changed default name
        help="Path to backup the original dataset file before overwriting."
    )
    
    args = parser.parse_args()
    deduplicate_dataset(args.dataset_path, args.backup_path)
