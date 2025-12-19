import argparse
import json
import sys
import os
import requests
import shutil
from typing import Optional, List, Dict, Any
from huggingface_hub import HfApi, configure_http_backend, hf_hub_download

# Configure encoding for Windows consoles
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

def get_auth_headers(token: Optional[str]):
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers

def cmd_search(args):
    """Search for models or datasets."""
    api = HfApi(token=args.token)
    
    try:
        if args.type == "model":
            results = api.list_models(
                search=args.query,
                limit=args.limit,
                sort="downloads",
                direction=-1,
                full=False
            )
        else:
            results = api.list_datasets(
                search=args.query,
                limit=args.limit,
                sort="downloads",
                direction=-1,
                full=False
            )

        output = []
        for r in results:
            output.append({
                "id": r.id,
                "name": r.id.split('/')[-1],
                "downloads": getattr(r, "downloads", 0),
                "likes": getattr(r, "likes", 0),
                "author": r.author,
                "tags": getattr(r, "tags", [])
            })
        
        print(json.dumps(output))
        
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

def detect_quantization(filename: str) -> str:
    """Detects quantization type from filename."""
    name_lower = filename.lower()

    if "fp16" in name_lower or "f16" in name_lower:
        return "FP16"
    elif "bf16" in name_lower:
        return "BF16"
    elif "fp32" in name_lower or "f32" in name_lower:
        return "FP32"
    elif "q8" in name_lower:
        return "Q8"
    elif "q6" in name_lower:
        return "Q6"
    elif "q5" in name_lower:
        return "Q5"
    elif "q4" in name_lower:
        return "Q4"
    elif "q3" in name_lower:
        return "Q3"
    elif "q2" in name_lower:
        return "Q2"
    else:
        return "other"

def cmd_list_files(args):
    """List files in a repository with enhanced metadata."""
    api = HfApi(token=args.token)
    try:
        repo_info = api.repo_info(repo_id=args.repo_id, repo_type=args.type)
        siblings = repo_info.siblings
        
        files_with_metadata: List[Dict[str, Any]] = []
        
        # Define relevant extensions for datasets
        dataset_exts = (".parquet", ".json", ".jsonl", ".csv", ".arrow", ".txt", ".zip")

        for s in siblings:
            if s.rfilename.startswith(".git"):
                continue

            file_path = s.rfilename
            file_name_lower = file_path.lower()
            
            file_type = "other"
            quantization = None
            is_mmproj = False

            if args.type == "dataset":
                # For datasets, accept data files
                if file_name_lower.endswith(dataset_exts):
                    file_type = "dataset_file"
                elif file_name_lower == "readme.md":
                    file_type = "info"
            else:
                # Logic for Models
                if file_name_lower.endswith(".gguf"):
                    file_type = "gguf"
                    quantization = detect_quantization(file_path)
                    if "mmproj" in file_name_lower:
                        is_mmproj = True
                elif file_name_lower.endswith((".bin", ".safetensors")):
                    file_type = "weight"
                elif file_name_lower.endswith((".json", ".model", ".txt")):
                     # Include config files for models
                     if "config" in file_name_lower or "tokenizer" in file_name_lower or "vocab" in file_name_lower:
                         file_type = "config"
            
            files_with_metadata.append({
                "path": file_path,
                "size": getattr(s, "size", None),
                "lfs": getattr(s, "lfs", None),
                "file_type": file_type,
                "quantization": quantization,
                "is_mmproj": is_mmproj,
            })
            
        print(json.dumps(files_with_metadata))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

def cmd_download(args):
    """Download specific files from a repository using hf_hub_download."""
    base_folder = os.path.abspath(args.output)
    
    files_to_download = args.files.split(',') if args.files else []
    repo_type = "dataset" if args.type == "dataset" else "model"
    
    # If no specific files are requested, download the entire snapshot
    if not files_to_download:
        print(f"No specific files provided. Downloading full snapshot to: {base_folder}", file=sys.stderr)
        sys.stderr.flush()
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=args.repo_id,
                repo_type=repo_type,
                token=args.token,
                local_dir=base_folder,
                local_dir_use_symlinks=False
            )
            print("PROGRESS:100", file=sys.stderr)
            print("Download complete.", file=sys.stderr)
            return
        except Exception as e:
            print(f"Error downloading snapshot: {e}", file=sys.stderr)
            sys.exit(1)

    print(f"Starting download to: {base_folder}", file=sys.stderr)
    sys.stderr.flush()
    
    total_files = len(files_to_download)
    repo_type = "dataset" if args.type == "dataset" else "model"

    for idx, filename in enumerate(files_to_download):
        filename = filename.strip()
        if not filename: continue
        
        print(f"Processing file {idx + 1}/{total_files}: {filename}", file=sys.stderr)
        sys.stderr.flush()
        
        try:
            hf_hub_download(
                repo_id=args.repo_id,
                filename=filename,
                repo_type=repo_type,
                token=args.token,
                local_dir=base_folder,
                local_dir_use_symlinks=False,
            )
            
            # Emit progress update (per file completion)
            global_percent = int(((idx + 1) / total_files) * 100)
            print(f"PROGRESS:{global_percent}", file=sys.stderr)
            sys.stderr.flush()

        except Exception as e:
            print(f"Error downloading {filename}: {e}", file=sys.stderr)
            sys.exit(1)

    print("PROGRESS:100", file=sys.stderr)
    print("Download complete.", file=sys.stderr)

def cmd_convert(args):
    """Convert a downloaded dataset to Arrow format compatible with train.py."""
    try:
        from datasets import load_dataset
        import pandas as pd
    except ImportError:
        print(json.dumps({"error": "Missing required libraries. Please install 'datasets' and 'pandas'."}))
        sys.exit(1)

    source_path = os.path.abspath(args.source_path)
    output_path = os.path.abspath(args.output_path)
    processed_dir = os.path.join(output_path, "processed_data")

    if not os.path.exists(source_path):
        print(json.dumps({"error": f"Conversion failed: Unable to find '{source_path}'"}), file=sys.stderr)
        sys.exit(1)

    try:
        print(f"Loading dataset from {source_path}...", file=sys.stderr)
        
        # Recursively find all dataset files in the directory and subdirectories
        data_files = []
        if os.path.isdir(source_path):
            for root, dirs, files in os.walk(source_path):
                for file in files:
                    lower_file = file.lower()
                    if lower_file.endswith(('.csv', '.json', '.jsonl', '.parquet', '.arrow')):
                        file_path = os.path.join(root, file)
                        data_files.append(file_path)
        
        if not data_files:
            print(json.dumps({"error": f"Conversion failed: No dataset files found in '{source_path}'"}), file=sys.stderr)
            sys.exit(1)
        
        print(f"Found {len(data_files)} dataset file(s), loading dataset...", file=sys.stderr)
        
        # Detect file format from first file
        first_file = data_files[0].lower()
        
        if first_file.endswith(('.json', '.jsonl')):
            print("Detected JSON/JSONL. Using robust generator with schema normalization...", file=sys.stderr)
            
            from datasets import Dataset, Features, Sequence, Value

            # Define strict schema to prevent type errors while allowing multimodal data
            # Use list notation [{...}] for sequence of structs to avoid ambiguity
            features = Features({
                "conversations": [{
                    "role": Value("string"),
                    "content": Value("string"),
                    "images": Sequence(Value("string")),
                    "audio": Sequence(Value("string"))
                }]
            })

            def normalize_message(msg):
                # Standardized extraction
                role = 'unknown'
                content = ''
                images = []
                audio = []

                # 1. Extract Role & Content
                if 'role' in msg: role = msg['role']
                elif 'from' in msg: role = msg['from']
                
                if 'content' in msg: content = msg['content']
                elif 'value' in msg: content = msg['value']

                # 2. Extract Multimodal (Images) - handle string or list
                if 'image' in msg:
                    val = msg['image']
                    if isinstance(val, list): images.extend([str(v) for v in val])
                    elif val: images.append(str(val))
                if 'images' in msg:
                    val = msg['images']
                    if isinstance(val, list): images.extend([str(v) for v in val])
                    elif val: images.append(str(val))

                # 3. Extract Multimodal (Audio)
                if 'audio' in msg:
                    val = msg['audio']
                    if isinstance(val, list): audio.extend([str(v) for v in val])
                    elif val: audio.append(str(val))

                # 4. Normalize Role
                if role == 'human': role = 'user'
                if role == 'gpt': role = 'assistant'
                
                # 5. Return strict dict matching Feature schema
                return {
                    'role': str(role),
                    'content': str(content),
                    'images': images if images else [],    # Empty list for consistency
                    'audio': audio if audio else []        # Empty list for consistency
                }

            def json_generator(file_paths):
                for file_path in file_paths:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if not line.strip(): continue
                            try:
                                item = json.loads(line)
                                msgs = item.get('conversations', item.get('messages', []))
                                
                                if not msgs and isinstance(item, list):
                                    msgs = item
                                
                                if not msgs: continue 
                                
                                normalized_msgs = [normalize_message(m) for m in msgs]
                                yield {"conversations": normalized_msgs}
                            except Exception:
                                continue

            dataset = Dataset.from_generator(
                json_generator, 
                gen_kwargs={"file_paths": data_files},
                features=features  # Explicit schema prevents inferred type mismatches (NULL vs LIST)
            )
            
        elif first_file.endswith('.parquet'):
            dataset = load_dataset("parquet", data_files=data_files)
        elif first_file.endswith('.csv'):
            dataset = load_dataset("csv", data_files=data_files)
        elif first_file.endswith('.arrow'):
            dataset = load_dataset("arrow", data_files=data_files)
        else:
            # Try auto-detect
            dataset = load_dataset(data_files=data_files)
        
        print(f"Saving Arrow dataset to {processed_dir}...", file=sys.stderr)
        os.makedirs(processed_dir, exist_ok=True)
        dataset.save_to_disk(processed_dir)
        
        print(json.dumps({"success": True, "path": processed_dir}))

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"DEBUG: {tb}", file=sys.stderr)
        print(json.dumps({"error": f"Conversion failed: {str(e)}\n\nTraceback:\n{tb}"}), file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="HuggingFace Manager")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Search Command
    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("--query", type=str, required=True)
    search_parser.add_argument("--type", type=str, choices=["model", "dataset"], default="model")
    search_parser.add_argument("--limit", type=int, default=20)
    search_parser.add_argument("--token", type=str, default=None)

    # List Files Command
    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("--repo_id", type=str, required=True)
    list_parser.add_argument("--type", type=str, choices=["model", "dataset"], default="model")
    list_parser.add_argument("--token", type=str, default=None)

    # Download Command
    dl_parser = subparsers.add_parser("download")
    dl_parser.add_argument("--repo_id", type=str, required=True)
    dl_parser.add_argument("--files", type=str, help="Comma separated list of files")
    dl_parser.add_argument("--output", type=str, required=True)
    dl_parser.add_argument("--type", type=str, choices=["model", "dataset"], default="model") # Added type here
    dl_parser.add_argument("--token", type=str, default=None)

    # Convert Command
    conv_parser = subparsers.add_parser("convert")
    conv_parser.add_argument("--source_path", type=str, required=True)
    conv_parser.add_argument("--output_path", type=str, required=True)
    
    # Parse
    args = parser.parse_args()

    # Get token from Env if not provided
    if hasattr(args, 'token') and args.token is None:
        args.token = os.environ.get("HF_TOKEN")

    if args.command == "search":
        cmd_search(args)
    elif args.command == "list":
        cmd_list_files(args)
    elif args.command == "download":
        cmd_download(args)
    elif args.command == "convert":
        cmd_convert(args)

if __name__ == "__main__":
    main()