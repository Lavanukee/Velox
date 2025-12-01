import argparse
import json
import sys
import os
import requests
import shutil
from typing import Optional, List, Dict, Any
from huggingface_hub import HfApi, configure_http_backend

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
    """Download specific files from a repository."""
    # Ensure token is used if provided
    
    base_folder = os.path.abspath(args.output)
    os.makedirs(base_folder, exist_ok=True)
    
    files_to_download = args.files.split(',') if args.files else []
    
    if not files_to_download:
        print("Error: No files specified for download.", file=sys.stderr)
        sys.exit(1)

    print(f"Starting download to: {base_folder}", file=sys.stderr)
    sys.stderr.flush()
    
    total_files = len(files_to_download)
    
    for idx, filename in enumerate(files_to_download):
        filename = filename.strip()
        if not filename: continue

        # Construct destination path
        dest_path = os.path.join(base_folder, filename)
        dest_dir = os.path.dirname(dest_path)
        os.makedirs(dest_dir, exist_ok=True)
        
        print(f"Processing file {idx + 1}/{total_files}: {filename}", file=sys.stderr)
        sys.stderr.flush()
        
        try:
            # Get the URL
            # Note: repo_type needs to be in URL for datasets? 
            # huggingface.co/datasets/user/repo/resolve... vs huggingface.co/user/repo/resolve...
            
            url_type_segment = "datasets/" if args.type == "dataset" else ""
            url = f"https://huggingface.co/{url_type_segment}{args.repo_id}/resolve/main/{filename}"
            headers = get_auth_headers(args.token)
            
            # Emit start of file download
            print(f"PROGRESS:{int((idx / total_files) * 100)}", file=sys.stderr)
            sys.stderr.flush()
            
            # Stream download
            with requests.get(url, headers=headers, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                block_size = 1024 * 1024 # 1MB chunks
                wrote = 0
                
                with open(dest_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=block_size):
                        if chunk:
                            wrote += len(chunk)
                            f.write(chunk)
                            
                            # Calculate progress
                            if total_size > 0:
                                file_progress = (wrote / total_size)
                                global_percent = int(((idx + file_progress) / total_files) * 100)
                                print(f"PROGRESS:{global_percent}", file=sys.stderr)
                                sys.stderr.flush()

        except Exception as e:
            print(f"Error downloading {filename}: {e}", file=sys.stderr)
            sys.stderr.flush()
            sys.exit(1)

    print("PROGRESS:100", file=sys.stderr)
    sys.stderr.flush()
    print("Download complete.", file=sys.stderr)
    sys.stderr.flush()

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
        if first_file.endswith('.parquet'):
            dataset = load_dataset("parquet", data_files=data_files)
        elif first_file.endswith('.csv'):
            dataset = load_dataset("csv", data_files=data_files)
        elif first_file.endswith(('.json', '.jsonl')):
            dataset = load_dataset("json", data_files=data_files)
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
        print(json.dumps({"error": f"Conversion failed: {str(e)}"}), file=sys.stderr)
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