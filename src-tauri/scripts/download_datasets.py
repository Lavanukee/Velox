import os
import sys
import logging
import argparse
from pathlib import Path
import shutil

# Monkeypatch tqdm to capture progress
try:
    from tqdm import tqdm as original_tqdm
    import tqdm

    class CustomTqdm(original_tqdm):
        def update(self, n=1):
            super().update(n)
            if self.total and self.n:
                percent = int(self.n / self.total * 100)
                # Avoid printing too often to reduce overhead
                if not hasattr(self, '_last_percent') or self._last_percent != percent:
                    print(f"PROGRESS:{percent}", file=sys.stderr, flush=True)
                    self._last_percent = percent

    tqdm.tqdm = CustomTqdm
    # Also patch auto which is often used
    if hasattr(tqdm, 'auto'):
        tqdm.auto.tqdm = CustomTqdm
except ImportError:
    pass

from huggingface_hub import snapshot_download, HfApi

token = os.environ.get('HF_TOKEN', None)

# Configure logging to sys.stderr, which Tauri can capture
logging.basicConfig(level=logging.INFO, stream=sys.stderr, format='%(levelname)s: %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Download a Hugging Face dataset.")
    parser.add_argument(
        "--dataset_id",
        type=str,
        required=True,
        help="Hugging Face Dataset ID (e.g., 'HuggingFaceH4/ultrachat_200k')."
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Local directory to save the dataset, relative to src-tauri."
    )
    parser.add_argument(
        "--ignore_patterns",
        nargs='*',
        default=[],
        help="List of glob patterns to ignore during download (e.g., '*.git', '*.DS_Store')."
    )
    args = parser.parse_args()

    output_path = Path(args.output_folder)
    
    # Ensure the output directory exists before any disk operations or downloads
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        logging.info(f"Checking disk space for dataset '{args.dataset_id}'...")
        api = HfApi(token=token)
        dataset_info = api.repo_info(repo_id=args.dataset_id, repo_type="dataset")
        
        total_dataset_size = sum(file.size for file in dataset_info.siblings if file.size is not None)
            
        logging.info(f"Calculated total size for download: {total_dataset_size} bytes")
        
        check_path = output_path if output_path.exists() else Path('.')
        _, _, free_space = shutil.disk_usage(check_path.anchor)
        
        logging.info(f"Download size: {total_dataset_size / 1e9:.2f} GB, Available free space: {free_space / 1e9:.2f} GB")
        
        if total_dataset_size * 1.1 > free_space:
            error_msg = f"Not enough disk space. Required: {total_dataset_size / 1e9:.2f} GB (10% buffer), Available: {free_space / 1e9:.2f} GB"
            logging.error(f"ERROR: {error_msg}")
            print(f"ERROR: {error_msg}", file=sys.stderr, flush=True)
            sys.exit(1)

    except Exception as e:
        logging.warning(f"Warning: Could not check disk space for dataset download. Error: {e}")

    logging.info(f"Starting download of dataset '{args.dataset_id}' to '{output_path}'...")

    try:
        snapshot_download(
            repo_id=args.dataset_id,
            local_dir=str(output_path),
            local_dir_use_symlinks=False,
            repo_type="dataset",
            token=token,
            ignore_patterns=args.ignore_patterns
        )
        logging.info("Dataset download complete.")
        print("PROGRESS:100", file=sys.stderr, flush=True) # Emit 100% on completion
    except Exception as e:
        error_str = str(e)
        if "401" in error_str or "403" in error_str:
            logging.error(f"Authentication failed for '{args.dataset_id}'. This might be a gated repository. Please check your HF Token.")
        else:
            logging.error(f"Failed to download dataset '{args.dataset_id}': {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()