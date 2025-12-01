import argparse
import json
import os
import sys
from huggingface_hub import HfApi, utils

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, stream=sys.stderr, format='%(message)s')

def main():
    parser = argparse.ArgumentParser(description="List files in a Hugging Face repository.")
    parser.add_argument("--repo_id", type=str, required=True, help="Repository ID")
    parser.add_argument("--repo_type", type=str, default="model", help="Repository type (model, dataset)")
    args = parser.parse_args()

    token = os.environ.get('HF_TOKEN', None)
    api = HfApi(token=token)

    try:
        # Get repo info which includes siblings (files)
        repo_info = api.repo_info(repo_id=args.repo_id, repo_type=args.repo_type, files_metadata=True)
        
        files = []
        for file in repo_info.siblings:
            files.append({
                "path": file.rfilename,
                "size": file.size,
                "lfs": file.lfs if hasattr(file, 'lfs') else None
            })
        
        # Output JSON to stdout
        print(json.dumps(files))

    except utils.RepositoryNotFoundError:
        print(json.dumps({"error": "Repository not found"}), file=sys.stdout)
        sys.exit(1)
    except utils.GatedRepoError:
        print(json.dumps({"error": "Gated repository. Token required."}), file=sys.stdout)
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stdout)
        sys.exit(1)

if __name__ == "__main__":
    main()
