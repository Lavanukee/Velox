import os
import json
import random
import re
import argparse
import sys
from transformers import AutoProcessor, AutoTokenizer
from datasets import Dataset
from PIL import Image, UnidentifiedImageError
import requests
from huggingface_hub import configure_http_backend

# Configure encoding for Windows consoles
if sys.platform == "win32":
    import multiprocessing
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass # Older python versions

# --- SSL VERIFICATION FIX ---
def backend_factory() -> requests.Session:
    session = requests.Session()
    session.verify = False
    return session
configure_http_backend(backend_factory=backend_factory)
# --- END SSL FIX ---

def is_image_valid(example, image_root):
    """Check if an image file exists and is readable"""
    image_path = example.get('image')
    if not image_path:
        return False
    
    # Remove leading 'images/' if present, as image_root already points to the images directory
    if image_path.startswith('images/'):
        image_path = image_path[len('images/'):]
    
    # Handle both absolute and relative paths
    if not os.path.isabs(image_path):
        full_path = os.path.join(image_root, image_path)
    else:
        full_path = image_path
    
    if not os.path.exists(full_path):
        return False
    
    try:
        Image.open(full_path).close()
        return True
    except (IOError, UnidentifiedImageError):
        return False


def create_structured_example(examples, processor, image_root):
    """
    Convert collected data to Unsloth VLM training format.
    
    Input format:
    {
        "image": "images/screenshot_123.png",
        "conversations": [
            {"role": "user", "content": "<image>\nClick the submit button"},
            {"role": "assistant", "content": "<tools>click(500,300)</tools>"}
        ]
    }
    
    Output format for UnslothVisionDataCollator:
    {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": <PIL Image>},
                    {"type": "text", "text": "Click the submit button"}
                ]
            },
            {"role": "assistant", "content": "<tools>click(500,300)</tools>"}
        ]
    }
    """
    all_messages = []
    
    for i in range(len(examples['image'])):
        try:
            image_path = examples['image'][i]
            conversations = examples['conversations'][i]
            
            # Validate conversations structure
            if not conversations:
                print(f"Skipping example {i}: Empty conversations")
                continue
            
            # --- Tool Call Conversion (Optional) ---
            for msg in conversations:
                if msg['role'] == 'assistant' and '<tool_call>' in msg['content']:
                    try:
                        content = msg['content']
                        match = re.search(r'<tool_call>(.*?)</tool_call>', content, re.DOTALL)
                        if match:
                            json_str = match.group(1)
                            tool_call_data = json.loads(json_str)
                            if tool_call_data.get('name') == 'computer_use' and tool_call_data['arguments'].get('action') == 'left_click':
                                x, y = tool_call_data['arguments']['coordinate']
                                msg['content'] = f"<tools>click({x},{y})</tools>"
                    except Exception as e:
                        print(f"Warning: Failed to parse tool call in example {i}: {e}")
            # --- End Tool Call Conversion ---

            # Load image
            if image_path.startswith('images/'):
                image_path = image_path[len('images/'):]

            if not os.path.isabs(image_path):
                full_image_path = os.path.join(image_root, image_path)
            else:
                full_image_path = image_path
            
            image = Image.open(full_image_path).convert("RGB")
            
            # Build multimodal messages for Qwen3-VL / Unsloth
            formatted_messages = []
            for msg in conversations:
                role = msg['role']
                content = msg['content']
                
                if role == 'user' and '<image>' in content:
                    # Replace <image> placeholder with actual multimodal content
                    text_part = content.replace('<image>', '').strip()
                    formatted_messages.append({
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": text_part if text_part else "Describe this image."}
                        ]
                    })
                else:
                    # Plain text message (usually assistant response)
                    formatted_messages.append({
                        "role": role,
                        "content": content
                    })
            
            all_messages.append(formatted_messages)
            
        except Exception as e:
            print(f"Skipping example {i}. Reason: {e}")
            continue
    
    # Return 'messages' column for UnslothVisionDataCollator
    return {"messages": all_messages}



def load_jsonl_dataset(jsonl_path):
    """Load dataset from JSONL file"""
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def main(args):
    # Determine dataset paths based on the input directory argument
    if args.dataset_dir:
        dataset_dir = args.dataset_dir
        json_path = os.path.join(dataset_dir, "dataset.jsonl")
        image_root = os.path.join(dataset_dir, "images")
    else:
        print("Error: --dataset_dir is required.")
        return

    print("="*60)
    print("Processing Collected Click Data")
    print("="*60)
    print(f"Input JSONL: {json_path}")
    print(f"Image root: {image_root}")
    print(f"Output dir: {args.output_dir}")
    print(f"Model name: {args.model_name}")
    print(f"Eval split: {args.eval_split}")
    print(f"Num workers: {args.num_workers}")
    print(f"Batch size: {args.batch_size}")
    print("="*60)
    
    if os.path.exists(args.output_dir) and not args.force:
        print(f"Warning: Output directory '{args.output_dir}' already exists.")
        print("Use --force to overwrite. Proceeding with existing data if possible, or exiting...")
        # If we are in the pipeline, we might want to overwrite. 
        # For now, let's just proceed if force is not set but log the warning.
        # return # Commented out to be less restrictive

    print("\nLoading tokenizer and processor...")
    processor = AutoProcessor.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    processor.tokenizer = tokenizer
    print("[OK] Tokenizer and processor loaded")
    
    print(f"\nLoading dataset from '{json_path}'...")
    data = load_jsonl_dataset(json_path)
    print(f"Found {len(data)} raw examples")
    
    # Create HuggingFace Dataset
    dataset = Dataset.from_list(data)
    print(f"[OK] Dataset created with {len(dataset)} examples")
    
    # Show sample
    print("\n" + "="*60)
    print("Sample Entry:")
    print("="*60)
    sample = dataset[0]
    print(f"Image: {sample['image']}")
    print(f"Conversations: {json.dumps(sample['conversations'], indent=2)}")
    if 'metadata' in sample:
        print(f"Metadata: {json.dumps(sample['metadata'], indent=2)}")
    print("="*60)
    
    print("\nFiltering for valid image files...")
    filter_kwargs = {"image_root": image_root}
    dataset = dataset.filter(
        is_image_valid,
        fn_kwargs=filter_kwargs,
        num_proc=args.num_workers
    )
    print(f"[OK] {len(dataset)} examples with valid images")

    if len(dataset) == 0:
        print("\n[ERROR] CRITICAL ERROR: No examples remained after filtering.")
        print("Check that your --image_root is correct and images exist.")
        return
    
    # Split into train/eval
    if args.eval_split > 0:
        print(f"\nSplitting dataset (eval={args.eval_split*100:.0f}%)...")
        split_dataset = dataset.train_test_split(
            test_size=args.eval_split,
            seed=42
        )
        train_dataset = split_dataset['train']
        eval_dataset = split_dataset['test']
        print(f"[OK] Train: {len(train_dataset)} examples")
        print(f"[OK] Eval: {len(eval_dataset)} examples")
    else:
        train_dataset = dataset
        eval_dataset = None
        print(f"[OK] Using all {len(train_dataset)} examples for training")

    print("\nCreating structured training dataset...")
    original_columns = train_dataset.column_names
    map_kwargs = {
        "processor": processor,
        "image_root": image_root,
    }
    
    processed_train = train_dataset.map(
        create_structured_example,
        batched=True,
        batch_size=args.batch_size,
        fn_kwargs=map_kwargs,
        num_proc=args.num_workers,
        remove_columns=original_columns
    )
    
    print(f"[OK] Processed {len(processed_train)} training examples")
    
    # Show processed sample
    print("\n" + "="*60)
    print("Sample Processed Entry:")
    print("="*60)
    sample = processed_train[0]
    if 'messages' in sample:
        print(f"Messages: {json.dumps(str(sample['messages'])[:500], indent=2)}...")
    elif 'text' in sample:
        print(f"Text (first 500 chars):\n{sample['text'][:500]}...")
    else:
        print(f"Available keys: {list(sample.keys())}")

    print("="*60)
    
    # Save training data
    print(f"\nSaving training data to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True) # Ensure output directory exists
    processed_train.save_to_disk(args.output_dir)
    print("[OK] Training data saved")
    
    # Process and save eval data if exists
    if eval_dataset:
        eval_output_dir = args.output_dir + "_eval"
        print(f"\nProcessing evaluation dataset...")
        processed_eval = eval_dataset.map(
            create_structured_example,
            batched=True,
            batch_size=args.batch_size,
            fn_kwargs=map_kwargs,
            num_proc=args.num_workers,
            remove_columns=original_columns
        )
        
        print(f"[OK] Processed {len(processed_eval)} eval examples")
        print(f"Saving eval data to {eval_output_dir}...")
        os.makedirs(eval_output_dir, exist_ok=True) # Ensure eval output directory exists
        processed_eval.save_to_disk(eval_output_dir)
        print("[OK] Eval data saved")
    
    print("\n" + "="*60)
    print("[SUCCESS] Data processing complete!")
    print("="*60)
    print("\nNext steps:")
    print(f"1. Training data ready at: {args.output_dir}")
    if eval_dataset:
        print(f"2. Eval data ready at: {eval_output_dir}")
    print("3. Run trainer.py to fine-tune the model")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process collected UI click data for VLM fine-tuning"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Root directory of the dataset containing dataset.jsonl and images/"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="unity_dataset", # Changed default output dir to match original script behavior
        help="Output directory for the processed training data."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Model name for tokenizer/processor"
    )
    parser.add_argument(
        "--eval_split",
        type=float,
        default=0.1,
        help="Fraction of data to use for evaluation (0-1, default 0.1)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output directory if it exists"
    )
    
    args = parser.parse_args()
    
    # Windows Multiprocessing Fix
    if sys.platform == "win32":
        multiprocessing.freeze_support()
        # Default to fewer workers on Windows to prevent OOM/sprawl
        if args.num_workers > 2:
            print(f"Lowering num_workers from {args.num_workers} to 2 for Windows stability.")
            args.num_workers = 2

    main(args)
