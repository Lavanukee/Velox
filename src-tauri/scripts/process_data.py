import os
import json
import random
import re
import argparse
from transformers import AutoProcessor, AutoTokenizer
from datasets import Dataset
from PIL import Image, UnidentifiedImageError
import requests
from huggingface_hub import configure_http_backend

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
    Convert collected data to training format.
    
    Input format:
    {
        "image": "images/screenshot_123.png",
        "conversations": [
            {"role": "user", "content": "<image>\nClick the submit button"},
            {"role": "assistant", "content": "<tools>click(500,300)</tools>"}
        ]
    }
    
    Output format:
    {
        "text": "formatted chat template",
        "image": PIL.Image
    }
    """
    texts, images = [], []
    
    for i in range(len(examples['image'])):
        try:
            image_path = examples['image'][i]
            conversations = examples['conversations'][i]
            
            # Validate conversations structure
            if not conversations or len(conversations) != 2:
                print(f"Skipping example {i}: Invalid conversation structure")
                continue
            
            # Get user prompt and assistant response
            user_msg = conversations[0]
            assistant_msg = conversations[1]
            
            if user_msg['role'] != 'user' or assistant_msg['role'] != 'assistant':
                print(f"Skipping example {i}: Invalid roles")
                continue
            
            # --- Tool Call Conversion ---
            # Convert <tool_call> JSON format to <tools>click(x,y)</tools> format
            assistant_content = assistant_msg['content']
            match = re.search(r'<tool_call>(.*?)</tool_call>', assistant_content, re.DOTALL)
            
            if match:
                json_str = match.group(1)
                tool_call_data = json.loads(json_str)
                
                if tool_call_data.get('name') == 'computer_use' and tool_call_data['arguments'].get('action') == 'left_click':
                    x, y = tool_call_data['arguments']['coordinate']
                    assistant_msg['content'] = f"<tools>click({x},{y})</tools>"
                else:
                    print(f"Skipping example {i}: Unsupported tool call format")
                    continue
            # If no <tool_call> is found, assume it's already in the correct format or skip if empty/invalid.
            # The original code handles the case where it's already in <tools> format.
            # --- End Tool Call Conversion ---

            # Load image
            # Remove leading 'images/' if present, as image_root already points to the images directory
            if image_path.startswith('images/'):
                image_path = image_path[len('images/'):]

            if not os.path.isabs(image_path):
                full_image_path = os.path.join(image_root, image_path)
            else:
                full_image_path = image_path
            
            image = Image.open(full_image_path).convert("RGB")
            
            # Build conversation for chat template
            # User message already has <image> tag from data collection
            conversation = [
                {"role": "user", "content": user_msg['content']},
                {"role": "assistant", "content": assistant_msg['content']}
            ]
            
            # Apply chat template
            text = processor.tokenizer.apply_chat_template(
                conversation, 
                tokenize=False, 
                add_generation_prompt=False
            )
            
            texts.append(text)
            images.append(image)
            
        except Exception as e:
            print(f"Skipping example {i}. Reason: {e}")
            continue
    
    return {"image": images, "text": texts}


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
    
    if os.path.exists(args.output_dir):
        print(f"Error: Output directory '{args.output_dir}' already exists.")
        print("Please remove it first or specify a different output path.")
        return

    print("\nLoading tokenizer and processor...")
    processor = AutoProcessor.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    processor.tokenizer = tokenizer
    print("✅ Tokenizer and processor loaded")
    
    print(f"\nLoading dataset from '{json_path}'...")
    data = load_jsonl_dataset(json_path)
    print(f"Found {len(data)} raw examples")
    
    # Create HuggingFace Dataset
    dataset = Dataset.from_list(data)
    print(f"✅ Dataset created with {len(dataset)} examples")
    
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
    print(f"✅ {len(dataset)} examples with valid images")

    if len(dataset) == 0:
        print("\n❌ CRITICAL ERROR: No examples remained after filtering.")
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
        print(f"✅ Train: {len(train_dataset)} examples")
        print(f"✅ Eval: {len(eval_dataset)} examples")
    else:
        train_dataset = dataset
        eval_dataset = None
        print(f"✅ Using all {len(train_dataset)} examples for training")

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
    
    print(f"✅ Processed {len(processed_train)} training examples")
    
    # Show processed sample
    print("\n" + "="*60)
    print("Sample Processed Entry:")
    print("="*60)
    sample = processed_train[0]
    print(f"Text (first 500 chars):\n{sample['text'][:500]}...")

    print("="*60)
    
    # Save training data
    print(f"\nSaving training data to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True) # Ensure output directory exists
    processed_train.save_to_disk(args.output_dir)
    print("✅ Training data saved")
    
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
        
        print(f"✅ Processed {len(processed_eval)} eval examples")
        print(f"Saving eval data to {eval_output_dir}...")
        os.makedirs(eval_output_dir, exist_ok=True) # Ensure eval output directory exists
        processed_eval.save_to_disk(eval_output_dir)
        print("✅ Eval data saved")
    
    print("\n" + "="*60)
    print("🎉 Data processing complete!")
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
        default=12,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Batch size for processing"
    )
    
    args = parser.parse_args()
    main(args)
