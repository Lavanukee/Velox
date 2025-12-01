# docker run --gpus all -it -p 8888:8888 -p 6006:6006 -v "C:\Users\Jedd\desktop\TuningPipeline:/workspace/unsloth" unsloth/unsloth:latest
# tensorboard --logdir=outputs


"""
Fine-tune vision-language model for UI click prediction
Adapted for <tools>click(x,y)</tools> format with normalized coordinates (1-1000)
"""

import os
from unsloth import FastLanguageModel
import torch
import numpy as np
from transformers import TrainingArguments, AutoProcessor, EarlyStoppingCallback
from datasets import Dataset
from trl import SFTTrainer
import re
import argparse # Added argparse

# ============================================================================
# METRICS FOR CLICK COORDINATE PREDICTION
# ============================================================================

def parse_click_from_text(text):
    """
    Extract click coordinates from model output.
    Supports formats:
    - <tools>click(500,300)</tools>
    - <tool_call>click(500,300)</tool_call>
    - click(500,300)
    
    Returns: (x, y) tuple or None if not found
    """
    try:
        # Try to find tools tags first
        if "<tools>" in text and "</tools>" in text:
            content = text.split("<tools>")[1].split("</tools>")[0]
        elif "<tool_call>" in text and "</tool_call>" in text:
            content = text.split("<tool_call>")[1].split("</tool_call>")[0]
        else:
            content = text
        
        # Extract click(x,y) pattern
        pattern = r'click\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)'
        match = re.search(pattern, content, re.IGNORECASE)
        
        if match:
            x = int(match.group(1))
            y = int(match.group(2))
            # Validate range (should be 1-1000 for normalized coords)
            if 1 <= x <= 1000 and 1 <= y <= 1000:
                return (x, y)
        
        return None
    except Exception as e:
        return None


def compute_click_distance(pred_coords, true_coords):
    """
    Compute normalized distance between predicted and true click coordinates.
    Returns value between 0 and 1 (0 = perfect match, 1 = maximum distance)
    """
    if pred_coords is None or true_coords is None:
        return 1.0  # Maximum error
    
    px, py = pred_coords
    tx, ty = true_coords
    
    # Euclidean distance in normalized space (1-1000)
    distance = np.sqrt((px - tx)**2 + (py - ty)**2)
    
    # Maximum possible distance is diagonal: sqrt(999^2 + 999^2) ≈ 1413
    max_distance = np.sqrt(999**2 + 999**2)
    
    # Normalize to 0-1
    normalized_distance = distance / max_distance
    
    return normalized_distance


def compute_click_accuracy(pred_coords, true_coords, threshold_pixels=50):
    """
    Check if click is within threshold pixels of target.
    Threshold in normalized space (e.g., 50 = 5% of 1000)
    """
    if pred_coords is None or true_coords is None:
        return False
    
    px, py = pred_coords
    tx, ty = true_coords
    
    distance = np.sqrt((px - tx)**2 + (py - ty)**2)
    return distance <= threshold_pixels


# tokenizer needs to be accessible globally for compute_metrics
tokenizer = None 

def compute_metrics(eval_pred):
    """
    Compute click prediction accuracy metrics.
    
    Metrics:
    - parsed_rate: % of responses with valid click format
    - distance_mean: Average normalized distance (0-1)
    - accuracy@50: % within 50 normalized pixels (~5%)
    - accuracy@100: % within 100 normalized pixels (~10%)
    - exact_match: % of exact coordinate matches
    """
    predictions, labels = eval_pred
    
    # Handle tuple output
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    # Convert logits to token IDs
    if len(predictions.shape) == 3:
        predictions = np.argmax(predictions, axis=-1)
    
    # Decode
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Compute metrics
    distances = []
    parsed = 0
    acc_50 = 0
    acc_100 = 0
    exact = 0
    total = len(decoded_preds)
    
    for pred_text, label_text in zip(decoded_preds, decoded_labels):
        pred_coords = parse_click_from_text(pred_text)
        true_coords = parse_click_from_text(label_text)
        
        if pred_coords is not None:
            parsed += 1
        
        if pred_coords and true_coords:
            # Distance metric
            distance = compute_click_distance(pred_coords, true_coords)
            distances.append(distance)
            
            # Accuracy thresholds
            if compute_click_accuracy(pred_coords, true_coords, threshold_pixels=50):
                acc_50 += 1
            if compute_click_accuracy(pred_coords, true_coords, threshold_pixels=100):
                acc_100 += 1
            
            # Exact match
            if pred_coords == true_coords:
                exact += 1
    
    return {
        "parsed_rate": parsed / total if total > 0 else 0.0,
        "distance_mean": np.mean(distances) if distances else 1.0,
        "accuracy@50": acc_50 / total if total > 0 else 0.0,
        "accuracy@100": acc_100 / total if total > 0 else 0.0,
        "exact_match": exact / total if total > 0 else 0.0,
    }


def main(args):
    global tokenizer # Declare tokenizer as global

    PROCESSED_TRAIN_DIR = args.train_data_dir
    PROCESSED_EVAL_DIR = args.eval_data_dir
    OUTPUT_DIR = args.output_dir
    
    model_name = args.model_name
    max_seq_length = args.max_seq_length
    
    print("="*60)
    print("Training Configuration - Click Prediction Model")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Format: <tools>click(x,y)</tools>")
    print(f"Coordinates: Normalized 1-1000")
    print(f"Training data: {PROCESSED_TRAIN_DIR}")
    print(f"Eval data: {PROCESSED_EVAL_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"LoRA rank: {args.lora_r} with dropout {args.lora_dropout}")
    print(f"Batch size: {args.per_device_train_batch_size} x {args.gradient_accumulation_steps} accumulation = {args.per_device_train_batch_size * args.gradient_accumulation_steps} effective")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.num_train_epochs}")
    print(f"Eval frequency: Every {args.eval_steps} steps")
    print("="*60)
    
    # ============================================================================
    # LOAD MODEL AND TOKENIZER
    # ============================================================================
    
    print("\nLoading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    processor = AutoProcessor.from_pretrained(model_name)
    processor.tokenizer = tokenizer
    print("✅ Model loaded")
    
    # ============================================================================
    # ADD LORA
    # ============================================================================
    
    
    # High-rank LoRA for precision
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,  # Much higher rank
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,  # Slight increase for regularization
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    
    
    print(f"✅ LoRA adapters added (r={args.lora_r}, dropout={args.lora_dropout})")
    
    # ============================================================================
    # LOAD DATASETS
    # ============================================================================
    
    print(f"\nLoading training data from {PROCESSED_TRAIN_DIR}...")
    if not os.path.exists(PROCESSED_TRAIN_DIR):
        print(f"ERROR: Training data not found at {PROCESSED_TRAIN_DIR}")
        print("Run process_collected_data.py first!")
        exit(1)
    
    try:
        # Load from HuggingFace Dataset format (Arrow files)
        train_dataset = Dataset.load_from_disk(PROCESSED_TRAIN_DIR)
        print(f"✅ Training examples: {len(train_dataset):,}")
        
        # Check dataset structure
        print(f"Dataset columns: {train_dataset.column_names}")
        
        # Verify required columns exist
        if 'text' not in train_dataset.column_names or 'image' not in train_dataset.column_names:
            print("ERROR: Dataset missing 'text' or 'image' columns!")
            print(f"Available columns: {train_dataset.column_names}")
            print("\nYour dataset appears to be in the wrong format.")
            print("Please run process_collected_data.py to create the correct format.")
            exit(1)
        
        # Show a sample
        print("\nSample training example:")
        sample = train_dataset[0]
        print(f"  Text preview: {sample['text'][:200]}...")
        if hasattr(sample['image'], 'size'):
            print(f"  Image size: {sample['image'].size}")
        
    except Exception as e:
        print(f"ERROR loading training data: {e}")
        print("\nTroubleshooting:")
        print(f"1. Check that {PROCESSED_TRAIN_DIR} exists")
        print(f"2. Verify it contains Arrow files (data-*.arrow)")
        print(f"3. Run process_collected_data.py if needed")
        exit(1)
    
    # Load eval dataset if it exists
    eval_dataset = None
    if os.path.exists(PROCESSED_EVAL_DIR):
        print(f"\nLoading eval data from {PROCESSED_EVAL_DIR}...")
        try:
            eval_dataset = Dataset.load_from_disk(PROCESSED_EVAL_DIR)
            # Limit eval size for speed
            eval_dataset = eval_dataset.select(range(min(100, len(eval_dataset))))
            print(f"✅ Eval examples: {len(eval_dataset)}")
        except Exception as e:
            print(f"⚠️ Could not load eval dataset: {e}")
            eval_dataset = None
    else:
        print("\n⚠️ No eval dataset found. Training without evaluation.")
        print(f"Expected location: {PROCESSED_EVAL_DIR}")
        print("To enable evaluation, run process_collected_data.py with --eval_split > 0")
    
    # ============================================================================
    # TRAINING ARGUMENTS
    # ============================================================================
    
    print("\nConfiguring training arguments...")
    
    
    training_args = TrainingArguments(
        # Training duration
        num_train_epochs=args.num_train_epochs,
        max_steps=-1,
        
        # Batch sizes
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,
        
        # Learning rate
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        
        # Mixed precision
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        
        # Optimizer
        optim=args.optim,
        
        # Logging
        logging_steps=args.logging_steps,
        
        # Evaluation
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=args.eval_steps if eval_dataset else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        
        # Best model selection
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        greater_is_better=False,
        
        # Output
        output_dir=OUTPUT_DIR,
        report_to="tensorboard",
        
        # Memory optimizations
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_pin_memory=False, # Keeping as False for now
        dataloader_num_workers=0, # Keeping as 0 for now
        lr_scheduler_type=args.lr_scheduler_type,
    )
    
    # ============================================================================
    # SETUP TRAINER
    # ============================================================================
    
    print("\nSetting up trainer...")
    
    trainer_kwargs = {
        "model": model,
        "train_dataset": train_dataset,
        "processor": processor,
        "max_seq_length": max_seq_length,
        "dataset_text_field": "text",
        "args": training_args,
    }
    
    if eval_dataset:
        trainer_kwargs["eval_dataset"] = eval_dataset
        trainer_kwargs["compute_metrics"] = compute_metrics
        trainer_kwargs["callbacks"] = [EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
    
    trainer = SFTTrainer(**trainer_kwargs)
    
    print("✅ Trainer configured")
    
    if eval_dataset:
        print("\nMetrics that will be tracked:")
        print("  - eval_loss: Overall loss (lower is better)")
        print("  - parsed_rate: % of valid click(x,y) predictions")
        print("  - distance_mean: Average click distance (0-1, lower is better)")
        print("  - accuracy@50: % of clicks within 50 pixels (~5%)")
        print("  - accuracy@100: % of clicks within 100 pixels (~10%)")
        print("  - exact_match: % of perfect coordinate matches")
    
    # ============================================================================
    # TRAIN
    # ============================================================================
    
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    print("\nMonitor progress:")
    print(f"  tensorboard --logdir {OUTPUT_DIR}")
    print("\nExpected time depends on dataset size:")
    print(f"  ~{len(train_dataset) // 1000} hours per epoch (approximate)")
    print("="*60)
    
    trainer.train()
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    
    # ============================================================================
    # SAVE FINAL MODEL
    # ============================================================================
    
    final_path = os.path.join(OUTPUT_DIR, "final_model")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\n✅ Model saved to {final_path}")
    
    print("\nNext steps:")
    print("1. Check TensorBoard for metrics")
    print("2. Merge LoRA adapters")
    print("3. Test inference with click predictions")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune vision-language model for UI click prediction."
    )
    
    # Paths and model
    parser.add_argument("--train_data_dir", type=str, default="unity_dataset",
                        help="Directory containing the processed training dataset.")
    parser.add_argument("--eval_data_dir", type=str, default="unity_dataset_eval",
                        help="Directory containing the processed evaluation dataset.")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Output directory for trained model and logs.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
                        help="Base model name for fine-tuning.")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Maximum sequence length for tokenizer.")

    # LoRA parameters
    parser.add_argument("--lora_r", type=int, default=512,
                        help="LoRA attention dimension (rank).")
    parser.add_argument("--lora_alpha", type=int, default=512,
                        help="Alpha parameter for LoRA scaling.")
    parser.add_argument("--lora_dropout", type=float, default=0.15,
                        help="Dropout probability for LoRA layers.")

    # TrainingArguments parameters
    parser.add_argument("--num_train_epochs", type=int, default=20,
                        help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2,
                        help="Batch size per device during training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2,
                        help="Batch size per device during evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--eval_accumulation_steps", type=int, default=2,
                        help="Number of predictions steps to accumulate before moving the tensors to the CPU.")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Initial learning rate for AdamW optimizer.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Proportion of training steps to perform linear warmup for.")
    parser.add_argument("--optim", type=str, default="adamw_8bit",
                        help="Optimizer to use (e.g., 'adamw_8bit', 'adamw_hf').")
    parser.add_argument("--logging_steps", type=int, default=50,
                        help="Log every N updates steps.")
    parser.add_argument("--eval_steps", type=int, default=100,
                        help="Run an evaluation every N steps during training.")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="Save checkpoint every N updates steps.")
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="Limit the total amount of checkpoints. Deletes older checkpoints.")
    parser.add_argument("--gradient_checkpointing", type=bool, default=True,
                        help="Whether to use gradient checkpointing for memory efficiency.")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                        help="Learning rate scheduler type (e.g., 'linear', 'cosine').")
    parser.add_argument("--early_stopping_patience", type=int, default=5,
                        help="Patience for early stopping.")

    args = parser.parse_args()
    main(args)