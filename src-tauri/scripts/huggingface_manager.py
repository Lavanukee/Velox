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



# ============================================================================
# COMPREHENSIVE DATASET FORMAT DETECTION AND CONVERSION SYSTEM
# ============================================================================
# Supports: VLM Grounding (GroundUI), VLM Captioning, ShareGPT, Alpaca, DPO,
#           Q&A, Code, Prompt/Response, Multi-turn, and many fallbacks
# ============================================================================

class DatasetFormatDetector:
    """
    Comprehensive dataset format detection with confidence scoring.
    Returns the best conversion strategy for any given dataset structure.
    Uses HuggingFace Features API for robust modality detection.
    """
    
    # Format priority order (higher = tried first)
    FORMAT_PRIORITY = [
        'vlm_messages',      # Already in VLM messages format (ideal)
        'vlm_grounding',     # Image + instruction + bbox (GroundUI-1K style)
        'vlm_captioning',    # Image + caption pairs
        'vlm_vqa',           # Image + question + answer
        'conversations',     # ShareGPT/conversations style
        'messages',          # OpenAI messages format
        'alpaca',            # instruction/input/output
        'dpo',              # chosen/rejected pairs
        'qa',               # question/answer
        'code',             # code instruction/response
        'prompt_response',  # prompt/response or prompt/completion
        'text_only',        # Just text column
        'heuristic',        # Last resort - find any usable text
    ]
    
    @staticmethod
    def detect_modality(dataset):
        """
        Detect dataset modality and column types using Features API.
        Returns detailed info about image, audio, and text columns.
        """
        from datasets import Image, Audio, Value, Sequence, ClassLabel
        
        info = {
            'has_images': False,
            'has_audio': False,
            'image_cols': [],
            'audio_cols': [],
            'text_cols': [],
            'list_cols': [],
            'bbox_cols': [],
            'structure_cols': []  # dict/struct columns
        }
        
        # If dataset has no features (e.g. IterabelDataset or raw dict), fallback to column names
        if not hasattr(dataset, 'features') or not dataset.features:
            return DatasetFormatDetector._fallback_modality_detection(dataset)
            
        for col_name, feature in dataset.features.items():
            # Check for Image
            if isinstance(feature, Image):
                info['has_images'] = True
                info['image_cols'].append(col_name)
                continue
                
            # Check for Audio
            if isinstance(feature, Audio):
                info['has_audio'] = True
                info['audio_cols'].append(col_name)
                continue
                
            # Check for generic text
            if isinstance(feature, Value) and feature.dtype == 'string':
                info['text_cols'].append(col_name)
                
            # Check for Lists/Sequences
            if isinstance(feature, Sequence) or isinstance(feature, list):
                info['list_cols'].append(col_name)
                # Check for bbox-like sequences (list of 4 floats/ints)
                try:
                    if isinstance(feature, Sequence) and isinstance(feature.feature, Value):
                        if feature.feature.dtype in ['float32', 'float64', 'int32', 'int64']:
                             if col_name in ['bbox', 'bounding_box', 'box', 'coordinates']:
                                 info['bbox_cols'].append(col_name)
                except:
                    pass
                    
            # Check for structures (dictionaries), e.g. 'messages' or 'conversations'
            if isinstance(feature, dict) or (hasattr(feature, 'dtype') and feature.dtype == 'struct'):
                info['structure_cols'].append(col_name)
        
        return info

    @staticmethod
    def _fallback_modality_detection(dataset):
        """Fallback detection based on column names if Features API not available."""
        columns = set(dataset.column_names)
        info = {
            'has_images': False, 'image_cols': [],
            'has_audio': False, 'audio_cols': [],
            'text_cols': [], 'list_cols': [], 'bbox_cols': [], 'structure_cols': []
        }
        for col in columns:
            col_lower = col.lower()
            if col_lower in ['image', 'img', 'images', 'picture', 'photo']:
                info['has_images'] = True
                info['image_cols'].append(col)
            elif col_lower in ['audio', 'sound', 'voice']:
                info['has_audio'] = True
                info['audio_cols'].append(col)
            elif col_lower in ['bbox', 'bounding_box']:
                info['bbox_cols'].append(col)
            else:
                info['text_cols'].append(col)
        return info
    
    @staticmethod
    def detect_format(dataset) -> tuple:
        """
        Detect the format of a dataset.
        Returns: (format_name, confidence, detection_details)
        """
        modality_info = DatasetFormatDetector.detect_modality(dataset)
        columns = set(dataset.column_names)
        sample = dataset[0] if len(dataset) > 0 else {}
        
        print(f"  Modality Info: {modality_info}", file=sys.stderr)
        
        results = []
        
        for fmt in DatasetFormatDetector.FORMAT_PRIORITY:
            if fmt == 'heuristic':
                continue
            
            score, details = DatasetFormatDetector._score_format(fmt, columns, sample, dataset, modality_info)
            if score > 0:
                results.append((fmt, score, details))
        
        # Sort by confidence
        results.sort(key=lambda x: x[1], reverse=True)
        
        if results:
            return results[0]
            
        return ('heuristic', 0.1, {'reason': 'No matching format found, using heuristic'})
    
    @staticmethod
    def _score_format(fmt, columns, sample, dataset, modality_info) -> tuple:
        """Score a specific format based on modality and columns."""
        score = 0
        details = {'matched': [], 'missing': []}
        
        # --- VLM Formats ---
        if fmt == 'vlm_messages':
            if 'messages' in columns:
                score += 2
                # Check for proper multimodal structure
                if DatasetFormatDetector._has_multimodal_messages(dataset):
                    score += 3  # Strong signal
                    details['is_multimodal_ready'] = True
                else:
                    details['is_multimodal_ready'] = False
        
        elif fmt == 'vlm_grounding':
            # Needs image + text + optional bbox
            if modality_info['has_images']:
                score += 2
                matched_instruct = set(modality_info['text_cols']) & {'instruction', 'prompt', 'query', 'text', 'caption'}
                if matched_instruct:
                    score += 1
                if 'bbox' in columns or 'bounding_box' in columns or modality_info['bbox_cols']:
                    score += 2  # Strong indicator for detection/grounding
        
        elif fmt == 'vlm_captioning':
            # Needs image + caption/text
            if modality_info['has_images']:
                score += 2
                matched = set(modality_info['text_cols']) & {'caption', 'text', 'description', 'label', 'alt_text'}
                if matched:
                    score += 1.5
                # Verify it's NOT a grounding dataset
                if 'bbox' not in columns and not modality_info['bbox_cols']:
                    score += 0.5
        
        elif fmt == 'vlm_vqa':
            # Needs image + question + answer
            if modality_info['has_images']:
                score += 2
                has_q = bool(set(modality_info['text_cols']) & {'question', 'query'})
                has_a = bool(set(modality_info['text_cols']) & {'answer', 'response', 'label'})
                if has_q and has_a:
                    score += 2
        
        # --- Text Formats ---
        elif fmt == 'conversations':
            if 'conversations' in columns:
                score += 4
        
        elif fmt == 'messages':
            if 'messages' in columns and not modality_info['has_images']:
                score += 3
        
        elif fmt == 'alpaca':
            if 'instruction' in columns and 'output' in columns:
                score += 4
        
        elif fmt == 'dpo':
            if 'chosen' in columns and 'rejected' in columns:
                score += 4
        
        elif fmt == 'qa':
            if 'question' in columns and 'answer' in columns:
                score += 3
                
        elif fmt == 'code':
            if ('code' in columns or 'solution' in columns) and ('problem' in columns or 'instruction' in columns):
                score += 3
                
        elif fmt == 'prompt_response':
            if 'prompt' in columns and ('response' in columns or 'completion' in columns):
                score += 3
                
        elif fmt == 'text_only':
            if not modality_info['has_images'] and len(modality_info['text_cols']) == 1 and len(columns) == 1:
                score += 2
            if 'text' in columns:
                 score += 1

        return (score, details)

    @staticmethod
    def _has_multimodal_messages(dataset):
        """Check if messages column contains proper multimodal format."""
        try:
            sample = dataset[0]
            msgs = sample.get('messages', [])
            if not msgs or not isinstance(msgs, list):
                return False
            # Check structure: list of dicts with role/content
            first_msg = msgs[0]
            content = first_msg.get('content')
            
            # Check if content is list (multimodal signal)
            if isinstance(content, list):
                has_image_type = any(item.get('type') == 'image' for item in content)
                return has_image_type
            return False
        except:
            return False


def detect_and_normalize_schema(dataset, is_vision_model=False):
    """
    Detect dataset schema and convert to training-ready format.
    
    For VLM training (is_vision_model=True): outputs 'messages' column with multimodal content
    For text training: outputs 'conversations' or 'text' column
    
    Returns: (converted_dataset, format_info)
    """
    from datasets import Dataset
    
    columns = set(dataset.column_names)
    print(f"  Detecting format for columns: {columns}", file=sys.stderr)
    
    # Detect format
    detected_format, confidence, details = DatasetFormatDetector.detect_format(dataset)
    print(f"  Detected format: {detected_format} (confidence: {confidence:.2f})", file=sys.stderr)
    print(f"  Details: {details}", file=sys.stderr)
    
    # Route to appropriate converter
    converters = {
        'vlm_messages': lambda ds: ds,  # Already good
        'vlm_grounding': convert_vlm_grounding,
        'vlm_captioning': convert_vlm_captioning,
        'vlm_vqa': convert_vlm_vqa,
        'conversations': normalize_conversations_dataset,
        'messages': convert_messages_to_conversations,
        'alpaca': convert_alpaca_format,
        'dpo': convert_dpo_pairs,
        'qa': convert_qa_format,
        'code': convert_code_format,
        'prompt_response': convert_prompt_response,
        'text_only': lambda ds: ds,  # Keep as-is
        'heuristic': attempt_heuristic_conversion,
    }
    
    converter = converters.get(detected_format, attempt_heuristic_conversion)
    
    try:
        converted = converter(dataset)
        
        # Capture metadata for UI tags
        modality_info = DatasetFormatDetector.detect_modality(dataset)
        modalities = []
        if modality_info['has_images']: modalities.append("Vision")
        if modality_info['has_audio']: modalities.append("Audio")
        if modality_info['text_cols']: modalities.append("Text")
        
        format_info = {
            'detected_format': detected_format,
            'confidence': confidence,
            'original_columns': list(columns),
            'final_columns': converted.column_names,
            'num_examples': len(converted),
            'modalities': modalities,
            'success': True,
        }
        
        print(f"  ✓ Conversion successful. Final columns: {converted.column_names}", file=sys.stderr)
        return converted, format_info
        
    except Exception as e:
        print(f"  ✗ Conversion failed: {e}", file=sys.stderr)
        # Try heuristic as fallback
        try:
            converted = attempt_heuristic_conversion(dataset, columns)
            return converted, {
                'detected_format': 'heuristic_fallback',
                'original_format': detected_format,
                'error': str(e),
                'success': True,
            }
        except Exception as e2:
            return dataset, {
                'detected_format': 'failed',
                'error': str(e2),
                'success': False,
            }


def convert_vlm_grounding(dataset):
    """
    Convert VLM grounding datasets (like GroundUI-1K) to training format.
    Input: image + instruction + bbox → Output: multimodal messages for grounding
    """
    # Detect columns using Features API if possible
    modality = DatasetFormatDetector.detect_modality(dataset)
    columns = set(dataset.column_names)
    
    # Smart column detection
    image_col = next((c for c in modality['image_cols']), None) or \
                next((c for c in ['image', 'image_path', 'screenshot', 'img'] if c in columns), None)
                
    # Prefer text columns that look like instructions
    possible_instructions = modality['text_cols'] or list(columns)
    instruction_col = next((c for c in ['instruction', 'prompt', 'query', 'text', 'raw_instruction'] if c in possible_instructions), None)
    
    bbox_col = next((c for c in modality['bbox_cols']), None) or \
               next((c for c in ['bbox', 'bounding_box', 'box', 'coordinates', 'target'] if c in columns), None)
               
    resolution_col = next((c for c in ['resolution', 'screen_size', 'size'] if c in columns), None)
    
    import os
    num_proc = os.cpu_count() or 1
    print(f"    VLM Grounding: image={image_col}, instruction={instruction_col}, bbox={bbox_col}, num_proc={num_proc}", file=sys.stderr)
    
    def transform(example):
        # Get values
        image = example.get(image_col) if image_col else None
        instruction = example.get(instruction_col, 'Describe this image.') if instruction_col else 'Describe this image.'
        bbox = example.get(bbox_col) if bbox_col else None
        resolution = example.get(resolution_col) if resolution_col else None
        
        # Format bbox for output
        if bbox is not None:
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                # Normalize bbox format
                x1, y1, x2, y2 = bbox[:4]
                # If we have resolution, convert to relative coordinates
                if resolution and len(resolution) >= 2:
                    w, h = resolution[:2]
                    if w > 0 and h > 0:
                        # Convert to center point (common for click grounding)
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        assistant_content = f"<point>({cx:.1f}, {cy:.1f})</point>"
                    else:
                        assistant_content = f"<box>({x1}, {y1}, {x2}, {y2})</box>"
                else:
                    assistant_content = f"<box>({x1}, {y1}, {x2}, {y2})</box>"
            else:
                assistant_content = str(bbox)
        else:
            assistant_content = "I cannot locate the specified element."
        
        # Build multimodal messages ensuring STRICT consistency
        # User message: ALWAYS a list of dicts
        # Assistant: ALWAYS a list of dicts (for Arrow consistency)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": str(instruction)}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": str(assistant_content)}
                ]
            }
        ]
        
        return {
            "messages": messages,
            "image": image,  # Preserve original image
        }
    
    return dataset.map(transform, num_proc=num_proc)


def convert_vlm_captioning(dataset):
    """Convert image-caption pairs to VLM training format."""
    # Detect columns
    modality = DatasetFormatDetector.detect_modality(dataset)
    columns = set(dataset.column_names)
    
    image_col = next((c for c in modality['image_cols']), None) or \
                next((c for c in ['image', 'image_path', 'url', 'image_url', 'file_name', 'img'] if c in columns), None)
    
    caption_col = next((c for c in ['caption', 'text', 'description', 'label', 'alt_text', 'title'] if c in columns), None)
    
    print(f"    VLM Captioning: image={image_col}, caption={caption_col}", file=sys.stderr)
    
    def transform(example):
        image = example.get(image_col) if image_col else None
        caption = example.get(caption_col, '') if caption_col else ''
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe this image in detail."}
                ]
            },
            {
                "role": "assistant",
                "content": str(caption)
            }
        ]
        
        return {
            "messages": messages,
            "image": image,
        }
    
    return dataset.map(transform, num_proc=1)


def convert_vlm_vqa(dataset):
    """Convert VQA (Visual Question Answering) datasets."""
    # Detect columns
    modality = DatasetFormatDetector.detect_modality(dataset)
    columns = set(dataset.column_names)
    
    image_col = next((c for c in modality['image_cols']), None) or \
                next((c for c in ['image', 'image_path', 'img'] if c in columns), None)
    
    question_col = next((c for c in ['question', 'query', 'prompt', 'text'] if c in columns), None)
    answer_col = next((c for c in ['answer', 'response', 'output', 'label'] if c in columns), None)
    
    print(f"    VLM VQA: image={image_col}, question={question_col}, answer={answer_col}", file=sys.stderr)
    
    def transform(example):
        image = example.get(image_col) if image_col else None
        question = example.get(question_col, 'What is in this image?') if question_col else 'What is in this image?'
        answer = example.get(answer_col, '') if answer_col else ''
        
        # Handle list of answers (common in VQA)
        if isinstance(answer, list):
            answer = answer[0] if answer else ''
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": str(question)}
                ]
            },
            {
                "role": "assistant",
                "content": str(answer)
            }
        ]
        
        return {
            "messages": messages,
            "image": image,
        }
    
    return dataset.map(transform, num_proc=1)


def normalize_conversations_dataset(dataset):
    """Normalize existing conversations dataset and add messages format."""
    def normalize_fn(example):
        msgs = example.get('conversations') or example.get('messages', [])
        normalized_convs = []
        normalized_msgs = []
        
        # Get image if present
        image = example.get('image') or example.get('images')
        if isinstance(image, list) and image:
            image = image[0]
        
        first_user = True
        for msg in msgs:
            role = msg.get('role') or msg.get('from', 'unknown')
            content = msg.get('content') or msg.get('value', '')
            
            # Normalize role names
            if role in ('human', 'user', 'User', 'Human'):
                role = 'user'
            if role in ('gpt', 'assistant', 'Assistant', 'GPT', 'bot', 'Bot'):
                role = 'assistant'
            if role == 'system':
                role = 'system'
            
            # For conversations format
            normalized_convs.append({
                'role': role, 
                'content': str(content), 
                'images': [], 
                'audio': []
            })
            
            # For messages format (VLM-ready)
            if role == 'user' and first_user and image is not None:
                # First user message gets the image
                normalized_msgs.append({
                    "role": role,
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": str(content).replace('<image>', '').strip()}
                    ]
                })
                first_user = False
            else:
                normalized_msgs.append({
                    "role": role,
                    "content": str(content)
                })
        
        result = {'conversations': normalized_convs, 'messages': normalized_msgs}
        if image is not None:
            result['image'] = image
        return result
    
    return dataset.map(normalize_fn, num_proc=1)


def convert_messages_to_conversations(dataset):
    """Convert OpenAI messages format to our standard format."""
    def transform(example):
        messages = example.get('messages', [])
        convs = []
        for msg in messages:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            # Handle multimodal content
            if isinstance(content, list):
                text_parts = [item.get('text', '') for item in content if item.get('type') == 'text']
                content = ' '.join(text_parts)
            convs.append({
                'role': role,
                'content': str(content),
                'images': [],
                'audio': []
            })
        return {'conversations': convs, 'messages': messages}
    
    return dataset.map(transform, num_proc=1)


def convert_alpaca_format(dataset):
    """Convert Alpaca instruction/input/output to both conversations and messages."""
    def transform(example):
        instruction = example.get('instruction', '')
        inp = example.get('input', '')
        output = example.get('output', '')
        
        user_content = f"{instruction}\n\n{inp}".strip() if inp else instruction
        
        return {
            'conversations': [
                {'role': 'user', 'content': user_content, 'images': [], 'audio': []},
                {'role': 'assistant', 'content': output, 'images': [], 'audio': []}
            ],
            'messages': [
                {'role': 'user', 'content': user_content},
                {'role': 'assistant', 'content': output}
            ],
            'text': f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
        }
    
    return dataset.map(transform, num_proc=1)


def convert_dpo_pairs(dataset):
    """Convert DPO chosen/rejected pairs."""
    def transform(example):
        prompt = example.get('prompt', example.get('input', example.get('question', '')))
        chosen = example.get('chosen', '')
        rejected = example.get('rejected', '')
        
        return {
            'prompt': str(prompt),
            'chosen': str(chosen),
            'rejected': str(rejected),
            'conversations': [
                {'role': 'user', 'content': str(prompt), 'images': [], 'audio': []},
                {'role': 'assistant', 'content': str(chosen), 'images': [], 'audio': []}
            ],
            'messages': [
                {'role': 'user', 'content': str(prompt)},
                {'role': 'assistant', 'content': str(chosen)}
            ]
        }
    
    return dataset.map(transform, num_proc=1)


def convert_qa_format(dataset):
    """Convert question/answer format."""
    def transform(example):
        question = example.get('question', '')
        answer = example.get('answer', '')
        context = example.get('context', example.get('passage', ''))
        
        user_content = f"{context}\n\n{question}".strip() if context else question
        
        return {
            'conversations': [
                {'role': 'user', 'content': user_content, 'images': [], 'audio': []},
                {'role': 'assistant', 'content': str(answer), 'images': [], 'audio': []}
            ],
            'messages': [
                {'role': 'user', 'content': user_content},
                {'role': 'assistant', 'content': str(answer)}
            ],
            'text': f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"
        }
    
    return dataset.map(transform, num_proc=1)


def convert_code_format(dataset):
    """Convert code instruction/solution datasets."""
    columns = set(dataset.column_names)
    
    problem_col = next((c for c in ['problem', 'instruction', 'prompt', 'question', 'task'] if c in columns), None)
    code_col = next((c for c in ['code', 'solution', 'answer', 'output', 'response'] if c in columns), None)
    
    def transform(example):
        problem = example.get(problem_col, '') if problem_col else ''
        code = example.get(code_col, '') if code_col else ''
        
        return {
            'conversations': [
                {'role': 'user', 'content': str(problem), 'images': [], 'audio': []},
                {'role': 'assistant', 'content': str(code), 'images': [], 'audio': []}
            ],
            'messages': [
                {'role': 'user', 'content': str(problem)},
                {'role': 'assistant', 'content': str(code)}
            ],
            'text': f"<|im_start|>user\n{problem}<|im_end|>\n<|im_start|>assistant\n{code}<|im_end|>"
        }
    
    return dataset.map(transform, num_proc=1)


def convert_prompt_response(dataset):
    """Convert prompt/response or prompt/completion format."""
    columns = set(dataset.column_names)
    
    prompt_col = next((c for c in ['prompt', 'input', 'query'] if c in columns), None)
    response_col = next((c for c in ['response', 'completion', 'output', 'answer', 'reply'] if c in columns), None)
    
    def transform(example):
        prompt = example.get(prompt_col, '') if prompt_col else ''
        response = example.get(response_col, '') if response_col else ''
        
        return {
            'conversations': [
                {'role': 'user', 'content': str(prompt), 'images': [], 'audio': []},
                {'role': 'assistant', 'content': str(response), 'images': [], 'audio': []}
            ],
            'messages': [
                {'role': 'user', 'content': str(prompt)},
                {'role': 'assistant', 'content': str(response)}
            ],
            'text': f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
        }
    
    return dataset.map(transform, num_proc=1)


def attempt_heuristic_conversion(dataset, columns=None):
    """
    Last resort: try to find ANY usable text and create a training format.
    This should rarely be needed if format detection is working.
    """
    if columns is None:
        columns = set(dataset.column_names)
    else:
        columns = set(columns)
    
    print("    Heuristic conversion: attempting to find usable content...", file=sys.stderr)
    
    # Priority order for finding text content
    text_candidates = [
        'text', 'content', 'body', 'message', 'description', 
        'caption', 'sentence', 'paragraph', 'document', 'article',
        'input', 'output', 'prompt', 'response', 'answer', 'question',
        'instruction', 'code', 'solution'
    ]
    
    # Find the best text column
    found_col = None
    for candidate in text_candidates:
        if candidate in columns:
            found_col = candidate
            print(f"    Heuristic: Found '{found_col}' column to use as text", file=sys.stderr)
            break
    
    # Check for image column
    image_col = next((c for c in ['image', 'image_path', 'img', 'screenshot'] if c in columns), None)
    
    def transform(example):
        result = {}
        
        if found_col:
            text = str(example.get(found_col, ''))
        else:
            # Concatenate all string columns
            parts = []
            for col in columns:
                val = example.get(col)
                if isinstance(val, str) and val.strip():
                    parts.append(val)
            text = '\n'.join(parts)
        
        # Create basic format
        result['text'] = text
        
        # Create conversations/messages if we have enough content
        if len(text) > 10:
            result['conversations'] = [
                {'role': 'user', 'content': 'Continue:', 'images': [], 'audio': []},
                {'role': 'assistant', 'content': text, 'images': [], 'audio': []}
            ]
            
            if image_col and example.get(image_col):
                result['messages'] = [
                    {
                        'role': 'user',
                        'content': [
                            {'type': 'image', 'image': example.get(image_col)},
                            {'type': 'text', 'text': 'Describe this image.'}
                        ]
                    },
                    {'role': 'assistant', 'content': text}
                ]
                result['image'] = example.get(image_col)
            else:
                result['messages'] = [
                    {'role': 'user', 'content': 'Continue:'},
                    {'role': 'assistant', 'content': text}
                ]
        
        return result
    
    return dataset.map(transform, num_proc=1)



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
        # IMPORTANT: Skip 'processed_data' subdirectories as they contain Arrow output
        data_files = []
        if os.path.isdir(source_path):
            for root, dirs, files in os.walk(source_path):
                # Skip processed_data directories (they contain Arrow output, not source data)
                if 'processed_data' in root or 'cache-' in root:
                    continue
                # Also remove processed_data from dirs to prevent os.walk descending into it
                dirs[:] = [d for d in dirs if d != 'processed_data' and not d.startswith('cache-')]
                
                for file in files:
                    lower_file = file.lower()
                    # Only process source data formats, NOT Arrow (which is output format)
                    if lower_file.endswith(('.csv', '.json', '.jsonl', '.parquet')):
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
            # ROBUST PARQUET LOADING: Handle heterogeneous schemas across subdirectories
            # Group files by parent directory - each subdirectory may have different schemas
            from collections import defaultdict
            files_by_dir = defaultdict(list)
            for f in data_files:
                parent = os.path.dirname(f) or "root"
                files_by_dir[parent].append(f)
            
            print(f"  Found {len(files_by_dir)} director(y/ies) with parquet files", file=sys.stderr)
            
            all_datasets = []
            load_errors = []
            
            for dir_path, files in files_by_dir.items():
                dir_name = os.path.basename(dir_path) or "root"
                try:
                    print(f"    Loading {len(files)} file(s) from '{dir_name}'...", file=sys.stderr)
                    ds = load_dataset("parquet", data_files=files, split="train")
                    print(f"      ✓ Loaded {len(ds)} rows, columns: {ds.column_names}", file=sys.stderr)
                    all_datasets.append((dir_name, ds))
                except Exception as e:
                    error_msg = str(e)[:100]  # Truncate long error messages
                    print(f"      ✗ Failed: {error_msg}", file=sys.stderr)
                    load_errors.append((dir_name, error_msg))
            
            if not all_datasets:
                # Try loading each file individually as last resort
                print("  Attempting individual file loading as fallback...", file=sys.stderr)
                for f in data_files:
                    try:
                        fname = os.path.basename(f)
                        ds = load_dataset("parquet", data_files=[f], split="train")
                        print(f"    ✓ Loaded {fname}: {len(ds)} rows", file=sys.stderr)
                        all_datasets.append((fname, ds))
                    except Exception as e:
                        continue
            
            if not all_datasets:
                error_summary = "; ".join([f"{d}: {e}" for d, e in load_errors[:3]])
                raise ValueError(f"No valid parquet files could be loaded. Errors: {error_summary}")
            
            # Use the largest dataset (most rows) as the primary
            all_datasets.sort(key=lambda x: len(x[1]), reverse=True)
            chosen_name, dataset = all_datasets[0]
            print(f"  ✓ Using '{chosen_name}' with {len(dataset)} rows (largest of {len(all_datasets)} valid)", file=sys.stderr)
            
        elif first_file.endswith('.csv'):
            dataset = load_dataset("csv", data_files=data_files, split="train")
        elif first_file.endswith('.arrow'):
            dataset = load_dataset("arrow", data_files=data_files, split="train")
        else:
            # Try auto-detect
            dataset = load_dataset(data_files=data_files, split="train")
        
        # Handle DatasetDict (when dataset has splits like 'train', 'test')
        from datasets import DatasetDict
        if isinstance(dataset, DatasetDict):
            # Use 'train' split if available, otherwise first available split
            if 'train' in dataset:
                dataset = dataset['train']
            else:
                first_split = list(dataset.keys())[0]
                print(f"  Using '{first_split}' split from DatasetDict", file=sys.stderr)
                dataset = dataset[first_split]
        
        # Apply intelligent schema detection and normalization
        print("Analyzing and normalizing dataset schema...", file=sys.stderr)
        dataset, format_info = detect_and_normalize_schema(dataset)
        
        if not format_info.get('success', True):
            print(f"⚠️ Warning: Format detection had issues: {format_info.get('error', 'Unknown')}", file=sys.stderr)
        
        print(f"  Detected format: {format_info.get('detected_format', 'unknown')}", file=sys.stderr)
        print(f"  Final columns: {dataset.column_names}", file=sys.stderr)
        print(f"Saving Arrow dataset to {processed_dir}...", file=sys.stderr)
        os.makedirs(processed_dir, exist_ok=True)
        dataset.save_to_disk(processed_dir)
        
        # Save format info for diagnostics
        format_info_path = os.path.join(processed_dir, "format_info.json")
        with open(format_info_path, 'w', encoding='utf-8') as f:
            # Convert sets to lists for JSON serialization
            serializable_info = {}
            for k, v in format_info.items():
                if isinstance(v, set):
                    serializable_info[k] = list(v)
                else:
                    serializable_info[k] = v
            json.dump(serializable_info, f, indent=2)
        
        print(json.dumps({"success": True, "path": processed_dir, "format_info": format_info}))

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