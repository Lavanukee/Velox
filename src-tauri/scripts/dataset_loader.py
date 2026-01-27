"""
Dataset loader for Velox Data Collection panel.
Provides paginated reading and metadata extraction for various dataset formats.
"""

import sys
import os
import json
import argparse
import io
from pathlib import Path
from typing import Optional, Dict, Any, List

# Ensure UTF-8 output for Windows and other systems
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')


def load_jsonl(path: str, offset: int = 0, limit: int = 100) -> tuple[List[Dict], int, List[str]]:
    """Load a JSONL file with pagination."""
    rows = []
    total_count = 0
    columns = set()
    
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if isinstance(row, dict):
                    columns.update(row.keys())
                    if offset <= total_count < offset + limit:
                        rows.append(row)
                total_count += 1
            except json.JSONDecodeError:
                continue
    
    return rows, total_count, list(columns)


def load_csv(path: str, offset: int = 0, limit: int = 100) -> tuple[List[Dict], int, List[str]]:
    """Load a CSV file with pagination."""
    import csv
    rows = []
    columns = []
    total_count = 0
    
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        for i, row in enumerate(reader):
            if offset <= total_count < offset + limit:
                rows.append(dict(row))
            total_count += 1
    
    return rows, total_count, columns


def load_parquet(path: str, offset: int = 0, limit: int = 100) -> tuple[List[Dict], int, List[str]]:
    """Load a Parquet file with pagination."""
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(path)
        total_count = table.num_rows
        columns = table.schema.names
        
        # Slice for pagination
        sliced = table.slice(offset, limit)
        rows = sliced.to_pylist()
        
        return rows, total_count, columns
    except ImportError:
        return [], 0, []


def load_huggingface_dataset(dataset_path: str, split: str = "train", offset: int = 0, limit: int = 100) -> tuple[List[Dict], int, List[str]]:
    """Load a HuggingFace datasets library dataset."""
    try:
        from datasets import load_from_disk, load_dataset
        
        # Try loading from disk first
        if os.path.isdir(dataset_path):
            ds = load_from_disk(dataset_path)
        else:
            # Assume it's a dataset name on the hub
            ds = load_dataset(dataset_path, trust_remote_code=True)
        
        # Get the split
        if hasattr(ds, 'keys') and split in ds.keys():
            ds = ds[split]
        elif hasattr(ds, 'keys') and 'train' in ds.keys():
            ds = ds['train']
        
        total_count = len(ds)
        columns = ds.column_names if hasattr(ds, 'column_names') else []
        
        # Slice for pagination
        rows = [ds[i] for i in range(offset, min(offset + limit, total_count))]
        
        return rows, total_count, columns
    except Exception as e:
        print(f"Error loading HuggingFace dataset: {e}", file=sys.stderr)
        return [], 0, []


def detect_format(path: str) -> str:
    """Detect the format of a dataset file or directory."""
    if os.path.isdir(path):
        # Check for HuggingFace dataset format
        if os.path.exists(os.path.join(path, "dataset_info.json")) or \
           os.path.exists(os.path.join(path, "state.json")):
            return "huggingface"
        # Check for Parquet files (recursive)
        parquet_files = list(Path(path).rglob("*.parquet"))
        if parquet_files:
            return "parquet"
        # Check for JSONL files
        jsonl_files = list(Path(path).glob("*.jsonl")) + list(Path(path).glob("*.json"))
        if jsonl_files:
            return "jsonl"
        return "unknown"
    
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.jsonl', '.json']:
        return 'jsonl'
    elif ext == '.csv':
        return 'csv'
    elif ext == '.parquet':
        return 'parquet'
    return 'unknown'


def make_json_serializable(obj):
    """Recursively convert objects to JSON-serializable types."""
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, bytes):
        try:
            import base64
            return f"data:image/png;base64,{base64.b64encode(obj).decode('utf-8')}"
        except:
            return f"<binary data: {len(obj)} bytes>"
    if hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    if hasattr(obj, 'tolist'):  # numpy array
        return obj.tolist()
    return str(obj)

def load_dataset_preview(path: str, offset: int = 0, limit: int = 100, split: str = "train") -> Dict[str, Any]:
    """
    Main entry point for loading dataset preview.
    Returns a dict with rows, totalCount, columns, and format.
    """
    format_type = detect_format(path)
    
    try:
        if format_type == 'jsonl':
            if os.path.isdir(path):
                # Find first jsonl file in directory
                jsonl_files = list(Path(path).glob("*.jsonl")) + list(Path(path).glob("*.json"))
                if jsonl_files:
                    path = str(jsonl_files[0])
                else:
                    return {"rows": [], "totalCount": 0, "columns": [], "format": "unknown", "error": "No JSONL files found"}
            rows, total, columns = load_jsonl(path, offset, limit)
        elif format_type == 'csv':
            rows, total, columns = load_csv(path, offset, limit)
        elif format_type == 'parquet':
            if os.path.isdir(path):
                parquet_files = list(Path(path).rglob("*.parquet"))
                if parquet_files:
                    path = str(parquet_files[0])
            rows, total, columns = load_parquet(path, offset, limit)
        elif format_type == 'huggingface':
            rows, total, columns = load_huggingface_dataset(path, split, offset, limit)
        else:
            return {"rows": [], "totalCount": 0, "columns": [], "format": format_type, "error": f"Unknown format for: {path}"}
        
        return {
            "rows": make_json_serializable(rows),
            "totalCount": total,
            "columns": columns,

        

            "format": format_type
        }
    except Exception as e:
        import traceback
        return {
            "rows": [],
            "totalCount": 0,
            "columns": [],
            "format": format_type,
            "error": f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        }


def save_dataset(path: str, rows: List[Dict], format_type: str = "auto") -> Dict[str, Any]:
    """
    Save a list of rows to a dataset file.
    Supports JSONL, JSON, CSV, Parquet.
    """
    try:
        # Detect format if auto
        if format_type == "auto":
            format_type = detect_format(path)
            if format_type == "unknown":
                # Default to jsonl if new file or unknown
                ext = os.path.splitext(path)[1].lower()
                if ext == ".csv": format_type = "csv"
                elif ext == ".parquet": format_type = "parquet"
                elif ext == ".json": format_type = "json"
                else: format_type = "jsonl"

        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        if format_type == "jsonl":
            with open(path, 'w', encoding='utf-8') as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')
        
        elif format_type == "json":
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(rows, f, indent=2, ensure_ascii=False)
                
        elif format_type == "csv":
            import csv
            if not rows:
                with open(path, 'w', encoding='utf-8') as f: pass
            else:
                columns = list(rows[0].keys())
                # Union of all keys if sparse
                all_keys = set().union(*(r.keys() for r in rows))
                columns = sorted(list(all_keys))
                
                with open(path, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=columns)
                    writer.writeheader()
                    writer.writerows(rows)
                    
        elif format_type == "parquet":
            import pyarrow as pa
            import pyarrow.parquet as pq
            if not rows:
                raise ValueError("Cannot save empty parquet file without schema")
            table = pa.Table.from_pylist(rows)
            pq.write_table(table, path)
            
        else:
            return {"success": False, "error": f"Unsupported save format: {format_type}"}
            
        return {"success": True, "count": len(rows)}
        
    except Exception as e:
        import traceback
        return {"success": False, "error": f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"}


def apply_edits(path: str, edits: List[Dict]) -> Dict[str, Any]:
    """
    Apply edits to a dataset file.
    Edits is a list of dicts: {"rowIndex": int, "data": dict}
    "data" is the new row content. If None, row is deleted.
    """
    import shutil
    import tempfile
    
    try:
        format_type = detect_format(path)
        if format_type not in ["jsonl", "csv"]:
            return {"success": False, "error": f"Editing not supported for format: {format_type}. Convert to JSONL/CSV first."}
            
        # Create map of index -> new_data
        edit_map = {e["rowIndex"]: e.get("data") for e in edits}
        
        # Create temp file
        fd, temp_path = tempfile.mkstemp(dict=True)
        os.close(fd)
        
        count = 0
        written_count = 0
        
        if format_type == "jsonl":
            with open(path, 'r', encoding='utf-8') as fin, \
                 open(temp_path, 'w', encoding='utf-8') as fout:
                for i, line in enumerate(fin):
                    if i in edit_map:
                        new_data = edit_map[i]
                        if new_data is not None:
                            fout.write(json.dumps(new_data, ensure_ascii=False) + '\n')
                            written_count += 1
                    else:
                        fout.write(line)
                        written_count += 1
                    count += 1
                    
        elif format_type == "csv":
            import csv
            with open(path, 'r', encoding='utf-8', newline='') as fin, \
                 open(temp_path, 'w', encoding='utf-8', newline='') as fout:
                reader = csv.DictReader(fin)
                if not reader.fieldnames:
                    return {"success": False, "error": "CSV has no headers"}
                    
                # We need to handle potential new columns from edits
                # For now, restrict edits to existing columns + any found in edits
                # detailed handling omitted for brevity, assuming schema consistency or permissive writer
                
                # First pass: identify all columns if we want to support adding columns?
                # For simplicity, let's stick to original headers for now.
                fieldnames = reader.fieldnames
                writer = csv.DictWriter(fout, fieldnames=fieldnames)
                writer.writeheader()
                
                for i, row in enumerate(reader):
                    if i in edit_map:
                        new_data = edit_map[i]
                        if new_data is not None:
                            # ensure new data matches schema or ignore extras?
                            # For safety in this simpler impl, filter keys
                            filtered = {k: v for k, v in new_data.items() if k in fieldnames}
                            writer.writerow(filtered)
                            written_count += 1
                    else:
                        writer.writerow(row)
                        written_count += 1
                    count += 1

        # Move temp file to original
        shutil.move(temp_path, path)
        return {"success": True, "count": written_count}
        
    except Exception as e:
        import traceback
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        return {"success": False, "error": f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"}


def bulk_edit(path: str, operation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply bulk edits to a dataset file (Find & Replace).
    """
    import shutil
    import tempfile
    import re
    
    try:
        format_type = detect_format(path)
        if format_type not in ["jsonl", "csv"]:
            return {"success": False, "error": f"Bulk editing not supported for format: {format_type}. Convert to JSONL first."}
            
        op_type = operation.get("type")
        if op_type == "replace":
            column = operation.get("column")
            find_pattern = operation.get("find")
            replace_with = operation.get("replace")
            is_regex = operation.get("is_regex", False)
            
            if not column or find_pattern is None:
                return {"success": False, "error": "Missing column or find pattern"}
                
            # Compile regex if needed
            if is_regex:
                try:
                    pattern = re.compile(find_pattern)
                except re.error as e:
                    return {"success": False, "error": f"Invalid regex: {e}"}
            
            # Create temp file
            fd, temp_path = tempfile.mkstemp(dict=True)
            os.close(fd)
            
            count = 0
            modified_count = 0
            
            if format_type == "jsonl":
                with open(path, 'r', encoding='utf-8') as fin, \
                     open(temp_path, 'w', encoding='utf-8') as fout:
                    for line in fin:
                        try:
                            row = json.loads(line)
                            original_val = row.get(column)
                            if isinstance(original_val, str):
                                if is_regex:
                                    new_val, n = pattern.subn(replace_with, original_val)
                                    if n > 0:
                                        row[column] = new_val
                                        modified_count += 1
                                else:
                                    if find_pattern in original_val:
                                        row[column] = original_val.replace(find_pattern, replace_with)
                                        modified_count += 1
                                        
                            fout.write(json.dumps(row, ensure_ascii=False) + '\n')
                        except json.JSONDecodeError:
                            fout.write(line) # Keep invalid lines
                        count += 1
                        
            elif format_type == "csv":
                import csv
                with open(path, 'r', encoding='utf-8', newline='') as fin, \
                     open(temp_path, 'w', encoding='utf-8', newline='') as fout:
                    reader = csv.DictReader(fin)
                    writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
                    writer.writeheader()
                    
                    for row in reader:
                        original_val = row.get(column)
                        if isinstance(original_val, str):
                            if is_regex:
                                new_val, n = pattern.subn(replace_with, original_val)
                                if n > 0:
                                    row[column] = new_val
                                    modified_count += 1
                            else:
                                if find_pattern in original_val:
                                    row[column] = original_val.replace(find_pattern, replace_with)
                                    modified_count += 1
                        writer.writerow(row)
                        count += 1

            shutil.move(temp_path, path)
            return {"success": True, "count": modified_count, "total_scanned": count}
            
        else:
            return {"success": False, "error": f"Unknown operation type: {op_type}"}

    except Exception as e:
        import traceback
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        return {"success": False, "error": f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset loader for Velox")
    subparsers = parser.add_subparsers(dest="command")

    # Load command
    load_parser = subparsers.add_parser("load")
    load_parser.add_argument("path", type=str)
    load_parser.add_argument("--offset", type=int, default=0)
    load_parser.add_argument("--limit", type=int, default=100)
    load_parser.add_argument("--split", type=str, default="train")
    
    # Save command
    save_parser = subparsers.add_parser("save")
    save_parser.add_argument("path", type=str)
    save_parser.add_argument("--data", type=str, help="JSON string of rows")
    
    # Edit command
    edit_parser = subparsers.add_parser("edit")
    edit_parser.add_argument("path", type=str)
    edit_parser.add_argument("--edits", type=str, help="JSON string of edits: [{'rowIndex': 0, 'data': {...}}]")
    
    # Bulk Edit command
    bulk_edit_parser = subparsers.add_parser("bulk_edit")
    bulk_edit_parser.add_argument("path", type=str)
    bulk_edit_parser.add_argument("--operation", type=str, help="JSON string of operation config")
    
    args = parser.parse_args()
    
    if args.command == "load" or not args.command:
        # Default to load for backward compatibility if just path is passed (need to handle argparse quirk)
        # However, since we defined subparsers, the structure changes slightly.
        # Let's support the old direct usage by checking if sys.argv[1] is not a command
        pass 

    # Re-evaluating default behavior to maintain compatibility
    if len(sys.argv) > 1 and sys.argv[1] not in ["load", "save"]:
        # Legacy mode: python dataset_loader.py path --offset ...
        # Manually parse or just default to load logic
        dataset_path = sys.argv[1]
        offset = 0
        limit = 100
        split = "train"
        
        # Simple manual parse for legacy args
        if "--offset" in sys.argv:
            offset = int(sys.argv[sys.argv.index("--offset") + 1])
        if "--limit" in sys.argv:
            limit = int(sys.argv[sys.argv.index("--limit") + 1])
        if "--split" in sys.argv:
            split = sys.argv[sys.argv.index("--split") + 1]
            
        result = load_dataset_preview(dataset_path, offset, limit, split)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    elif args.command == "load":
        result = load_dataset_preview(args.path, args.offset, args.limit, args.split)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    elif args.command == "save":
        try:
            rows = json.loads(args.data)
            result = save_dataset(args.path, rows)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        except Exception as e:
            print(json.dumps({"success": False, "error": str(e)}))
            
    elif args.command == "edit":
        try:
            edits = json.loads(args.edits)
            result = apply_edits(args.path, edits)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        except Exception as e:
            print(json.dumps({"success": False, "error": str(e)}))

    elif args.command == "bulk_edit":
        try:
            # args.edits here is a JSON string defining the operation
            # e.g. {"type": "replace", "column": "instruction", "find": "foo", "replace": "bar", "is_regex": false}
            operation = json.loads(args.operation)
            result = bulk_edit(args.path, operation)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        except Exception as e:
            print(json.dumps({"success": False, "error": str(e)}))
