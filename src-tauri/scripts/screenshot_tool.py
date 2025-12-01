import sys
import time
from pathlib import Path
from datetime import datetime
import argparse
import mss
import mss.tools
import logging

# Configure logging to sys.stderr, which Tauri can capture
logging.basicConfig(level=logging.INFO, stream=sys.stderr, format='%(levelname)s: %(message)s')

def take_screenshot(output_dir_relative_to_src_tauri: str, filename_prefix: str = "screenshot") -> Path:
    """Takes a full-screen screenshot, saves it, and returns the path relative to src-tauri."""
    
    # Resolve the absolute path of the directory where the screenshot will be saved
    # Path.cwd() in this context is expected to be the src-tauri directory (set by Rust's current_dir)
    abs_output_dir = Path.cwd() / output_dir_relative_to_src_tauri
    
    logging.info(f"Ensuring output directory exists: {abs_output_dir}")
    abs_output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info("Taking screenshot in 3 seconds. Please switch to the desired window.")
    time.sleep(3)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.png"
    abs_filepath = abs_output_dir / filename
    
    with mss.mss() as sct:
        monitor_number = 1
        # ... (Monitor selection logic remains the same) ...
        if monitor_number >= len(sct.monitors):
             monitor_number = 0
        mon = sct.monitors[monitor_number]
        
        sct_img = sct.grab(mon)
        
        # Save to the absolute path
        mss.tools.to_png(sct_img.rgb, sct_img.size, output=str(abs_filepath))

    # --- CRITICAL FIX ---
    # Construct the path to return. This must be relative to src-tauri and use forward slashes.
    # E.g., "data/raw_screenshots/my_screenshot.png"
    path_to_return_for_frontend = str(Path(output_dir_relative_to_src_tauri) / filename).replace("\\", "/")

    # Print to stderr so Rust can capture it
    print(f"INFO: Screenshot saved to {path_to_return_for_frontend}", file=sys.stderr)
    logging.info(f"Screenshot path sent to stderr: {path_to_return_for_frontend}")
    
    return abs_filepath # This return is for internal Python use, Rust only cares about stderr

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Take a full-screen screenshot and save it to a specified directory."
    )
    # The default is still used if run directly, but Rust passes an explicit path.
    parser.add_argument("--output_dir", type=str, default="data/raw_screenshots",
                        help="Output directory relative to src-tauri.")
    parser.add_argument("--filename_prefix", type=str, default="screenshot")
    args = parser.parse_args()
    
    try:
        # Pass the output_dir directly, it's expected to be relative to src-tauri
        take_screenshot(args.output_dir, args.filename_prefix)
    except Exception as e:
        logging.error(f"An unexpected error occurred during screenshot: {e}")
        sys.exit(1)