import sys
import subprocess
import os

def run_pip(args, description):
    print(f"--- {description} ---")
    cmd = [sys.executable, "-m", "pip"] + args
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd)
        print("Success.\n")
    except subprocess.CalledProcessError as e:
        print(f"Failed: {e}\n")
        # Don't exit immediately, try to continue if possible, or at least show what happened
        return False
    return True

def fix_install():
    # 1. Upgrade build tools
    run_pip(["install", "--upgrade", "pip", "setuptools", "wheel", "packaging"], "Upgrading build tools")

    # 2. Install xformers (usually matches torch version)
    # For Torch 2.5.1+cu124, we need compatible xformers.
    # If not found, we might skip or try a specific index.
    # Let's try standard install, it usually finds the right one for 2.5.1
    run_pip(["install", "xformers", "--index-url", "https://download.pytorch.org/whl/cu124"], "Installing xformers")

    # 3. Install bitsandbytes for Windows
    # unsloth usually needs bitsandbytes. On Windows, we might need a specific build or just 'bitsandbytes' if it supports Windows now (it does mostly via wrapper or specific versions)
    # Or use the one from unsloth's recommendation
    run_pip(["install", "bitsandbytes"], "Installing bitsandbytes")

    # 4. Install other dependencies
    run_pip(["install", "trl", "peft", "accelerate", "transformers", "datasets", "huggingface_hub"], "Installing core ML libs")

    # 5. Install unsloth no-deps first to avoid build isolation issues if possible
    # We use the git url
    unsloth_url = "git+https://github.com/unslothai/unsloth.git"
    
    print("--- Installing Unsloth (no deps) ---")
    # Try installing without dependencies first since we installed most of them
    if not run_pip(["install", "--no-deps", unsloth_url], "Installing Unsloth (no deps)"):
        print("Retrying Unsloth with dependencies...")
        run_pip(["install", unsloth_url], "Installing Unsloth (with deps)")

    # 6. Install unsloth-zoo
    run_pip(["install", "unsloth-zoo"], "Installing unsloth-zoo")

if __name__ == "__main__":
    fix_install()
