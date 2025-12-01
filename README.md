# Velox

Velox is a native Windows desktop UI for managing all your local machine learning artifacts (HuggingFace models, LoRA adapters) and converting them to GGUF formats. It is built with TypeScript/React (frontend) and Tauri (backend), bundling a local Python environment so you don't have to deal with WSL or system dependency conflicts or even worry about keeping your files organzied yourself.

## Quick Install

1.  Go to the **[Releases Page](https://github.com/lavanukee/velox/releases)**.
2.  Download the `Velox_Setup_x64.exe` installer.
3.  Run the installer.
    *   *Note: You may see a Windows warning because this is an unsigned app and I dont have 400 dollars to do that. Click "More Info" -> "Run Anyway".*
4.  the first time you run it, the app should automatically set up the isolated Python environment.

**Prerequisites:**
*   Windows 10/11
*   NVIDIA GPU

## What this UI should do
*   **Native Setup:** Runs on Windows without Docker or WSL or any other dependency hell issues.
*   **Conversion:** One-click conversion of HuggingFace model directories and LoRA adapters to GGUF.
*   **Organization:** Manages your `data/models` and `data/loras` folders so your directory doesn't become a mess.
*   **Utilities:** Quick convert panel for processing external files.

## Development Setup

**Prerequisites:** Node.js (v18+), npm/yarn, Rust (via rustup), Python.

1.  **Install JS dependencies:**
    ```powershell
    npm install
    ```

2.  **Run in Dev Mode:**
    ```powershell
    npm run tauri dev
    ```

3.  **Build for Production:**
    ```powershell
    npm run tauri build
    ```

## Architecture

*   `src/`: React + TypeScript frontend.
*   `src-tauri/`: Rust backend and python scripts.
*   `python-x86_64-windows/`: Portable Python runtime (downloaded/installed at runtime).
*   `data/`: Where all your dataset/lora/model files live.
    *   Models go in `data/models/`
    *   Adapters go in `data/loras/`

## Roadmap & Known Issues

This is an alpha release.

*   **Unsloth Integration:** Full fine-tuning integration is currently a work in progress (getting complex dependencies packaged portably is taking longer than expected).
*   **Drag-and-Drop:** The visual drag-and-drop area in the Quick Convert panel is currently broken. Please use the "Browse" button instead.
*   **Quick Browse:** Currently configured to select directories only, not individual files.
*   **UI Glitches:** Sometimes `mmproj` is unable to be selected in the inference window. The progress bar may not show up reliably.


## Acknowledgements

This project stands on the shoulders of giants. Thank you to all who made this possible:

*   **Hugging Face:** For their search API, fine-tuning capabilities, the `transformers` library, and the enormous ecosystem that this application is built to manage.
*   **llama.cpp:** For the powerful inference engine and GGUF conversion scripts.
*   **Tauri:** For providing a lightweight and efficient framework to build this desktop application.


## License

**MIT License**
(I don't really plan to sue anyone over this, but the text is there for legal safety.)