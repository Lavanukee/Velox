# Velox

Velox is a native Windows desktop UI for managing local machine learning artifacts (HuggingFace models, LoRA adapters) and converting them to GGUF formats. Built with TypeScript/React and Tauri, it downloads and sets up a local Python environment at runtime to eliminate WSL or system dependency conflicts.

## Features

*   **Native Windows App:** Runs directly on Windows without Docker, WSL, or complex dependency management.
*   **GGUF Conversion:** One-click conversion for HuggingFace model directories and LoRA adapters.
*   **Inference Playground:** A built-in testing area to chat with and test your tuned models immediately.
*   **Unsloth & TensorBoard:** Fully integrated Unsloth support with embedded TensorBoard metrics to monitor your progress.
*   **Artifact Management:** Automatically organizes and manages your `data/models` and `data/loras` directories.
*   **Isolated Environment:** Automatically downloads a portable Python runtime on first launch, ensuring no conflicts with system-wide Python installations.
*   **Quick Convert:** Dedicated panel for processing external files outside of the managed directories.

## Quick Install

1.  Go to the **[Releases Page](https://github.com/lavanukee/velox/releases)**.
2.  Download the `Velox_Setup_x64.exe` installer.
3.  Run the installer.
    *   *Note: You may see a Windows warning because this is an unsigned app and I dont have 400 dollars to do that. Click "More Info" -> "Run Anyway".*
4.  The first time you run it, the app will download and set up the isolated Python environment automatically.

**Prerequisites:**
*   Windows 10/11
*   NVIDIA GPU



## Development Setup

**Prerequisites:** Node.js (v18+), Rust (via rustup), Python.

1.  **Install dependencies:**
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

feel free to contribute, pull requests are welcome.

## Architecture

*   `src/`: React + TypeScript frontend.
*   `src-tauri/`: Rust backend and Python integration scripts.
*   `data/`: Local storage for models and adapters.
    *   `data/models/`: HuggingFace model storage.
    *   `data/loras/`: LoRA adapter storage.

## Roadmap & Known Issues

Velox is currently in **Alpha**.

**issues:**

*   **General Stability:** Expect bugs and a rough UI experience. The **Inference Playground** is particularly experimental and may have layout or functional inconsistencies.
*   **Dataset & Model Support:** Most but not all Hugging Face artifacts are compatible. Parsing and processing for Unsloth fine-tuning is limited to supported architectures.
*   **GGUF Conversion:** Subject to `llama.cpp` architectural support. the app tries to grab the latest `llama.cpp` binaries on startup to ensure you get compatibility as soon as its out. Unsloth Dynamic GGUFs are on the roadmap but not implemented yet.
*   **Hardware Testing:** Currently only verified on NVIDIA 30-series (3090 Ti). Feedback on 40-series or 50-series performance and compatibility is VERY welcome, please let me know if you have a 40-series or 50-series gpu and if it works or what errors you run into.

**roadmap:**

*   **General Stability:** Expect bugs and a rough UI experience. The **Inference Playground** is particularly experimental and may have layout or functional inconsistencies.
*   **Beta Release:** I hope to have a beta release out by febuary 2026 or so, with a much more stable experience.
*   **Feature Implementation:** I also hope to have the following features implemented:
    *   Unsloth Dynamic GGUFs
    *   More seamless general inference experience and the ability to use this as an LMStudio alternative.
    *   More multimodal support: Audio in and out, and possibly image and video out aswell, 3d models aswell?
    *   A whole lot of UI polish
*   **Hardware Compatibility (possibly):**
    *   AMD/Intel GPU Support
    *   Possibly Macos Support

Though I don't have much access to any of those hardware, I'm always open to suggestions and feedback.


## Acknowledgements

This project stands on the shoulders of giants. Thank you to all who made this possible:

*   **Hugging Face:** For their search API, fine-tuning capabilities, the `transformers` library, and the enormous ecosystem that this application is largely built for.
*   **llama.cpp:** For the powerful inference engine and GGUF conversion scripts.
*   **Tauri:** For providing a lightweight and efficient framework to build this desktop application.
*   **Unsloth:** For making fine-tuning actually fast and efficient.

## License

**MIT License**