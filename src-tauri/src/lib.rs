use log::{debug, error};
use reqwest;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::future;
use std::io::Cursor;
use std::path::PathBuf;
use std::process::Command as StdCommand;
use std::process::Stdio;
use std::sync::Arc;
use tauri::{AppHandle, Emitter, Manager, State, Window};
use tokio::fs;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command as TokioCommand};
use tokio::sync::Mutex;

const MODELS_CONFIG_FILE: &str = "../models_config.json";

// --- NEW: Hardware & Download Logic ---

#[derive(Debug, Serialize, Clone)]
enum ComputeBackend {
    Cuda,
    #[allow(dead_code)]
    Metal,
    Cpu,
}

fn detect_backend() -> ComputeBackend {
    #[cfg(target_os = "macos")]
    {
        // Check for Apple Silicon
        let output = StdCommand::new("sysctl")
            .arg("-n")
            .arg("machdep.cpu.brand_string")
            .output()
            .ok();

        if let Some(out) = output {
            let s = String::from_utf8_lossy(&out.stdout).to_lowercase();
            if s.contains("apple") {
                return ComputeBackend::Metal;
            }
        }
        return ComputeBackend::Cpu; // Intel Mac
    }

    #[cfg(target_os = "windows")]
    {
        // 1. PRIORITY CHECK: Try nvidia-smi (The gold standard for checking CUDA availability)
        // This checks if the driver utility is actually in the PATH and executable.
        debug!("Detecting Backend: Attempting nvidia-smi check...");
        match StdCommand::new("nvidia-smi").arg("-L").output() {
            Ok(output) => {
                let s = String::from_utf8_lossy(&output.stdout).to_lowercase();
                debug!("nvidia-smi output: {}", s);
                if s.contains("gpu") || s.contains("nvidia") {
                    debug!("Backend Decision: CUDA (via nvidia-smi)");
                    return ComputeBackend::Cuda;
                }
            }
            Err(e) => debug!("nvidia-smi check failed/not found: {}", e),
        }

        // 2. SECONDARY CHECK: Check Environment Variables
        // Sometimes binary isn't in PATH, but CUDA_PATH is set by the installer.
        if std::env::var("CUDA_PATH").is_ok() {
            debug!("Backend Decision: CUDA (via CUDA_PATH env var)");
            return ComputeBackend::Cuda;
        }

        // 3. FALLBACK CHECK: WMIC
        debug!("Detecting Backend: Attempting WMIC check...");
        let output = StdCommand::new("wmic")
            .args(&["path", "win32_videocontroller", "get", "name"])
            .output()
            .ok();

        if let Some(out) = output {
            let s = String::from_utf8_lossy(&out.stdout).to_lowercase();
            debug!("WMIC output: {}", s);
            if s.contains("nvidia") {
                debug!("Backend Decision: CUDA (via WMIC)");
                return ComputeBackend::Cuda;
            }
        }

        // 4. Default to CPU
        debug!("Backend Decision: CPU (No NVIDIA indicators found)");
        return ComputeBackend::Cpu;
    }

    #[cfg(target_os = "linux")]
    {
        // Check 1: nvidia-smi
        if let Ok(output) = StdCommand::new("nvidia-smi").arg("-L").output() {
            let s = String::from_utf8_lossy(&output.stdout).to_lowercase();
            if s.contains("nvidia") {
                return ComputeBackend::Cuda;
            }
        }

        // Check 2: lspci
        let output = StdCommand::new("lspci").output().ok();
        if let Some(out) = output {
            let s = String::from_utf8_lossy(&out.stdout).to_lowercase();
            if s.contains("nvidia") {
                return ComputeBackend::Cuda;
            }
        }
        return ComputeBackend::Cpu;
    }
}

// Helper to get the path where we store the binary
fn get_binaries_dir(app_handle: &AppHandle) -> PathBuf {
    app_handle
        .path()
        .app_data_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join("binaries")
}

#[tauri::command]
async fn check_llama_binary_command(app_handle: AppHandle) -> Result<bool, String> {
    let bin_dir = get_binaries_dir(&app_handle);
    debug!("Binaries directory: {:?}", bin_dir);
    let name = if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    };
    let path = bin_dir.join(name);
    debug!("Checking for llama binary at: {:?}", path);
    let exists = path.exists();
    debug!("Llama binary exists: {}", exists);
    Ok(exists)
}

#[tauri::command]
async fn download_llama_binary_command(
    window: Window,
    app_handle: AppHandle,
) -> Result<String, String> {
    let backend = detect_backend();
    debug!("Detected backend: {:?}", backend);

    // 1. Determine Keywords
    let (main_keywords, dep_keywords): (Vec<&str>, Vec<&str>) =
        match (std::env::consts::OS, &backend) {
            ("windows", ComputeBackend::Cuda) => (
                vec!["bin-win-cuda-12", "x64"],
                vec!["cudart", "win", "cu12"],
            ),
            ("windows", ComputeBackend::Cpu) => (vec!["bin-win-x64"], vec![]),
            ("macos", ComputeBackend::Metal) => (vec!["bin-macos-arm64"], vec![]),
            ("macos", ComputeBackend::Cpu) => (vec!["bin-macos-x64"], vec![]),
            ("linux", ComputeBackend::Cuda) => (vec!["bin-ubuntu-x64-cuda"], vec![]),
            ("linux", _) => (vec!["bin-ubuntu-x64"], vec![]),
            _ => (vec!["bin-win-x64"], vec![]),
        };

    let log_msg = format!(
        "Searching for engine: {:?} (Deps: {:?})",
        main_keywords, dep_keywords
    );
    window.emit("log", &log_msg).unwrap();
    debug!("{}", log_msg);

    // 2. Fetch Release Info
    let client = reqwest::Client::new();
    let release_url = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest";

    let resp = client
        .get(release_url)
        .header("User-Agent", "tauri-app")
        .send()
        .await
        .map_err(|e| format!("GitHub API error: {}", e))?
        .error_for_status()
        .map_err(|e| format!("GitHub API status error: {}", e))?;

    let resp_json: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| format!("JSON parse error: {}", e))?;
    let assets = resp_json["assets"]
        .as_array()
        .ok_or("No assets found in release")?;

    // 3. Helper to find url (Updated to support Exclusion)
    let find_asset_url = |keywords: &Vec<&str>, exclude: Option<&str>| -> Option<String> {
        assets
            .iter()
            .find(|a| {
                let name = a["name"].as_str().unwrap_or("");
                let matches_keywords =
                    name.ends_with(".zip") && keywords.iter().all(|k| name.contains(k));

                // If exclude is provided, ensure name DOES NOT contain it
                let not_excluded = match exclude {
                    Some(ex) => !name.contains(ex),
                    None => true,
                };

                matches_keywords && not_excluded
            })
            .and_then(|a| a["browser_download_url"].as_str().map(|s| s.to_string()))
    };

    // 4. Resolve Main URL
    // CRITICAL FIX: If we are on Windows CUDA, we explicitly exclude "cudart" from the MAIN search
    // so we don't accidentally grab the dependency zip as the engine zip.
    let exclude_filter = if cfg!(windows) && matches!(backend, ComputeBackend::Cuda) {
        Some("cudart")
    } else {
        None
    };

    let main_url = match find_asset_url(&main_keywords, exclude_filter) {
        Some(url) => url,
        None => {
            // Fallback for Windows CUDA -> CPU
            if main_keywords.contains(&"bin-win-cuda-12") {
                window
                    .emit("log", "CUDA binary not found, falling back to CPU...")
                    .unwrap();
                let cpu_keywords = vec!["bin-win-x64"];
                match find_asset_url(&cpu_keywords, None) {
                    Some(url) => url,
                    None => return Err("Failed to find CUDA or CPU binaries.".into()),
                }
            } else {
                return Err(format!("Asset not found for keywords: {:?}", main_keywords));
            }
        }
    };

    let dep_url = if !dep_keywords.is_empty() {
        find_asset_url(&dep_keywords, None)
    } else {
        None
    };

    // 5. Download and Extract
    let bin_dir = get_binaries_dir(&app_handle);
    if !bin_dir.exists() {
        fs::create_dir_all(&bin_dir)
            .await
            .map_err(|e| e.to_string())?;
    }

    let downloads = if let Some(dep) = dep_url {
        // Note: dep_url is the cudart zip, main_url is the actual engine zip
        vec![("Engine", main_url), ("Drivers", dep)]
    } else {
        vec![("Engine", main_url)]
    };

    for (label, url) in downloads {
        window
            .emit("log", format!("Downloading {}...", label))
            .unwrap();
        debug!("Downloading {} from {}", label, url);

        let response = client
            .get(&url)
            .send()
            .await
            .map_err(|e| format!("Failed to download {}: {}", label, e))?
            .error_for_status()
            .map_err(|e| format!("Failed to download {} (HTTP Error): {}", label, e))?;

        let content = response
            .bytes()
            .await
            .map_err(|e| format!("Failed to read bytes: {}", e))?;

        let reader = Cursor::new(content);
        let mut archive = zip::ZipArchive::new(reader)
            .map_err(|e| format!("Failed to unzip {}: {}", label, e))?;

        for i in 0..archive.len() {
            let mut file = archive.by_index(i).unwrap();

            let outpath = match file.enclosed_name() {
                Some(path) => path.to_owned(),
                None => continue,
            };

            let file_name_raw = outpath.file_name();
            if file_name_raw.is_none() {
                continue;
            }
            let file_name = file_name_raw.unwrap().to_string_lossy().to_string();
            let name_lower = file_name.to_lowercase();

            let is_exe = name_lower == "llama-server.exe"
                || name_lower == "server.exe"
                || name_lower.contains("llama-server");
            let is_dll = name_lower.ends_with(".dll");

            if is_exe || (cfg!(windows) && is_dll) {
                let dest_path = bin_dir.join(&file_name);

                debug!("Extracting {:?} to {:?}", outpath, dest_path);
                // Reduce log noise slightly, only log main events
                if is_exe {
                    window
                        .emit("log", format!("Extracting Binary: {}", file_name))
                        .unwrap();
                }

                let mut outfile = std::fs::File::create(&dest_path)
                    .map_err(|e| format!("File creation failed: {}", e))?;
                std::io::copy(&mut file, &mut outfile)
                    .map_err(|e| format!("File write failed: {}", e))?;

                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    if is_exe {
                        let mut perms = std::fs::metadata(&dest_path).unwrap().permissions();
                        perms.set_mode(0o755);
                        std::fs::set_permissions(&dest_path, perms).unwrap();
                    }
                }
            }
        }
    }

    // Final Verification
    let server_path = bin_dir.join(if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    });
    if server_path.exists() {
        window
            .emit("log", "Engine installed successfully!".to_string())
            .unwrap();
        Ok("Installation complete.".into())
    } else {
        Err("Installation finished, but llama-server binary is missing.".into())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScreenshotData {
    pub path: String,
}

// --- Helper: Resolve Python Path and Working Directory ---
fn get_python_command() -> Result<(String, PathBuf), String> {
    let src_tauri_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let project_root_dir = src_tauri_dir
        .parent()
        .expect("Failed to get project root directory")
        .to_path_buf();

    // 1. Try to find the bundled portable python
    let portable_python_path = project_root_dir.join("python/python.exe");
    if portable_python_path.exists() {
        let mut python_exe = portable_python_path
            .canonicalize()
            .unwrap_or(portable_python_path.clone())
            .to_string_lossy()
            .to_string();
        if cfg!(windows) && python_exe.starts_with(r"\\?\") {
            python_exe = python_exe[4..].to_string();
        }
        debug!("Using portable Python executable: {}", python_exe);
        return Ok((python_exe, project_root_dir));
    }

    // Error: No portable python found
    error!("Portable Python not found at {:?}", portable_python_path);
    Err(format!(
        "Portable Python not found. Expected at: {}",
        portable_python_path.display()
    ))
}

// --- Helper: Resolve Script Path ---
// Script paths are relative to the project root when get_python_command returns project_root_dir as work_dir
fn get_script_path(script_name: &str) -> String {
    let script_path_in_project_root = PathBuf::from("src-tauri/scripts").join(script_name);
    script_path_in_project_root.to_string_lossy().to_string()
}

// --- Debugging for get_python_command ---
#[allow(dead_code)] // This command is for debugging and might not be directly invoked from the frontend.
#[tauri::command]
async fn debug_python_path_command() -> Result<String, String> {
    let (python_exe, work_dir) = get_python_command()?;
    Ok(format!(
        "Python Exe: {}, Working Dir: {:?}",
        python_exe, work_dir
    ))
}

// --- ModelConfig Struct ---
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelConfig {
    pub id: String,
    pub name: String,
    pub base_model_id: String,
    pub download_output_folder: String,
    pub yolo_model_path: String,
    pub raw_screenshot_dir: String,
    pub annotated_data_dir: String,
    pub cropped_image_dir: String,
    pub templates_dir: String,
    pub data_collection_output_dir: String,
    pub dataset_file: String,
    pub screenshots_dir: String,
    pub llm_server: String,
    pub llm_timeout: u64,
    pub match_threshold: f64,
    pub monitor_index: u64,
    pub screenshot_delay: f64,
    pub use_multiscale: bool,
    pub scale_factors: String,
    pub num_workers: u64,
    pub skip_image_save: bool,
    pub batch_write_dataset: bool,
    pub dataset_write_buffer: u64,
    pub crop_padding: u64,
    pub cache_descriptions: bool,
    pub description_file: String,
    pub num_iterations: u64,
    pub processing_dataset_dir: String,
    pub deduplication_backup_path: String,
    pub processing_output_dir: String,
    pub processing_model_name: String,
    pub eval_split: f64,
    pub processing_num_workers: u64,
    pub batch_size: u64,
    pub train_data_dir: String,
    pub eval_data_dir: String,
    pub training_output_dir: String,
    pub training_model_name: String,
    pub max_seq_length: u64,
    pub lora_r: u64,
    pub lora_alpha: u64,
    pub lora_dropout: f64,
    pub num_train_epochs: u64,
    pub per_device_train_batch_size: u64,
    pub per_device_eval_batch_size: u64,
    pub gradient_accumulation_steps: u64,
    pub eval_accumulation_steps: u64,
    pub learning_rate: f64,
    pub warmup_ratio: f64,
    pub optim: String,
    pub logging_steps: u64,
    pub eval_steps: u64,
    pub save_steps: u64,
    pub save_total_limit: u64,
    pub gradient_checkpointing: bool,
    pub lr_scheduler_type: String,
    pub early_stopping_patience: u64,
    pub llama_model_path: String,
    pub mmproj_path: String,
    pub lora_path: String,
    pub server_host: String,
    pub server_port: u16,
    pub n_gpu_layers: u64,
    pub ctx_size: u64,
    pub batch_size_inference: u64,
    pub ubatch_size: u64,
    pub temperature: f64,
    pub no_mmap: bool,
    pub flash_attn: bool,
    pub dataset_id: String,
    pub dataset_download_output_folder: String,
}

pub struct PythonProcessState {
    pub data_collector_child: Mutex<Option<Child>>,
    pub llama_server_child: Mutex<Option<Child>>,
    pub tensorboard_child: Mutex<Option<Child>>,
    pub tensorboard_port: Mutex<Option<u16>>,
}

pub struct ModelsState {
    pub configs: Mutex<HashMap<String, ModelConfig>>,
    pub active_downloads: Mutex<HashMap<String, u32>>,
}

impl ModelsState {
    async fn load_from_disk() -> Result<HashMap<String, ModelConfig>, String> {
        let config_path = PathBuf::from(MODELS_CONFIG_FILE);
        debug!("Attempting to load model config from: {:?}", config_path);
        if !config_path.exists() {
            debug!("Model config file not found at: {:?}", config_path);
            return Ok(HashMap::new());
        }
        let contents = fs::read_to_string(&config_path).await.map_err(|e| {
            error!("Failed to read model config file {:?}: {}", config_path, e);
            e.to_string()
        })?;
        let models: HashMap<String, ModelConfig> =
            serde_json::from_str(&contents).map_err(|e| {
                error!("Failed to parse model config from {:?}: {}", config_path, e);
                e.to_string()
            })?;
        debug!(
            "Successfully loaded {} model configs from {:?}.",
            models.len(),
            config_path
        );
        Ok(models)
    }
    async fn save_to_disk(configs: &HashMap<String, ModelConfig>) -> Result<(), String> {
        let config_path = PathBuf::from(MODELS_CONFIG_FILE);
        let serialized = serde_json::to_string_pretty(configs).map_err(|e| e.to_string())?;
        fs::write(&config_path, serialized)
            .await
            .map_err(|e| e.to_string())?;
        Ok(())
    }
}

impl Default for PythonProcessState {
    fn default() -> Self {
        PythonProcessState {
            data_collector_child: Mutex::new(None),
            llama_server_child: Mutex::new(None),
            tensorboard_child: Mutex::new(None),
            tensorboard_port: Mutex::new(None),
        }
    }
}

// Struct to hold the chat context (history) for the Llama server
#[derive(Default)]
pub struct LlamaChatContext {
    pub chat_history: Mutex<Vec<HashMap<String, String>>>,
}

// --- COMMANDS ---

#[tauri::command]
async fn save_model_config_command(
    models_state: State<'_, Arc<ModelsState>>,
    model_config: ModelConfig,
) -> Result<String, String> {
    debug!(
        "Attempting to save model config for ID: {}",
        model_config.id
    );
    let mut configs = models_state.configs.lock().await;

    // Resolve project root for path manipulation
    let src_tauri_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let project_root_dir = src_tauri_dir
        .parent()
        .expect("Failed to get project root directory")
        .to_path_buf();
    let lora_base_dir = project_root_dir.join("data").join("loras");

    // Process lora_path: convert absolute paths to relative if they are within data/loras, otherwise keep as is or empty.
    let mut processed_model_config = model_config.clone();
    if !processed_model_config.lora_path.is_empty() {
        let current_lora_path = PathBuf::from(&processed_model_config.lora_path);
        if current_lora_path.is_absolute() {
            if let Ok(relative_path) = current_lora_path.strip_prefix(&lora_base_dir) {
                processed_model_config.lora_path = relative_path.to_string_lossy().to_string();
                debug!(
                    "Converted absolute lora_path to relative: {}",
                    processed_model_config.lora_path
                );
            } else {
                // If it's an absolute path but not under data/loras, we log an error
                // and keep it as is, expecting the server start to handle its non-existence.
                // Or, clear it to prevent issues if it's not a known location.
                // For now, let's clear it to prevent trying to load invalid absolute paths.
                error!("Absolute lora_path '{}' is not within the expected data/loras directory. Clearing path to prevent issues.", processed_model_config.lora_path);
                processed_model_config.lora_path = "".to_string();
            }
        }
    }

    // Process llama_model_path: convert absolute paths to relative if they are within data/models, otherwise keep as is.
    let models_base_dir = project_root_dir.join("data").join("models");
    if !processed_model_config.llama_model_path.is_empty() {
        let current_model_path = PathBuf::from(&processed_model_config.llama_model_path);
        if current_model_path.is_absolute() {
            if let Ok(relative_path) = current_model_path.strip_prefix(&models_base_dir) {
                processed_model_config.llama_model_path =
                    relative_path.to_string_lossy().to_string();
                debug!(
                    "Converted absolute llama_model_path to relative: {}",
                    processed_model_config.llama_model_path
                );
            } else {
                // Similar to lora_path, if it's absolute but not in data/models, we should be cautious.
                // For now, keep it as is, but this might need further consideration if issues arise.
                // Or, if it's a model downloaded directly to project_root_dir, it might be canonical.
                debug!("Absolute llama_model_path '{}' is not within the expected data/models directory. Keeping as is.", processed_model_config.llama_model_path);
            }
        }
    }

    configs.insert(processed_model_config.id.clone(), processed_model_config);
    ModelsState::save_to_disk(&configs).await?;
    debug!("Model config saved successfully.");
    Ok("Saved.".to_string())
}

#[tauri::command]
async fn load_model_configs_command(
    models_state: State<'_, Arc<ModelsState>>,
) -> Result<Vec<ModelConfig>, String> {
    debug!("Loading model configurations.");
    let configs = models_state.configs.lock().await;
    Ok(configs.values().cloned().collect())
}

#[tauri::command]
async fn delete_model_config_command(
    models_state: State<'_, Arc<ModelsState>>,
    id: String,
) -> Result<String, String> {
    debug!("Attempting to delete model config for ID: {}", id);
    let mut configs = models_state.configs.lock().await;
    configs.remove(&id);
    ModelsState::save_to_disk(&configs).await?;
    debug!("Model config deleted successfully for ID: {}", id);
    Ok("Deleted.".to_string())
}

#[tauri::command]
async fn list_model_folders_command() -> Result<Vec<String>, String> {
    debug!("Listing model folders.");
    let models_dir = PathBuf::from("../data/models"); // Relative to src-tauri
    let mut folders = Vec::new();
    if models_dir.exists() && models_dir.is_dir() {
        let mut entries = fs::read_dir(models_dir).await.map_err(|e| {
            error!("Failed to read models directory: {}", e);
            e.to_string()
        })?;
        while let Some(entry) = entries.next_entry().await.map_err(|e| {
            error!("Failed to read directory entry: {}", e);
            e.to_string()
        })? {
            if entry
                .file_type()
                .await
                .map_err(|e| {
                    error!("Failed to get file type: {}", e);
                    e.to_string()
                })?
                .is_dir()
            {
                folders.push(entry.file_name().to_string_lossy().to_string());
            }
        }
    }
    // Truncate folder list for debug output if too long
    let display_folders = if folders.len() > 5 {
        format!("{:?}... ({} total)", &folders[0..5], folders.len())
    } else {
        format!("{:?}", folders)
    };
    debug!("Found model folders: {}", display_folders);
    Ok(folders)
}

#[tauri::command]
async fn list_finetuning_models_command() -> Result<Vec<String>, String> {
    debug!("Listing models suitable for fine-tuning.");
    let models_dir = PathBuf::from("../data/models");
    let mut models = Vec::new();

    if models_dir.exists() && models_dir.is_dir() {
        let mut entries = fs::read_dir(models_dir).await.map_err(|e| {
            error!("Failed to read models directory: {}", e);
            e.to_string()
        })?;

        while let Some(entry) = entries.next_entry().await.map_err(|e| {
            error!("Failed to read directory entry: {}", e);
            e.to_string()
        })? {
            if entry.file_type().await.map_err(|e| e.to_string())?.is_dir() {
                let folder_name = entry.file_name().to_string_lossy().to_string();
                let path = entry.path();

                // Filter out GGUF-only models
                let is_gguf_only = check_dir_for_gguf(&path).await;

                if !is_gguf_only {
                    // Convert folder name from "Org-Model" to "Org/Model" for HuggingFace format
                    models.push(folder_name);
                }
            }
        }
    }

    debug!("Found {} fine-tuning compatible models", models.len());
    Ok(models)
}

#[tauri::command]
async fn list_training_projects_command() -> Result<Vec<String>, String> {
    debug!("Listing existing training projects.");
    let outputs_dir = PathBuf::from("../data/outputs");
    let mut projects = Vec::new();

    if outputs_dir.exists() && outputs_dir.is_dir() {
        let mut entries = fs::read_dir(outputs_dir).await.map_err(|e| {
            error!("Failed to read outputs directory: {}", e);
            e.to_string()
        })?;

        while let Some(entry) = entries.next_entry().await.map_err(|e| {
            error!("Failed to read directory entry: {}", e);
            e.to_string()
        })? {
            if entry.file_type().await.map_err(|e| e.to_string())?.is_dir() {
                let folder_name = entry.file_name().to_string_lossy().to_string();
                // Skip special folders like .git, etc.
                if !folder_name.starts_with('.') {
                    projects.push(folder_name);
                }
            }
        }
    }

    projects.sort();
    debug!("Found {} training projects", projects.len());
    Ok(projects)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProjectLoraInfo {
    project_name: String,
    checkpoints: Vec<CheckpointInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CheckpointInfo {
    name: String, // e.g., "checkpoint-100", "final_model"
    path: String, // relative path from project root
    is_final: bool,
    step_number: Option<i32>,
}

#[tauri::command]
async fn list_loras_by_project_command() -> Result<Vec<ProjectLoraInfo>, String> {
    debug!("Listing LoRAs organized by training project.");
    let outputs_dir = PathBuf::from("../data/outputs");
    let mut projects_info = Vec::new();

    if outputs_dir.exists() && outputs_dir.is_dir() {
        let mut project_entries = fs::read_dir(&outputs_dir).await.map_err(|e| {
            error!("Failed to read outputs directory: {}", e);
            e.to_string()
        })?;

        while let Some(project_entry) = project_entries
            .next_entry()
            .await
            .map_err(|e| e.to_string())?
        {
            if !project_entry
                .file_type()
                .await
                .map_err(|e| e.to_string())?
                .is_dir()
            {
                continue;
            }

            let project_name = project_entry.file_name().to_string_lossy().to_string();
            if project_name.starts_with('.') {
                continue;
            }

            let project_path = project_entry.path();
            let mut checkpoints = Vec::new();

            // Scan for checkpoint directories and final_model
            let mut checkpoint_entries = fs::read_dir(&project_path)
                .await
                .map_err(|e| e.to_string())?;
            while let Some(checkpoint_entry) = checkpoint_entries
                .next_entry()
                .await
                .map_err(|e| e.to_string())?
            {
                if !checkpoint_entry
                    .file_type()
                    .await
                    .map_err(|e| e.to_string())?
                    .is_dir()
                {
                    continue;
                }

                let checkpoint_name = checkpoint_entry.file_name().to_string_lossy().to_string();
                let checkpoint_path = checkpoint_entry.path();

                // Check if this directory has adapter files (adapter_config.json)
                let adapter_config = checkpoint_path.join("adapter_config.json");
                if !adapter_config.exists() {
                    continue;
                }

                let is_final = checkpoint_name == "final_model";
                let step_number = if checkpoint_name.starts_with("checkpoint-") {
                    checkpoint_name
                        .strip_prefix("checkpoint-")
                        .and_then(|s| s.parse::<i32>().ok())
                } else {
                    None
                };

                let relative_path = format!("data/outputs/{}/{}", project_name, checkpoint_name);

                checkpoints.push(CheckpointInfo {
                    name: checkpoint_name,
                    path: relative_path,
                    is_final,
                    step_number,
                });
            }

            if !checkpoints.is_empty() {
                // Sort checkpoints: final_model first, then by step number descending
                checkpoints.sort_by(|a, b| {
                    if a.is_final && !b.is_final {
                        std::cmp::Ordering::Less
                    } else if !a.is_final && b.is_final {
                        std::cmp::Ordering::Greater
                    } else {
                        b.step_number.cmp(&a.step_number)
                    }
                });

                projects_info.push(ProjectLoraInfo {
                    project_name,
                    checkpoints,
                });
            }
        }
    }

    projects_info.sort_by(|a, b| a.project_name.cmp(&b.project_name));
    debug!("Found {} projects with LoRAs", projects_info.len());
    Ok(projects_info)
}

#[tauri::command]
async fn download_models_command(
    window: Window,
    models_state: State<'_, Arc<ModelsState>>,
    model_id_param: String,
) -> Result<String, String> {
    debug!(
        "Received download_models_command for model ID: {}",
        model_id_param
    );
    let configs = models_state.configs.lock().await;
    let config = configs.get(&model_id_param).ok_or("Config not found")?;
    debug!("Config found for model ID: {}", model_id_param);

    let (python_exe, work_dir) = get_python_command()?;
    let script = get_script_path("download_models.py");
    debug!(
        "Python executable: {:?}, Working directory: {:?}, Script: {}",
        python_exe, work_dir, script
    );

    let mut cmd = TokioCommand::new(&python_exe);
    cmd.arg(&script)
        .arg("--model_id")
        .arg(&config.base_model_id)
        .arg("--output_folder")
        .arg(&config.download_output_folder)
        .current_dir(&work_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    debug!("Spawning download process...");
    window
        .emit("log", format!("Starting download..."))
        .map_err(|e| e.to_string())?;

    let mut child = cmd.spawn().map_err(|e| {
        error!("Failed to spawn download process: {}", e);
        e.to_string()
    })?;

    // Store PID for cancellation
    if let Some(pid) = child.id() {
        let mut downloads = models_state.active_downloads.lock().await;
        downloads.insert(model_id_param.clone(), pid);
        debug!("Stored PID {} for download {}", pid, model_id_param);
    }

    let stdout = child.stdout.take().ok_or("No stdout")?;
    let stderr = child.stderr.take().ok_or("No stderr")?;

    // Stream logs
    let win_c = window.clone();
    tokio::spawn(async move {
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();
        while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
            debug!("PY_DOWNLOAD_STDOUT: {}", line.trim());
            win_c.emit("log", format!("PY: {}", line.trim())).unwrap();
            line.clear();
        }
    });

    let win_c = window.clone();
    let model_id = model_id_param.clone();
    tokio::spawn(async move {
        let mut reader = BufReader::new(stderr);
        let mut line = String::new();
        while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
            let trimmed = line.trim();
            debug!("PY_DOWNLOAD_STDERR: {}", trimmed);
            if trimmed.starts_with("PROGRESS:") {
                if let Some(value) = trimmed.strip_prefix("PROGRESS:") {
                    if let Ok(percentage) = value.parse::<u8>() {
                        // Emit event with correct structure: { id, progress }
                        #[derive(Serialize, Clone)]
                        struct ProgressPayload {
                            id: String,
                            progress: u8,
                        }
                        win_c
                            .emit(
                                "download_progress",
                                ProgressPayload {
                                    id: model_id.clone(),
                                    progress: percentage,
                                },
                            )
                            .unwrap();
                    }
                }
            } else if trimmed.contains("Not enough disk space") {
                win_c.emit("not_enough_space", trimmed).unwrap();
            } else {
                win_c.emit("log", format!("ERR: {}", trimmed)).unwrap();
            }
            line.clear();
        }
    });

    let status = child.wait().await.map_err(|e| {
        error!("Error waiting for download process: {}", e);
        e.to_string()
    })?;

    // Remove PID
    {
        let mut downloads = models_state.active_downloads.lock().await;
        downloads.remove(&model_id_param);
    }

    if status.success() {
        debug!("Download process completed successfully.");
        Ok("Download Complete".into())
    } else {
        error!("Download process failed with status: {:?}", status);
        Err("Download Failed".into())
    }
}

#[tauri::command]
async fn take_screenshot_path_command(
    window: Window,
    output_dir: String,
    filename_prefix: String,
) -> Result<String, String> {
    debug!(
        "Received take_screenshot_path_command. Output dir: {}, Prefix: {}",
        output_dir, filename_prefix
    );
    window.hide().map_err(|e| {
        error!("Failed to hide window: {}", e);
        e.to_string()
    })?;
    debug!("Window hidden.");

    let (python_exe, work_dir) = get_python_command()?;
    let script = get_script_path("screenshot_tool.py");
    debug!(
        "Python executable: {:?}, Working directory: {:?}, Script: {}",
        python_exe, work_dir, script
    );

    let mut cmd = TokioCommand::new(&python_exe);
    cmd.arg(&script)
        .arg("--output_dir")
        .arg(&output_dir)
        .arg("--filename_prefix")
        .arg(&filename_prefix)
        .current_dir(&work_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    debug!("Spawning screenshot process...");
    let mut child = cmd.spawn().map_err(|e| {
        error!("Failed to spawn screenshot process: {}", e);
        e.to_string()
    })?;
    let stdout = child.stdout.take().ok_or("No stdout stream")?;
    let stderr = child.stderr.take().ok_or("No stderr stream")?;
    debug!("Stdout and Stderr streams taken.");

    let mut reader_stdout = Some(BufReader::new(stdout));
    let mut reader_stderr = Some(BufReader::new(stderr));
    let mut path_found = String::new();
    let mut line_stdout = String::new();
    let mut line_stderr = String::new();

    debug!("Starting to read stdout and stderr streams...");
    loop {
        line_stdout.clear();
        line_stderr.clear();

        tokio::select! {
            result_stdout = async {
                if let Some(reader) = &mut reader_stdout {
                    reader.read_line(&mut line_stdout).await
                } else {
                    future::pending().await
                }
            } => {
                if let Ok(0) = result_stdout {
                    reader_stdout = None;
                    debug!("Stdout stream closed (EOF).");
                } else if let Ok(_) = result_stdout {
                    let trimmed = line_stdout.trim();
                    if !trimmed.is_empty() {
                        debug!("RAW_PYTHON_STDOUT: {}", trimmed);
                    }
                } else if let Err(e) = result_stdout {
                    error!("Error reading stdout from screenshot script: {}", e);
                    return Err(format!("Error reading stdout: {}", e));
                }
            },
            result_stderr = async {
                if let Some(reader) = &mut reader_stderr {
                    reader.read_line(&mut line_stderr).await
                } else {
                    future::pending().await
                }
            } => {
                if let Ok(0) = result_stderr {
                    reader_stderr = None;
                    debug!("Stderr stream closed (EOF).");
                } else if let Ok(_) = result_stderr {
                    let trimmed = line_stderr.trim();
                    if !trimmed.is_empty() {
                        debug!("RAW_PYTHON_STDERR: {}", trimmed);
                        window.emit("log", format!("ERR (SCREENSHOT): {}", trimmed)).map_err(|e| e.to_string())?;
                        if trimmed.contains("INFO: Screenshot saved to") {
                            if let Some(raw_path_part) = trimmed.split("INFO: Screenshot saved to").nth(1) {
                                let path_str = raw_path_part.trim();
                                path_found = path_str.to_string();
                                debug!("Extracted and directly used image path from Python: {}", path_found);
                            }
                        }
                    }
                } else if let Err(e) = result_stderr {
                    error!("Error reading stderr from screenshot script: {}", e);
                    return Err(format!("Error reading stderr: {}", e));
                }
            },
        }

        if reader_stdout.is_none() && reader_stderr.is_none() {
            debug!("Both stdout and stderr streams closed. Breaking loop.");
            break;
        }
    }

    let status = child.wait().await.map_err(|e| {
        error!("Error waiting for screenshot process: {}", e);
        e.to_string()
    })?;
    debug!("Screenshot process finished with status: {:?}", status);

    window.show().map_err(|e| {
        error!("Failed to show window: {}", e);
        e.to_string()
    })?;
    debug!("Window shown.");

    if !path_found.is_empty() {
        let clean_path = path_found.replace("\\", "/");
        debug!("Screenshot successful. Path: {}", clean_path);
        Ok(clean_path)
    } else {
        error!(
            "Screenshot script finished but returned no path. Path found: '{}'",
            path_found
        );
        Err("Screenshot script finished but returned no path.".into())
    }
}

#[tauri::command]
async fn save_annotation_crop_command(
    window: Window,
    image_path: String,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    label: String,
    save_dir: String,
) -> Result<String, String> {
    debug!("Received save_annotation_crop_command. Image path: {}, Box: ({},{},{},{}), Label: {}, Save dir: {}",
        image_path, x, y, w, h, label, save_dir);

    let (python_exe, work_dir) = get_python_command()?;

    // Adjust save_dir to be relative to the project root if it was originally relative to src-tauri
    // `save_dir` is expected to be relative to the project root already (e.g., "data/cropped_images")
    // The Python script will be executed with `current_dir` set to `project_root_dir`.
    // So, `Path.cwd() / r'{5}'` (where {5} is `save_dir`) will correctly resolve to `project_root_dir/data/cropped_images`.
    let python_code = format!(
        "from PIL import Image; import os; import sys; import logging; from pathlib import Path; logging.basicConfig(level=logging.INFO, stream=sys.stderr, format='%(levelname)s: %(message)s'); try: logging.info('Attempting to crop image: %s', r'{0}'); img = Image.open(r'{0}'); crop = img.crop(({1}, {2}, {1}+{3}, {2}+{4})); abs_save_dir = Path.cwd() / r'{5}'; os.makedirs(abs_save_dir, exist_ok=True); out_filename = f'{6}.png'; abs_out_path = abs_save_dir / out_filename; crop.save(abs_out_path); relative_path = abs_out_path.relative_to(Path.cwd()); path_to_return = str(relative_path).replace('\\\\', '/'); logging.info('Crop saved successfully to %s', path_to_return); print(f'SAVED_CROP_PATH:{{path_to_return}}'); except Exception as e: logging.error('Failed to save crop from %s: %s', r'{0}', e); sys.exit(1)",
        image_path, x, y, w, h, save_dir, label // Use save_dir directly
    );

    debug!("Python crop script code: {}", python_code);

    let mut cmd = TokioCommand::new(&python_exe);
    cmd.arg("-c")
        .arg(&python_code)
        .current_dir(&work_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    debug!("Spawning crop saving process...");
    let mut child = cmd.spawn().map_err(|e| {
        error!("Failed to spawn crop saving process: {}", e);
        e.to_string()
    })?;
    let stdout = child.stdout.take().ok_or("No stdout for crop save")?;
    let stderr = child.stderr.take().ok_or("No stderr for crop save")?;

    let win_c = window.clone();
    tokio::spawn(async move {
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();
        while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
            debug!("PY_CROP_STDOUT: {}", line.trim());
            win_c.emit("log", format!("CROP: {}", line.trim())).unwrap();
            line.clear();
        }
    });

    let win_c = window.clone();
    tokio::spawn(async move {
        let mut reader = BufReader::new(stderr);
        let mut line = String::new();
        while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
            debug!("PY_CROP_STDERR: {}", line.trim());
            win_c
                .emit("log", format!("CROP ERR: {}", line.trim()))
                .unwrap();
            line.clear();
        }
    });

    let status = child.wait().await.map_err(|e| {
        error!("Error waiting for crop saving process: {}", e);
        e.to_string()
    })?;

    if status.success() {
        debug!("Crop saving process completed successfully.");
        Ok("Crop saved.".into())
    } else {
        error!("Crop saving process failed with status: {:?}", status);
        Err("Failed to save crop.".into())
    }
}

#[tauri::command]
async fn download_datasets_command(
    window: Window,
    models_state: State<'_, Arc<ModelsState>>,
    dataset_id: String,
    output_folder: String,
) -> Result<String, String> {
    let (python_exe, work_dir) = get_python_command()?;
    let script = get_script_path("download_datasets.py");

    let mut cmd = TokioCommand::new(&python_exe);
    cmd.arg(&script)
        .arg("--dataset_id")
        .arg(&dataset_id)
        .arg("--output_folder")
        .arg(&output_folder)
        .current_dir(&work_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    debug!("Spawning dataset download process...");
    window
        .emit("log", format!("Starting dataset download..."))
        .map_err(|e| e.to_string())?;

    let mut child = cmd.spawn().map_err(|e| {
        error!("Failed to spawn dataset download process: {}", e);
        e.to_string()
    })?;

    // Store PID for cancellation
    if let Some(pid) = child.id() {
        let mut downloads = models_state.active_downloads.lock().await;
        downloads.insert(dataset_id.clone(), pid);
        debug!("Stored PID {} for dataset download {}", pid, dataset_id);
    }

    let stdout = child.stdout.take().ok_or("No stdout")?;
    let stderr = child.stderr.take().ok_or("No stderr")?;

    // Stream logs
    let win_c = window.clone();
    tokio::spawn(async move {
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();
        while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
            debug!("PY_DATASET_DOWNLOAD_STDOUT: {}", line.trim());
            win_c
                .emit("log", format!("PY_DATASET: {}", line.trim()))
                .unwrap();
            line.clear();
        }
    });

    let win_c = window.clone();
    let dataset_id_clone = dataset_id.clone();
    tokio::spawn(async move {
        let mut reader = BufReader::new(stderr);
        let mut line = String::new();
        while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
            let trimmed = line.trim();
            debug!("PY_DATASET_DOWNLOAD_STDERR: {}", trimmed);
            if trimmed.starts_with("PROGRESS:") {
                if let Some(value) = trimmed.strip_prefix("PROGRESS:") {
                    if let Ok(percentage) = value.parse::<u8>() {
                        // Emit generic download_progress event
                        #[derive(Serialize, Clone)]
                        struct ProgressPayload {
                            id: String,
                            progress: u8,
                        }
                        win_c
                            .emit(
                                "download_progress",
                                ProgressPayload {
                                    id: dataset_id_clone.clone(),
                                    progress: percentage,
                                },
                            )
                            .unwrap();
                    }
                }
            } else {
                win_c
                    .emit("log", format!("PY_DATASET_ERR: {}", trimmed))
                    .unwrap();
            }
            line.clear();
        }
    });

    let status = child.wait().await.map_err(|e| {
        error!("Error waiting for dataset download process: {}", e);
        e.to_string()
    })?;

    // Remove PID
    {
        let mut downloads = models_state.active_downloads.lock().await;
        downloads.remove(&dataset_id);
    }

    if status.success() {
        debug!("Dataset download process completed successfully.");
        Ok("Dataset Download Complete".into())
    } else {
        error!("Dataset download process failed with status: {:?}", status);
        Err("Dataset Download Failed".into())
    }
}

#[tauri::command]
async fn list_dataset_folders_command() -> Result<Vec<String>, String> {
    debug!("Listing dataset folders.");
    let datasets_dir = PathBuf::from("../data/datasets");
    let mut folders = Vec::new();
    if datasets_dir.exists() && datasets_dir.is_dir() {
        let mut entries = fs::read_dir(datasets_dir).await.map_err(|e| {
            error!("Failed to read datasets directory: {}", e);
            e.to_string()
        })?;
        while let Some(entry) = entries.next_entry().await.map_err(|e| {
            error!("Failed to read directory entry: {}", e);
            e.to_string()
        })? {
            if entry
                .file_type()
                .await
                .map_err(|e| {
                    error!("Failed to get file type: {}", e);
                    e.to_string()
                })?
                .is_dir()
            {
                folders.push(entry.file_name().to_string_lossy().to_string());
            }
        }
    }
    // Truncate folder list for debug output if too long
    let display_folders = if folders.len() > 5 {
        format!("{:?}... ({} total)", &folders[0..5], folders.len())
    } else {
        format!("{:?}", folders)
    };
    debug!("Found dataset folders: {}", display_folders);
    Ok(folders)
}

#[tauri::command]
async fn list_gguf_models_command() -> Result<Vec<String>, String> {
    debug!("Listing GGUF model files.");
    let models_dir = PathBuf::from("../data/models");
    let mut files = Vec::new();
    if models_dir.exists() && models_dir.is_dir() {
        let mut entries = fs::read_dir(models_dir).await.map_err(|e| {
            error!("Failed to read models directory for GGUF: {}", e);
            e.to_string()
        })?;
        while let Some(entry) = entries.next_entry().await.map_err(|e| {
            error!("Failed to read directory entry for GGUF: {}", e);
            e.to_string()
        })? {
            let path = entry.path();
            if path.is_file() && path.extension().map_or(false, |ext| ext == "gguf") {
                files.push(path.file_name().unwrap().to_string_lossy().to_string());
            } else if path.is_dir() {
                let mut sub_entries = fs::read_dir(&path).await.map_err(|e| {
                    error!("Failed to read subdirectory for GGUF: {}", e);
                    e.to_string()
                })?;
                while let Some(sub_entry) = sub_entries.next_entry().await.map_err(|e| {
                    error!("Failed to read sub-directory entry for GGUF: {}", e);
                    e.to_string()
                })? {
                    let sub_path = sub_entry.path();
                    if sub_path.is_file() && sub_path.extension().map_or(false, |ext| ext == "gguf")
                    {
                        let relative_path = sub_path
                            .strip_prefix(PathBuf::from("../data/models"))
                            .map_err(|e| e.to_string())?
                            .to_string_lossy()
                            .to_string()
                            .replace("\\", "/");
                        files.push(relative_path);
                    }
                }
            }
        }
    }
    // Truncate file list for debug output if too long
    let display_files = if files.len() > 5 {
        format!("{:?}... ({} total)", &files[0..5], files.len())
    } else {
        format!("{:?}", files)
    };
    debug!("Found GGUF model files: {}", display_files);
    Ok(files)
}

#[tauri::command]
async fn list_lora_adapters_command() -> Result<Vec<String>, String> {
    debug!("Listing LoRA adapter files.");
    let lora_dir = PathBuf::from("../data/loras");
    let mut files = Vec::new();
    if lora_dir.exists() && lora_dir.is_dir() {
        let mut entries = fs::read_dir(lora_dir).await.map_err(|e| {
            error!("Failed to read LoRA directory: {}", e);
            e.to_string()
        })?;
        while let Some(entry) = entries.next_entry().await.map_err(|e| {
            error!("Failed to read directory entry for LoRA: {}", e);
            e.to_string()
        })? {
            let path = entry.path();
            if path.is_file() && path.extension().map_or(false, |ext| ext == "bin") {
                files.push(path.file_name().unwrap().to_string_lossy().to_string());
            }
        }
    }
    // Truncate file list for debug output if too long
    let display_files = if files.len() > 5 {
        format!("{:?}... ({} total)", &files[0..5], files.len())
    } else {
        format!("{:?}", files)
    };
    debug!("Found LoRA adapter files: {}", display_files);
    Ok(files)
}

#[tauri::command]
async fn start_llama_server_command(
    window: Window,
    app_handle: AppHandle,
    python_process_state: State<'_, Arc<PythonProcessState>>,
    model_path: String,
    mmproj_path: String,
    lora_path: Option<String>,
    host: String,
    port: u16,
    n_gpu_layers: u64,
    ctx_size: u64,
    batch_size: u64,
    ubatch_size: u64,
    temp: f64,
    no_mmap: bool,
    flash_attn: bool,
) -> Result<String, String> {
    let mut child_guard = python_process_state.llama_server_child.lock().await;
    if child_guard.is_some() {
        return Err("Llama server is already running.".to_string());
    }

    let bin_dir = get_binaries_dir(&app_handle);
    let binary_name = if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    };
    let llama_server_path = bin_dir.join(binary_name);

    if !llama_server_path.exists() {
        let msg = format!(
            "Llama server not found at {:?}. Please download it from the menu.",
            llama_server_path
        );
        error!("{}", msg);
        return Err(msg);
    }

    let mut cmd = TokioCommand::new(&llama_server_path);
    cmd.current_dir(&bin_dir);

    let mut args: Vec<String> = Vec::new();
    let src_tauri_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let project_root_dir = src_tauri_dir
        .parent()
        .expect("Failed to get project root directory")
        .to_path_buf();

    let absolute_model_path = project_root_dir
        .join("data")
        .join("models")
        .join(&model_path);
    if !absolute_model_path.exists() {
        let msg = format!(
            "GGUF model not found at absolute path: {:?}",
            absolute_model_path
        );
        error!("{}", msg);
        return Err(msg);
    }

    args.push("-m".to_string());
    args.push(absolute_model_path.to_string_lossy().to_string());

    if !mmproj_path.trim().is_empty() {
        args.push("--mmproj".to_string());
        args.push(mmproj_path.clone());
    }
    args.push("--no-mmap".to_string());

    if let Some(lora) = lora_path {
        if !lora.trim().is_empty() {
            let final_lora_path: PathBuf;
            let potential_path = PathBuf::from(&lora);
            if potential_path.is_absolute() {
                final_lora_path = potential_path;
            } else {
                final_lora_path = project_root_dir.join("data").join("loras").join(&lora);
            }

            if final_lora_path.exists() {
                args.push("--lora".to_string());
                args.push(final_lora_path.to_string_lossy().to_string());
            } else {
                window
                    .emit(
                        "log",
                        format!(
                            "WARNING: LoRA path not found, skipping: {:?}",
                            final_lora_path
                        ),
                    )
                    .unwrap();
            }
        }
    }

    args.push("--host".to_string());
    args.push(host.clone());
    args.push("--port".to_string());
    args.push(port.to_string());
    args.push("-ngl".to_string());
    args.push(n_gpu_layers.to_string());
    args.push("-c".to_string());
    args.push(ctx_size.to_string());
    args.push("-b".to_string());
    args.push(batch_size.to_string());
    args.push("--ubatch-size".to_string());
    args.push(ubatch_size.to_string());

    args.push("-np".to_string());
    args.push("1".to_string());

    args.push("--no-mmap".to_string());
    if flash_attn {
        args.push("-fa".to_string());
        args.push("on".to_string());
    }

    // 4. Threads (-t 8).
    //    Reduces CPU overhead for driving the GPU.
    args.push("-t".to_string());
    args.push("8".to_string());

    args.push("--temp".to_string());
    args.push(temp.to_string());
    if no_mmap {
        args.push("--no-mmap".to_string());
    }

    // -------------------------------

    debug!(
        "Llama server command: {:?} with args: {:?}",
        llama_server_path, args
    );
    window
        .emit(
            "log",
            format!("Starting Llama Server (Single-User GPU Mode)..."),
        )
        .unwrap();

    cmd.args(args).stdout(Stdio::piped()).stderr(Stdio::piped());

    let mut child = cmd.spawn().map_err(|e| {
        error!("Failed to spawn Llama server process: {}", e);
        e.to_string()
    })?;

    let stdout = child.stdout.take().ok_or("No stdout for llama server")?;
    let stderr = child.stderr.take().ok_or("No stderr for llama server")?;

    let win_c = window.clone();
    tokio::spawn(async move {
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();
        while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
            let trimmed_line = line.trim();
            if !trimmed_line.is_empty() {
                debug!("LLAMA_SERVER_STDOUT: {}", trimmed_line);
                win_c
                    .emit("log", format!("LLAMA_SERVER: {}", trimmed_line))
                    .unwrap();
            }
            line.clear();
        }
    });

    // Added underscore to silence warning
    let _win_c_err = window.clone();
    tokio::spawn(async move {
        let mut reader = BufReader::new(stderr);
        let mut line = String::new();
        while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
            let trimmed_line = line.trim();
            if !trimmed_line.is_empty() {
                debug!("LLAMA_SERVER_STDERR: {}", trimmed_line);
            }
            line.clear();
        }
    });

    *child_guard = Some(child);
    Ok("Llama server started.".to_string())
}

#[tauri::command]
async fn stop_llama_server_command(
    python_process_state: State<'_, Arc<PythonProcessState>>,
) -> Result<String, String> {
    debug!("Received stop_llama_server_command.");
    let mut child_guard = python_process_state.llama_server_child.lock().await;
    if let Some(child) = child_guard.as_mut() {
        match child.kill().await {
            Ok(_) => {
                debug!("Llama server process killed successfully.");
                *child_guard = None;
                Ok("Llama server stopped.".to_string())
            }
            Err(e) => {
                error!("Failed to kill Llama server process: {}", e);
                Err(format!("Failed to stop Llama server: {}", e))
            }
        }
    } else {
        debug!("Llama server is not running.");
        Err("Llama server is not running.".into())
    }
}

#[tauri::command]
async fn send_chat_message_command(
    window: Window,
    llama_chat_context: State<'_, Arc<LlamaChatContext>>,
    host: String,
    port: u16,
    message: String,
    system_prompt: String,
    temperature: f64,
    top_p: f64,
    top_k: u64,
    _ctx_size: u64,
) -> Result<String, String> {
    debug!(
        "Received send_chat_message_command with message: {}",
        message
    );

    let mut chat_history = llama_chat_context.chat_history.lock().await;

    let mut user_message_map = HashMap::new();
    user_message_map.insert("role".to_string(), "user".to_string());
    user_message_map.insert("content".to_string(), message.clone());
    chat_history.push(user_message_map);

    let mut messages_for_server = Vec::new();

    if chat_history.len() == 1
        || chat_history
            .last()
            .map_or(false, |m| m.get("role").map_or(false, |r| r == "user"))
    {
        let mut system_message_map = HashMap::new();
        system_message_map.insert("role".to_string(), "system".to_string());
        system_message_map.insert("content".to_string(), system_prompt.clone());
        messages_for_server.push(system_message_map);
    }

    messages_for_server.extend(chat_history.iter().cloned());

    #[derive(Serialize)]
    struct LlamaChatRequest {
        messages: Vec<HashMap<String, String>>,
        temperature: f64,
        top_p: f64,
        top_k: u64,
        n_predict: i32,
        stop: Vec<String>,
        stream: bool,
    }

    let client = reqwest::Client::new();
    let connect_host = if host == "0.0.0.0" {
        "127.0.0.1"
    } else {
        &host
    };
    let url = format!("http://{}:{}/v1/chat/completions", connect_host, port);

    let request_body = LlamaChatRequest {
        messages: messages_for_server,
        temperature,
        top_p,
        top_k,
        n_predict: -1,
        stop: vec!["<|im_end|>".to_string(), "<|endoftext|>".to_string()],
        stream: false,
    };

    window
        .emit(
            "log",
            format!("Sending message to Llama server: '{}'", message),
        )
        .map_err(|e| e.to_string())?;

    let res = client
        .post(&url)
        .json(&request_body)
        .send()
        .await
        .map_err(|e| {
            error!("Failed to send message to Llama server: {}", e);
            e.to_string()
        })?;

    if res.status().is_success() {
        // We parse directly to Value to avoid unused struct warnings
        let val: serde_json::Value = res.json().await.map_err(|e| e.to_string())?;

        // Safely extract content from OpenAI format: choices[0].message.content
        let bot_response = val["choices"][0]["message"]["content"]
            .as_str()
            .ok_or("Failed to parse content from response")?
            .to_string();

        let mut bot_message_map = HashMap::new();
        bot_message_map.insert("role".to_string(), "assistant".to_string());
        bot_message_map.insert("content".to_string(), bot_response.clone());
        chat_history.push(bot_message_map);

        debug!("Llama server responded: {}", bot_response);
        Ok(bot_response)
    } else {
        let status = res.status();
        let text = res
            .text()
            .await
            .unwrap_or_else(|_| "No response text".to_string());
        error!(
            "Llama server returned an error: Status: {}, Response: {}",
            status, text
        );
        Err(format!("Llama server error: {} - {}", status, text))
    }
}

#[tauri::command]
async fn clear_chat_history_command(
    llama_chat_context: State<'_, Arc<LlamaChatContext>>,
) -> Result<(), String> {
    debug!("Clearing chat history");
    let mut chat_history = llama_chat_context.chat_history.lock().await;
    chat_history.clear();
    Ok(())
}

#[derive(Serialize, Clone)]
struct StreamChunk {
    content: String,
}

#[derive(Serialize, Clone)]
struct StreamMetrics {
    prompt_eval_time_ms: f64,
    eval_time_ms: f64,
    tokens_per_second: f64,
    total_tokens: u64,
}

#[tauri::command]
async fn send_chat_message_streaming_command(
    window: Window,
    llama_chat_context: State<'_, Arc<LlamaChatContext>>,
    host: String,
    port: u16,
    message: String,
    system_prompt: String,
    temperature: f64,
    top_p: f64,
    top_k: u64,
    _ctx_size: u64,
) -> Result<String, String> {
    use futures_util::StreamExt;

    debug!("Streaming chat message: {}", message);

    let mut chat_history = llama_chat_context.chat_history.lock().await;

    let mut user_message_map = HashMap::new();
    user_message_map.insert("role".to_string(), "user".to_string());
    user_message_map.insert("content".to_string(), message.clone());
    chat_history.push(user_message_map);

    let mut messages_for_server = Vec::new();

    if chat_history.len() == 1
        || chat_history
            .last()
            .map_or(false, |m| m.get("role").map_or(false, |r| r == "user"))
    {
        let mut system_message_map = HashMap::new();
        system_message_map.insert("role".to_string(), "system".to_string());
        system_message_map.insert("content".to_string(), system_prompt.clone());
        messages_for_server.push(system_message_map);
    }

    messages_for_server.extend(chat_history.iter().cloned());

    // Add stream_options to request usage stats if supported
    #[derive(Serialize)]
    struct StreamOptions {
        include_usage: bool,
    }

    #[derive(Serialize)]
    struct LlamaChatStreamRequest {
        messages: Vec<HashMap<String, String>>,
        temperature: f64,
        top_p: f64,
        top_k: u64,
        n_predict: i32,
        stop: Vec<String>,
        stream: bool,
        stream_options: Option<StreamOptions>,
    }

    let client = reqwest::Client::new();
    let connect_host = if host == "0.0.0.0" {
        "127.0.0.1"
    } else {
        &host
    };
    let url = format!("http://{}:{}/v1/chat/completions", connect_host, port);

    let request_body = LlamaChatStreamRequest {
        messages: messages_for_server,
        temperature,
        top_p,
        top_k,
        n_predict: -1,
        stop: vec!["<|im_end|>".to_string(), "<|endoftext|>".to_string()],
        stream: true,
        stream_options: Some(StreamOptions {
            include_usage: true,
        }),
    };

    window
        .emit(
            "log",
            format!("Starting streaming request to Llama server..."),
        )
        .map_err(|e| e.to_string())?;

    let start_time = std::time::Instant::now();
    let mut first_token_time: Option<std::time::Instant> = None;
    let mut token_count = 0;

    let response = client
        .post(&url)
        .json(&request_body)
        .send()
        .await
        .map_err(|e| {
            error!("Failed to send streaming request: {}", e);
            e.to_string()
        })?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response
            .text()
            .await
            .unwrap_or_else(|_| "No response text".to_string());
        error!("Llama server returned error: {} - {}", status, text);
        return Err(format!("Server error: {} - {}", status, text));
    }

    let mut stream = response.bytes_stream();
    let mut buffer = String::new();

    while let Some(item) = stream.next().await {
        let chunk = item.map_err(|e| e.to_string())?;
        let s = String::from_utf8_lossy(&chunk);
        buffer.push_str(&s);

        while let Some(pos) = buffer.find('\n') {
            let line = buffer[..pos].trim().to_string();
            buffer = buffer[pos + 1..].to_string();

            if line.starts_with("data: ") {
                let data_str = &line[6..];
                if data_str == "[DONE]" {
                    break;
                }

                if let Ok(json) = serde_json::from_str::<serde_json::Value>(data_str) {
                    // Handle content
                    if let Some(choices) = json.get("choices") {
                        if let Some(choice) = choices.get(0) {
                            if let Some(choice_obj) = choice.as_object() {
                                if let Some(delta) = choice_obj.get("delta") {
                                    if let Some(content) = delta.get("content") {
                                        if let Some(content_str) = content.as_str() {
                                            if !content_str.is_empty() {
                                                if first_token_time.is_none() {
                                                    first_token_time =
                                                        Some(std::time::Instant::now());
                                                }
                                                token_count += 1;

                                                let chunk_data = StreamChunk {
                                                    content: content_str.to_string(),
                                                };
                                                window
                                                    .emit("chat-stream-chunk", chunk_data)
                                                    .unwrap_or(());
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Calculate metrics
    let prompt_eval_time_ms = first_token_time
        .map(|t| t.duration_since(start_time).as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    let eval_time_ms = if let Some(first) = first_token_time {
        let now = std::time::Instant::now();
        now.duration_since(first).as_secs_f64() * 1000.0
    } else {
        0.0
    };

    let tps = if eval_time_ms > 0.0 {
        (token_count as f64) / (eval_time_ms / 1000.0)
    } else {
        0.0
    };

    let metrics = StreamMetrics {
        prompt_eval_time_ms,
        eval_time_ms,
        tokens_per_second: tps,
        total_tokens: token_count,
    };

    window
        .emit("chat-stream-done", metrics)
        .map_err(|e| e.to_string())?;

    Ok("Stream complete".to_string())
}

#[tauri::command]
async fn get_chat_response_command(
    _window: Window, // Window is not directly used for response retrieval but kept for consistency
    llama_chat_context: State<'_, Arc<LlamaChatContext>>,
) -> Result<Vec<HashMap<String, String>>, String> {
    debug!("Received get_chat_response_command.");
    let chat_history = llama_chat_context.chat_history.lock().await;
    Ok(chat_history.clone())
}

#[tauri::command]
async fn check_llama_server_status_command(
    python_process_state: State<'_, Arc<PythonProcessState>>,
) -> Result<bool, String> {
    debug!("Received check_llama_server_status_command.");
    let child_guard = python_process_state.llama_server_child.lock().await;
    Ok(child_guard.is_some())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceInfo {
    pub name: String,
    pub size: String,
    pub path: String,
    pub r#type: String,
    pub quantization: Option<String>,
    pub is_mmproj: bool,
    pub is_processed: Option<bool>,
    pub dataset_format: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Preset {
    pub id: String,
    pub name: String,
    pub model: String,
    pub lora: String,
}

const PRESETS_FILE: &str = "../presets.json";

const HF_TOKEN_FILE: &str = "../hf_token.txt";

// Enhanced resource listing with GGUF quantization detection
#[tauri::command]
async fn list_all_resources_command() -> Result<Vec<ResourceInfo>, String> {
    let mut resources = Vec::new();

    // HF Models
    if let Ok(mut entries) = fs::read_dir("../data/models").await {
        while let Ok(Some(entry)) = entries.next_entry().await {
            if entry.file_type().await.map(|t| t.is_dir()).unwrap_or(false) {
                let name = entry.file_name().to_string_lossy().to_string();
                let path_obj = entry.path();

                // Check if this directory contains GGUF files
                let has_gguf = check_dir_for_gguf(&path_obj).await;

                if !has_gguf {
                    // Regular HF model folder
                    let size = get_dir_size(&path_obj)
                        .await
                        .unwrap_or_else(|_| "Unknown".to_string());
                    resources.push(ResourceInfo {
                        name: name.clone(),
                        size,
                        path: format!("data/models/{}", name),
                        r#type: "model".to_string(),
                        quantization: None,
                        is_mmproj: false,
                        is_processed: None,
                        dataset_format: None,
                    });
                } else {
                    // Directory contains GGUF files - list them individually
                    if let Ok(mut gguf_entries) = fs::read_dir(&path_obj).await {
                        while let Ok(Some(gguf_entry)) = gguf_entries.next_entry().await {
                            let gguf_path = gguf_entry.path();
                            if gguf_path.extension().map_or(false, |ext| ext == "gguf") {
                                let gguf_name =
                                    gguf_entry.file_name().to_string_lossy().to_string();
                                let size = get_file_size(&gguf_path)
                                    .await
                                    .unwrap_or_else(|_| "Unknown".to_string());
                                let quant = detect_quantization(&gguf_name);
                                resources.push(ResourceInfo {
                                    name: gguf_name.clone(),
                                    size,
                                    path: format!("data/models/{}/{}", name, gguf_name),
                                    r#type: "gguf".to_string(),
                                    quantization: Some(quant),
                                    is_mmproj: gguf_name.contains("mmproj"),
                                    is_processed: None,
                                    dataset_format: None,
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    // GGUF Models in root models directory
    if let Ok(mut entries) = fs::read_dir("../data/models").await {
        while let Ok(Some(entry)) = entries.next_entry().await {
            let path = entry.path();
            if path.is_file() && path.extension().map_or(false, |ext| ext == "gguf") {
                let name = entry.file_name().to_string_lossy().to_string();
                let size = get_file_size(&path)
                    .await
                    .unwrap_or_else(|_| "Unknown".to_string());
                let quant = detect_quantization(&name);
                resources.push(ResourceInfo {
                    name: name.clone(),
                    size,
                    path: format!("data/models/{}", name),
                    r#type: "gguf".to_string(),
                    quantization: Some(quant),
                    is_mmproj: name.contains("mmproj"),
                    is_processed: None,
                    dataset_format: None,
                });
            }
        }
    }

    // LoRAs
    if let Ok(mut entries) = fs::read_dir("../data/loras").await {
        while let Ok(Some(entry)) = entries.next_entry().await {
            let path = entry.path();
            if path.is_file() {
                let name = entry.file_name().to_string_lossy().to_string();
                let size = get_file_size(&path)
                    .await
                    .unwrap_or_else(|_| "Unknown".to_string());
                resources.push(ResourceInfo {
                    name: name.clone(),
                    size,
                    path: format!("data/loras/{}", name),
                    r#type: "lora".to_string(),
                    quantization: None,
                    is_mmproj: false,
                    is_processed: None,
                    dataset_format: None,
                });
            }
        }
    }

    // Datasets
    if let Ok(mut entries) = fs::read_dir("../data/datasets").await {
        while let Ok(Some(entry)) = entries.next_entry().await {
            if entry.file_type().await.map(|t| t.is_dir()).unwrap_or(false) {
                let name = entry.file_name().to_string_lossy().to_string();
                let dataset_path = entry.path();
                let size = get_dir_size(&dataset_path)
                    .await
                    .unwrap_or_else(|_| "Unknown".to_string());

                // Check if processed_data folder exists
                let is_processed = dataset_path.join("processed_data").exists();

                // Detect dataset format from files in the directory
                let mut dataset_format: Option<String> = None;
                if let Ok(mut file_entries) = fs::read_dir(&dataset_path).await {
                    while let Ok(Some(file_entry)) = file_entries.next_entry().await {
                        let file_path = file_entry.path();
                        if file_path.is_file() {
                            if let Some(fmt) = detect_dataset_format(&file_path) {
                                dataset_format = Some(fmt);
                                break; // Found format, move on
                            }
                        }
                    }
                }

                resources.push(ResourceInfo {
                    name: name.clone(),
                    size,
                    path: format!("data/datasets/{}", name),
                    r#type: "dataset".to_string(),
                    quantization: None,
                    is_mmproj: false,
                    is_processed: Some(is_processed),
                    dataset_format,
                });
            }
        }
    }

    Ok(resources)
}

async fn check_dir_for_gguf(dir: &PathBuf) -> bool {
    if let Ok(mut entries) = fs::read_dir(dir).await {
        while let Ok(Some(entry)) = entries.next_entry().await {
            if entry.path().extension().map_or(false, |ext| ext == "gguf") {
                return true;
            }
        }
    }
    false
}

fn detect_quantization(filename: &str) -> String {
    let name_lower = filename.to_lowercase();

    if name_lower.contains("fp16") || name_lower.contains("f16") {
        "FP16".to_string()
    } else if name_lower.contains("bf16") {
        "BF16".to_string()
    } else if name_lower.contains("fp32") || name_lower.contains("f32") {
        "FP32".to_string()
    } else if name_lower.contains("q8") {
        "Q8".to_string()
    } else if name_lower.contains("q7") {
        "Q7".to_string()
    } else if name_lower.contains("q6") {
        "Q6".to_string()
    } else if name_lower.contains("q5") {
        "Q5".to_string()
    } else if name_lower.contains("q4") {
        "Q4".to_string()
    } else if name_lower.contains("q3") {
        "Q3".to_string()
    } else if name_lower.contains("q2") {
        "Q2".to_string()
    } else {
        "other".to_string()
    }
}

// HuggingFace Token Management
#[tauri::command]
async fn get_hf_token_command() -> Result<String, String> {
    let token_path = PathBuf::from(HF_TOKEN_FILE);
    if token_path.exists() {
        let token = fs::read_to_string(&token_path)
            .await
            .map_err(|e| e.to_string())?;
        Ok(token.trim().to_string())
    } else {
        Err("No token found".to_string())
    }
}

#[tauri::command]
async fn save_hf_token_command(token: String) -> Result<String, String> {
    fs::write(HF_TOKEN_FILE, token.trim())
        .await
        .map_err(|e| e.to_string())?;
    Ok("Token saved".to_string())
}

// HuggingFace Search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HFSearchResult {
    pub id: String,
    pub name: String,
    pub downloads: u64,
    pub likes: u64,
}

// Export Resources
#[tauri::command]
async fn export_resources_command(
    resource_paths: Vec<String>,
    destination: String,
) -> Result<String, String> {
    let dest_path = PathBuf::from(&destination);

    if !dest_path.exists() {
        return Err("Destination path does not exist".to_string());
    }

    let count = resource_paths.len();
    for resource_path in resource_paths {
        let source = PathBuf::from("..").join(&resource_path);
        let file_name = source
            .file_name()
            .ok_or("Invalid source path")?
            .to_string_lossy()
            .to_string();
        let dest = dest_path.join(&file_name);

        if source.is_dir() {
            copy_dir_all(&source, &dest).await?;
        } else {
            fs::copy(&source, &dest).await.map_err(|e| e.to_string())?;
        }
    }

    Ok(format!("Exported {} resources", count))
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HFFile {
    pub path: String,
    pub size: Option<u64>,
    pub lfs: Option<serde_json::Value>,
    pub file_type: String, // e.g., "gguf", "mmproj", "weight", "other"
    pub quantization: Option<String>,
    pub is_mmproj: bool,
}

// --- UPDATED HUGGINGFACE INTEGRATION ---

#[tauri::command]
async fn search_huggingface_command(
    query: String,
    resource_type: String, // "model" or "dataset"
) -> Result<Vec<HFSearchResult>, String> {
    let (python_exe, work_dir) = get_python_command()?;
    let script = get_script_path("huggingface_manager.py");

    // Get token if exists
    let token = get_hf_token_command().await.ok();

    let mut cmd = TokioCommand::new(&python_exe);
    cmd.arg(&script)
        .arg("search")
        .arg("--query")
        .arg(&query)
        .arg("--type")
        .arg(&resource_type)
        .current_dir(&work_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    if let Some(t) = token {
        if !t.is_empty() {
            cmd.arg("--token").arg(t);
        }
    }

    let child = cmd
        .spawn()
        .map_err(|e| format!("Failed to spawn search: {}", e))?;
    let output = child.wait_with_output().await.map_err(|e| e.to_string())?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Search script failed: {}", stderr));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Parse JSON output from Python
    let results: Vec<HFSearchResult> = serde_json::from_str(&stdout).map_err(|e| {
        format!(
            "Failed to parse Python search results: {}. Output: {}",
            e, stdout
        )
    })?;

    Ok(results)
}

#[tauri::command]
async fn list_hf_repo_files_command(
    repo_id: String,
    token: Option<String>,
    resource_type: Option<String>,
) -> Result<Vec<HFFile>, String> {
    let (python_exe, work_dir) = get_python_command()?;
    let script = get_script_path("huggingface_manager.py");
    let repo_type = resource_type.as_deref().unwrap_or("model");

    let mut cmd = TokioCommand::new(&python_exe);
    cmd.arg(&script)
        .arg("list")
        .arg("--repo_id")
        .arg(&repo_id)
        .arg("--type")
        .arg(repo_type)
        .current_dir(&work_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    if let Some(t) = token {
        if !t.is_empty() {
            cmd.arg("--token").arg(t);
        }
    }

    let child = cmd.spawn().map_err(|e| e.to_string())?;
    let output = child.wait_with_output().await.map_err(|e| e.to_string())?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        // Attempt to parse JSON error from Python
        if let Ok(json_err) =
            serde_json::from_str::<serde_json::Value>(&String::from_utf8_lossy(&output.stdout))
        {
            if let Some(err_msg) = json_err.get("error").and_then(|v| v.as_str()) {
                return Err(err_msg.to_string());
            }
        }
        return Err(format!("Failed to list files: {}", stderr));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let files: Vec<HFFile> = serde_json::from_str(&stdout).map_err(|e| {
        format!(
            "Failed to parse Python list_hf_repo_files results: {}. Output: {}",
            e, stdout
        )
    })?;

    Ok(files)
}

#[tauri::command]
async fn download_hf_model_command(
    window: Window,
    models_state: State<'_, Arc<ModelsState>>,
    model_id: String,
    files: Option<Vec<String>>,
    token: Option<String>,
    task_id: Option<String>,
) -> Result<String, String> {
    let (python_exe, work_dir) = get_python_command()?;
    let script = get_script_path("huggingface_manager.py");

    let sanitized_repo_id = model_id.replace('/', "--");
    let base_output_folder = PathBuf::from("data")
        .join("models")
        .join(&sanitized_repo_id);

    let mut cmd = TokioCommand::new(&python_exe);
    cmd.arg(&script)
        .arg("download")
        .arg("--repo_id")
        .arg(&model_id);

    if let Some(f) = files {
        if f.is_empty() {
            return Err("No files selected for download".to_string());
        }
        let output_arg_path = base_output_folder.to_string_lossy().to_string();

        cmd.arg("--output")
            .arg(&output_arg_path)
            .arg("--files")
            .arg(f.join(","));
    }

    cmd.current_dir(&work_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    if let Some(t) = token {
        if !t.is_empty() {
            cmd.arg("--token").arg(t);
        }
    }

    // Spawn process
    let mut child = cmd.spawn().map_err(|e| e.to_string())?;

    // Store PID if task_id is provided
    if let Some(tid) = &task_id {
        if let Some(pid) = child.id() {
            let mut downloads = models_state.active_downloads.lock().await;
            downloads.insert(tid.clone(), pid);
            debug!("Stored PID {} for task {}", pid, tid);
        }
    }

    let stdout = child.stdout.take().ok_or("No stdout")?;
    let stderr = child.stderr.take().ok_or("No stderr")?;

    // Log stdout
    let win_c = window.clone();
    tokio::spawn(async move {
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();
        while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
            win_c
                .emit("log", format!("HF: {}", line.trim()))
                .unwrap_or(());
            line.clear();
        }
    });

    // Log stderr + Parse Progress
    let win_c = window.clone();
    let task_id_c = task_id.clone();
    tokio::spawn(async move {
        let mut reader = BufReader::new(stderr);
        let mut line = String::new();
        while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
            let trimmed = line.trim();
            if trimmed.starts_with("PROGRESS:") {
                if let Some(value) = trimmed.strip_prefix("PROGRESS:") {
                    if let Ok(percentage) = value.parse::<u8>() {
                        if let Some(tid) = &task_id_c {
                            let payload = serde_json::json!({ "id": tid, "progress": percentage });
                            win_c.emit("download_progress", payload).unwrap_or(());
                        } else {
                            win_c.emit("download_progress", percentage).unwrap_or(());
                        }
                    }
                }
            } else {
                win_c.emit("log", format!("HF: {}", trimmed)).unwrap_or(());
            }
            line.clear();
        }
    });

    // Spawn background task to monitor completion
    let win_c = window.clone();
    let task_id_c = task_id.clone();
    let models_state_c = (*models_state).clone();
    tokio::spawn(async move {
        let status = child.wait().await;

        // Remove PID
        if let Some(tid) = &task_id_c {
            let mut downloads = models_state_c.active_downloads.lock().await;
            downloads.remove(tid);
        }

        if let Some(tid) = task_id_c {
            if let Ok(status) = status {
                if status.success() {
                    let payload = serde_json::json!({ "id": tid, "status": "completed" });
                    win_c.emit("download_status", payload).unwrap_or(());
                } else {
                    let payload = serde_json::json!({ "id": tid, "status": "error" });
                    win_c.emit("download_status", payload).unwrap_or(());
                }
            } else {
                let payload = serde_json::json!({ "id": tid, "status": "error" });
                win_c.emit("download_status", payload).unwrap_or(());
            }
        }
    });

    // Return immediately - download happens in background
    Ok("Download started".to_string())
}

#[tauri::command]
async fn download_hf_dataset_command(
    window: Window,
    models_state: State<'_, Arc<ModelsState>>,
    dataset_id: String,
    files: Option<Vec<String>>,
    token: Option<String>,
    task_id: Option<String>,
) -> Result<String, String> {
    let (python_exe, work_dir) = get_python_command()?;
    let script = get_script_path("huggingface_manager.py");

    let sanitized_dataset_id = dataset_id.replace('/', "--");
    let output_folder = PathBuf::from("data")
        .join("datasets")
        .join(&sanitized_dataset_id);
    let output_arg_path = output_folder.to_string_lossy().to_string();

    let mut cmd = TokioCommand::new(&python_exe);
    cmd.arg(&script)
        .arg("download")
        .arg("--repo_id")
        .arg(&dataset_id)
        .arg("--output")
        .arg(&output_arg_path)
        .arg("--type")
        .arg("dataset");

    if let Some(f) = files {
        if !f.is_empty() {
            cmd.arg("--files").arg(f.join(","));
        }
    }

    cmd.current_dir(&work_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    if let Some(t) = token {
        if !t.is_empty() {
            cmd.arg("--token").arg(t);
        }
    }

    let mut child = cmd.spawn().map_err(|e| e.to_string())?;

    // Store PID if task_id is provided
    if let Some(tid) = &task_id {
        if let Some(pid) = child.id() {
            let mut downloads = models_state.active_downloads.lock().await;
            downloads.insert(tid.clone(), pid);
            debug!("Stored PID {} for task {}", pid, tid);
        }
    }

    let stdout = child.stdout.take().ok_or("No stdout")?;
    let stderr = child.stderr.take().ok_or("No stderr")?;

    let win_c = window.clone();
    tokio::spawn(async move {
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();
        while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
            win_c
                .emit("log", format!("HF_DS: {}", line.trim()))
                .unwrap_or(());
            line.clear();
        }
    });

    let win_c = window.clone();
    let task_id_c = task_id.clone();
    tokio::spawn(async move {
        let mut reader = BufReader::new(stderr);
        let mut line = String::new();
        while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
            let trimmed = line.trim();
            if trimmed.starts_with("PROGRESS:") {
                if let Some(value) = trimmed.strip_prefix("PROGRESS:") {
                    if let Ok(percentage) = value.parse::<u8>() {
                        if let Some(tid) = &task_id_c {
                            let payload = serde_json::json!({ "id": tid, "progress": percentage });
                            win_c.emit("download_progress", payload).unwrap_or(());
                        } else {
                            win_c.emit("download_progress", percentage).unwrap_or(());
                        }
                    }
                }
            } else {
                win_c
                    .emit("log", format!("HF_DS: {}", trimmed))
                    .unwrap_or(());
            }
            line.clear();
        }
    });

    // Spawn background task to monitor completion
    let win_c = window.clone();
    let task_id_c = task_id.clone();
    let models_state_c = (*models_state).clone();
    tokio::spawn(async move {
        let status = child.wait().await;

        // Remove PID
        if let Some(tid) = &task_id_c {
            let mut downloads = models_state_c.active_downloads.lock().await;
            downloads.remove(tid);
        }

        if let Some(tid) = task_id_c {
            if let Ok(status) = status {
                if status.success() {
                    let payload = serde_json::json!({ "id": tid, "status": "completed" });
                    win_c.emit("download_status", payload).unwrap_or(());
                } else {
                    let payload = serde_json::json!({ "id": tid, "status": "error" });
                    win_c.emit("download_status", payload).unwrap_or(());
                }
            } else {
                let payload = serde_json::json!({ "id": tid, "status": "error" });
                win_c.emit("download_status", payload).unwrap_or(());
            }
        }
    });

    Ok("Download started".to_string())
}

// Helper: Detect dataset format from file extensions
fn detect_dataset_format(path: &PathBuf) -> Option<String> {
    if let Some(ext) = path.extension() {
        match ext.to_str().unwrap_or("").to_lowercase().as_str() {
            "arrow" => Some("arrow".to_string()),
            "csv" => Some("csv".to_string()),
            "json" => Some("json".to_string()),
            "jsonl" => Some("jsonl".to_string()),
            "parquet" => Some("parquet".to_string()),
            _ => None,
        }
    } else {
        None
    }
}

#[tauri::command]
async fn convert_dataset_command(
    window: Window,
    source_path: String,
    destination_path: String,
) -> Result<String, String> {
    let (python_exe, work_dir) = get_python_command()?;
    let script = get_script_path("huggingface_manager.py");

    let source_abs = if PathBuf::from(&source_path).is_absolute() {
        source_path.clone()
    } else {
        work_dir.join(&source_path).to_string_lossy().to_string()
    };

    let dest_abs = if PathBuf::from(&destination_path).is_absolute() {
        destination_path.clone()
    } else {
        work_dir
            .join(&destination_path)
            .to_string_lossy()
            .to_string()
    };

    let mut cmd = TokioCommand::new(&python_exe);
    cmd.arg(&script)
        .arg("convert")
        .arg("--source_path")
        .arg(&source_abs)
        .arg("--output_path")
        .arg(&dest_abs)
        .current_dir(&work_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = cmd.spawn().map_err(|e| e.to_string())?;
    let stdout = child.stdout.take().ok_or("No stdout")?;
    let stderr = child.stderr.take().ok_or("No stderr")?;

    let win_c = window.clone();
    tokio::spawn(async move {
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();
        while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
            if !line.trim().is_empty() {
                win_c
                    .emit("log", format!("CONVERT: {}", line.trim()))
                    .unwrap_or(());
            }
            line.clear();
        }
    });

    let win_c = window.clone();
    tokio::spawn(async move {
        let mut reader = BufReader::new(stderr);
        let mut line = String::new();
        while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
            if !line.trim().is_empty() {
                win_c
                    .emit("log", format!("CONVERT: {}", line.trim()))
                    .unwrap_or(());
            }
            line.clear();
        }
    });

    let status = child.wait().await.map_err(|e| e.to_string())?;
    if status.success() {
        Ok("Dataset converted successfully".to_string())
    } else {
        Err("Dataset conversion failed".to_string())
    }
}

async fn get_file_size(path: &PathBuf) -> Result<String, String> {
    let metadata = fs::metadata(path).await.map_err(|e| e.to_string())?;
    Ok(format_size(metadata.len()))
}

async fn get_dir_size(path: &PathBuf) -> Result<String, String> {
    let mut total = 0u64;
    let mut stack = vec![path.clone()];

    while let Some(current) = stack.pop() {
        if let Ok(mut entries) = fs::read_dir(&current).await {
            while let Ok(Some(entry)) = entries.next_entry().await {
                let entry_path = entry.path();
                if let Ok(metadata) = entry.metadata().await {
                    if metadata.is_dir() {
                        stack.push(entry_path);
                    } else {
                        total += metadata.len();
                    }
                }
            }
        }
    }

    Ok(format_size(total))
}

fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

#[tauri::command]
async fn delete_resource_command(
    resource_type: String,
    resource_path: String,
) -> Result<String, String> {
    let full_path = PathBuf::from("..").join(&resource_path);

    if full_path.is_dir() {
        fs::remove_dir_all(&full_path)
            .await
            .map_err(|e| e.to_string())?;
    } else {
        fs::remove_file(&full_path)
            .await
            .map_err(|e| e.to_string())?;
    }

    Ok(format!("{} deleted successfully", resource_type))
}

#[tauri::command]
async fn import_resource_command(
    resource_type: String,
    source_path: String,
) -> Result<String, String> {
    let source = PathBuf::from(&source_path);
    let dest_base = match resource_type.as_str() {
        "model" => PathBuf::from("../data/models"),
        "gguf" => PathBuf::from("../data/models"),
        "lora" => PathBuf::from("../data/loras"),
        "dataset" => PathBuf::from("../data/datasets"),
        _ => return Err("Invalid resource type".to_string()),
    };

    let file_name = source
        .file_name()
        .ok_or("Invalid source path")?
        .to_string_lossy()
        .to_string();
    let dest = dest_base.join(&file_name);

    if source.is_dir() {
        copy_dir_all(&source, &dest).await?;
    } else {
        fs::copy(&source, &dest).await.map_err(|e| e.to_string())?;
    }

    Ok(format!("{} imported successfully", resource_type))
}

async fn copy_dir_all(src: &PathBuf, dst: &PathBuf) -> Result<(), String> {
    fs::create_dir_all(dst).await.map_err(|e| e.to_string())?;
    let mut entries = fs::read_dir(src).await.map_err(|e| e.to_string())?;

    while let Some(entry) = entries.next_entry().await.map_err(|e| e.to_string())? {
        let ty = entry.file_type().await.map_err(|e| e.to_string())?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());

        if ty.is_dir() {
            Box::pin(copy_dir_all(&src_path, &dst_path)).await?;
        } else {
            fs::copy(&src_path, &dst_path)
                .await
                .map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

// Preset management
#[tauri::command]
async fn load_presets_command() -> Result<Vec<Preset>, String> {
    let presets_path = PathBuf::from(PRESETS_FILE);
    if !presets_path.exists() {
        return Ok(Vec::new());
    }
    let contents = fs::read_to_string(&presets_path)
        .await
        .map_err(|e| e.to_string())?;
    let presets: Vec<Preset> = serde_json::from_str(&contents).map_err(|e| e.to_string())?;
    Ok(presets)
}

#[tauri::command]
async fn save_preset_command(preset: Preset) -> Result<String, String> {
    let mut presets = load_presets_command().await.unwrap_or_default();
    presets.push(preset);
    let serialized = serde_json::to_string_pretty(&presets).map_err(|e| e.to_string())?;
    fs::write(PRESETS_FILE, serialized)
        .await
        .map_err(|e| e.to_string())?;
    Ok("Preset saved".to_string())
}

#[tauri::command]
async fn convert_hf_to_gguf_command(
    window: Window,
    source_path: String,
    output_path: Option<String>,
    quantization_type: String, // New parameter
) -> Result<String, String> {
    let (python_exe, work_dir) = get_python_command()?;
    // Assuming the script is named convert_hf_to_gguf.py
    let script = get_script_path("convert_hf_to_gguf.py");

    let mut cmd = TokioCommand::new(&python_exe);
    cmd.arg(&script);

    // 1. Add optional flags first
    if let Some(out) = output_path {
        cmd.arg("--outfile").arg(out);
    }

    // Add quantization type
    cmd.arg("--outtype").arg(quantization_type);

    // 2. Add the positional 'model' argument (as defined in parse_args: parser.add_argument("model", ...))
    cmd.arg(&source_path);

    cmd.current_dir(&work_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = cmd.spawn().map_err(|e| e.to_string())?;

    // Take handles to stdout and stderr
    let stdout = child.stdout.take().ok_or("No stdout")?;
    let stderr = child.stderr.take().ok_or("No stderr")?;

    // Spawn a task to stream stdout to the frontend logs
    let win_c_stdout = window.clone();
    let stdout_handle = tokio::spawn(async move {
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();
        while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
            win_c_stdout
                .emit("log", format!("CONVERT: {}", line.trim()))
                .unwrap_or(());
            line.clear();
        }
    });

    // Spawn a task to stream stderr AND capture it for the error return
    let win_c_stderr = window.clone();
    let stderr_handle = tokio::spawn(async move {
        let mut reader = BufReader::new(stderr);
        let mut line = String::new();
        let mut captured_stderr = String::new();
        while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
            let trimmed = line.trim();
            if !trimmed.is_empty() {
                // Emit to UI log
                win_c_stderr
                    .emit("log", format!("CONVERT ERR: {}", trimmed))
                    .unwrap_or(());
                // Append to internal buffer
                captured_stderr.push_str(trimmed);
                captured_stderr.push('\n');
            }
            line.clear();
        }
        captured_stderr
    });

    // Wait for process to exit
    let status = child.wait().await.map_err(|e| e.to_string())?;

    // Ensure IO tasks finish
    let _ = stdout_handle.await;
    let full_stderr = stderr_handle.await.map_err(|e| e.to_string())?;

    if status.success() {
        Ok("Conversion complete".to_string())
    } else {
        error!("Conversion failed. Stderr: {}", full_stderr);
        Err(format!("Conversion failed: {}", full_stderr.trim()))
    }
}

#[tauri::command]
async fn convert_lora_to_gguf_command(
    window: Window,
    lora_path: String,
    base_path: String,
    output_path: Option<String>,
    quantization_type: String, // New parameter
) -> Result<String, String> {
    let (python_exe, work_dir) = get_python_command()?;
    let script = get_script_path("convert_lora_to_gguf.py");

    let mut cmd = TokioCommand::new(&python_exe);
    cmd.arg(&script)
        .arg("--base")
        .arg(&base_path)
        .arg("--lora")
        .arg(&lora_path);

    if let Some(out) = output_path {
        cmd.arg("--outfile").arg(out);
    }

    // Add quantization type
    cmd.arg("--outtype").arg(quantization_type);

    cmd.current_dir(&work_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = cmd.spawn().map_err(|e| e.to_string())?;
    let stdout = child.stdout.take().ok_or("No stdout")?;
    let stderr = child.stderr.take().ok_or("No stderr")?;

    let win_c_stdout = window.clone();
    let stdout_handle = tokio::spawn(async move {
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();
        while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
            win_c_stdout
                .emit("log", format!("CONVERT_LORA: {}", line.trim()))
                .unwrap_or(());
            line.clear();
        }
    });

    let win_c_stderr = window.clone();
    let stderr_handle = tokio::spawn(async move {
        let mut reader = BufReader::new(stderr);
        let mut line = String::new();
        let mut captured_stderr = String::new();
        while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
            let trimmed = line.trim();
            if !trimmed.is_empty() {
                win_c_stderr
                    .emit("log", format!("CONVERT_LORA ERR: {}", trimmed))
                    .unwrap_or(());
                captured_stderr.push_str(trimmed);
                captured_stderr.push('\n');
            }
            line.clear();
        }
        captured_stderr
    });

    let status = child.wait().await.map_err(|e| e.to_string())?;

    let _ = stdout_handle.await;
    let full_stderr = stderr_handle.await.map_err(|e| e.to_string())?;

    if status.success() {
        Ok("LoRA conversion complete".to_string())
    } else {
        Err(format!("LoRA conversion failed: {}", full_stderr.trim()))
    }
}

#[tauri::command]
async fn cancel_download_command(
    models_state: State<'_, Arc<ModelsState>>,
    task_id: String,
) -> Result<String, String> {
    debug!("Attempting to cancel download for task ID: {}", task_id);
    let mut downloads = models_state.active_downloads.lock().await;

    if let Some(pid) = downloads.get(&task_id) {
        debug!("Found PID {} for task {}", pid, task_id);

        #[cfg(target_os = "windows")]
        {
            let output = StdCommand::new("taskkill")
                .args(&["/F", "/PID", &pid.to_string()])
                .output()
                .map_err(|e| e.to_string())?;

            if output.status.success() {
                debug!("Successfully killed process {}", pid);
            } else {
                let stderr = String::from_utf8_lossy(&output.stderr);
                error!("Failed to kill process {}: {}", pid, stderr);
                // Don't return error, just log it, maybe it's already gone
            }
        }

        #[cfg(not(target_os = "windows"))]
        {
            let output = StdCommand::new("kill")
                .args(&["-9", &pid.to_string()])
                .output()
                .map_err(|e| e.to_string())?;
            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                error!("Failed to kill process {}: {}", pid, stderr);
            }
        }

        // Remove from map
        downloads.remove(&task_id);
        Ok(format!("Download {} cancelled", task_id))
    } else {
        // warn!("No active download found for task ID: {}", task_id);
        // It might have finished already
        Ok(format!(
            "Download {} not found or already finished",
            task_id
        ))
    }
}

// Training command with TensorBoard
#[tauri::command]
async fn start_training_command(
    window: Window,
    python_process_state: State<'_, Arc<PythonProcessState>>,
    project_name: String,
    model_path: String,
    dataset_path: String,
    num_epochs: u32,
    batch_size: u32,
    learning_rate: f64,
    lora_r: u32,
    lora_alpha: u32,
    max_seq_length: u32,
) -> Result<serde_json::Value, String> {
    let (python_exe, work_dir) = get_python_command()?;

    // Ensure TensorBoard is running with project-specific logdir
    let logdir = format!("data/outputs/{}", project_name);

    // Attempt to read existing port
    let existing_port = {
        let guard = python_process_state.tensorboard_port.lock().await;
        *guard
    };

    let tensorboard_port = if let Some(p) = existing_port {
        p
    } else {
        // Find a free port by binding to port 0 then dropping the listener
        let listener = std::net::TcpListener::bind("127.0.0.1:0")
            .map_err(|e| format!("Failed to bind ephemeral port: {}", e))?;
        let port = listener.local_addr().map_err(|e| e.to_string())?.port();
        drop(listener);

        // On Windows, prefer launching TensorBoard in a detached terminal to avoid shared shells
        #[cfg(target_os = "windows")]
        {
            // Prepare a logs directory and a .bat that launches TensorBoard redirecting output
            let project_root = work_dir.clone();
            let logs_dir = project_root.join("logs");
            if let Err(e) = std::fs::create_dir_all(&logs_dir) {
                window
                    .emit(
                        "log",
                        format!(
                            "TENSORBOARD: failed to create logs dir {:?}: {}",
                            logs_dir, e
                        ),
                    )
                    .ok();
            }

            let logfile = logs_dir.join(format!("tensorboard_{}.log", port));
            let batfile = logs_dir.join(format!("tensorboard_{}.bat", port));

            // Build the command line to run in the .bat and redirect stdout/stderr
            let bat_cmd = format!(
                "\"{}\" -m tensorboard.main --logdir \"{}\" --port {} --host 127.0.0.1 > \"{}\" 2>&1",
                python_exe,
                logdir,
                port,
                logfile.display()
            );

            // Write the bat file
            if let Err(e) = std::fs::write(&batfile, &bat_cmd) {
                window
                    .emit(
                        "log",
                        format!(
                            "TENSORBOARD: failed to write launcher bat {:?}: {}",
                            batfile, e
                        ),
                    )
                    .ok();
            } else {
                window
                    .emit(
                        "log",
                        format!(
                            "TENSORBOARD: wrote launcher bat at {:?}; logfile at {:?}",
                            batfile, logfile
                        ),
                    )
                    .ok();
            }

            // Start the .bat in a new window using cmd start. Use empty title "" to avoid title parsing issues.
            let bat_path_str = batfile.to_string_lossy().to_string();
            let spawn_res = StdCommand::new("cmd")
                .args(&["/C", "start", "", &bat_path_str])
                .current_dir(&work_dir)
                .spawn();

            match spawn_res {
                Ok(_p) => {
                    let mut pguard = python_process_state.tensorboard_port.lock().await;
                    *pguard = Some(port);

                    window
                        .emit(
                            "log",
                            format!(
                                "TENSORBOARD: Launched detached via bat; logfile: {:?}",
                                logfile
                            ),
                        )
                        .ok();

                    // Poll for readiness and emit STATUS
                    let win_poll = window.clone();
                    let check_port = port;
                    tokio::spawn(async move {
                        let client = reqwest::Client::new();
                        let url = format!("http://127.0.0.1:{}/", check_port);
                        let mut attempts = 0;
                        while attempts < 60 {
                            if client.get(&url).send().await.is_ok() {
                                win_poll
                                    .emit(
                                        "log",
                                        format!(
                                            "STATUS: TensorBoard available at http://localhost:{}",
                                            check_port
                                        ),
                                    )
                                    .ok();
                                return;
                            }
                            attempts += 1;
                            tokio::time::sleep(std::time::Duration::from_millis(1000)).await;
                        }
                        win_poll.emit("log", format!("TENSORBOARD: failed to respond on http://localhost:{} after timeout (check logfile {:?})", check_port, logfile)).ok();
                    });
                }
                Err(e) => {
                    window
                        .emit("log", format!("TENSORBOARD START FAILED (detached): {}", e))
                        .ok();
                }
            }
        }

        #[cfg(not(target_os = "windows"))]
        {
            let mut tb_cmd = TokioCommand::new(&python_exe);
            tb_cmd
                .arg("-m")
                .arg("tensorboard")
                .arg("--logdir")
                .arg(logdir)
                .arg("--port")
                .arg(port.to_string())
                .arg("--host")
                .arg("0.0.0.0")
                .current_dir(&work_dir)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());

            // Spawn TensorBoard and capture stdout/stderr to stream to frontend
            let mut tb_child = tb_cmd.spawn().map_err(|e| {
                format!(
                    "Failed to start TensorBoard: {}. Make sure dependencies are installed.",
                    e
                )
            })?;

            // Take handles to stdout/stderr before storing child
            let tb_stdout = tb_child.stdout.take();
            let tb_stderr = tb_child.stderr.take();

            // Store the TensorBoard child and port
            {
                let mut guard = python_process_state.tensorboard_child.lock().await;
                *guard = Some(tb_child);
                let mut pguard = python_process_state.tensorboard_port.lock().await;
                *pguard = Some(port);
            }

            // Stream stdout
            if let Some(stdout) = tb_stdout {
                let win_c2 = window.clone();
                tokio::spawn(async move {
                    let mut reader = BufReader::new(stdout);
                    let mut line = String::new();
                    while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
                        let trimmed = line.trim();
                        if !trimmed.is_empty() {
                            win_c2
                                .emit("log", format!("TENSORBOARD: {}", trimmed))
                                .unwrap_or(());
                        }
                        line.clear();
                    }
                });
            }

            // Stream stderr
            if let Some(stderr) = tb_stderr {
                let win_c3 = window.clone();
                tokio::spawn(async move {
                    let mut reader = BufReader::new(stderr);
                    let mut line = String::new();
                    while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
                        let trimmed = line.trim();
                        if !trimmed.is_empty() {
                            win_c3
                                .emit("log", format!("TENSORBOARD: {}", trimmed))
                                .unwrap_or(());
                        }
                        line.clear();
                    }
                });
            }

            // Poll TensorBoard HTTP endpoint until ready (timeout ~60s) and then emit STATUS
            {
                let win_poll = window.clone();
                let check_port = port;
                tokio::spawn(async move {
                    let client = reqwest::Client::new();
                    let url = format!("http://127.0.0.1:{}/", check_port);
                    let mut attempts = 0;
                    while attempts < 60 {
                        if client.get(&url).send().await.is_ok() {
                            win_poll
                                .emit(
                                    "log",
                                    format!(
                                        "STATUS: TensorBoard available at http://localhost:{}",
                                        check_port
                                    ),
                                )
                                .ok();
                            return;
                        }
                        attempts += 1;
                        tokio::time::sleep(std::time::Duration::from_millis(1000)).await;
                    }
                    win_poll
                        .emit(
                            "log",
                            format!(
                                "TENSORBOARD: failed to respond on http://localhost:{} after timeout",
                                check_port
                            ),
                        )
                        .ok();
                });
            }
        }

        port
    };

    // Start training script
    let script = get_script_path("train.py");
    let mut cmd = TokioCommand::new(&python_exe);
    cmd.arg(&script)
        .arg("--model")
        .arg(&model_path)
        .arg("--dataset")
        .arg(&dataset_path)
        .arg("--epochs")
        .arg(num_epochs.to_string())
        .arg("--batch_size")
        .arg(batch_size.to_string())
        .arg("--learning_rate")
        .arg(learning_rate.to_string())
        .arg("--lora_r")
        .arg(lora_r.to_string())
        .arg("--lora_alpha")
        .arg(lora_alpha.to_string())
        .arg("--max_seq_length")
        .arg(max_seq_length.to_string())
        .arg("--output_dir")
        .arg(format!("data/outputs/{}", project_name))
        .current_dir(&work_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = cmd.spawn().map_err(|e| e.to_string())?;
    let stdout = child.stdout.take().ok_or("No stdout")?;
    let stderr = child.stderr.take().ok_or("No stderr")?;

    let win_c = window.clone();
    tokio::spawn(async move {
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();
        while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
            win_c
                .emit("log", format!("TRAIN: {}", line.trim()))
                .unwrap();
            line.clear();
        }
    });

    let win_c = window.clone();
    tokio::spawn(async move {
        let mut reader = BufReader::new(stderr);
        let mut line = String::new();
        while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
            win_c
                .emit("log", format!("TRAIN: {}", line.trim()))
                .unwrap();
            line.clear();
        }
    });

    let mut result = serde_json::Map::new();
    result.insert(
        "tensorboard_port".to_string(),
        serde_json::json!(tensorboard_port),
    );
    Ok(serde_json::Value::Object(result))
}

#[tauri::command]
async fn check_python_installed_command() -> Result<bool, String> {
    match get_python_command() {
        Ok(_) => Ok(true),
        Err(_) => Ok(false),
    }
}

#[tauri::command]
async fn setup_python_env_command(
    window: Window,
    _app_handle: AppHandle,
) -> Result<String, String> {
    let (python_exe, work_dir) = get_python_command()?;
    debug!("Setting up Python environment using: {}", python_exe);
    window
        .emit("log", format!("Using Python: {}", python_exe))
        .unwrap();

    // Robustness: Clean up any invalid 'site-packages' directories (starting with ~)
    // This fixes "Ignoring invalid distribution ~ympy" warnings/errors.
    let python_path = PathBuf::from(&python_exe);
    if let Some(parent) = python_path.parent() {
        let site_packages = parent.join("Lib").join("site-packages");
        if site_packages.exists() {
            if let Ok(entries) = std::fs::read_dir(&site_packages) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                        if name.starts_with('~') {
                            debug!("Cleaning up invalid distribution directory: {:?}", path);
                            let _ = std::fs::remove_dir_all(&path);
                        }
                    }
                }
            }
        }
    }

    // Define commands to run
    let commands = vec![
        // 1. Upgrade pip
        vec!["-m", "pip", "install", "--upgrade", "pip"],
        // 2. Install all dependencies with specific versions
        vec![
            "-m",
            "pip",
            "install",
            "--extra-index-url",
            "https://download.pytorch.org/whl/cu121",
            "dotenv==0.9.9",
            "markdown-it-py==4.0.0",
            "datasets==4.3.0",
            "python-dotenv==1.2.1",
            "bitsandbytes==0.44.1",
            "peft==0.18.0",
            "diffusers==0.35.2",
            "PyRect==0.2.0",
            "PyScreeze==1.0.1",
            "python-dotenv==1.2.1",
            "pytweening==1.2.0",
            "referencing==0.37.0",
            "mistral_common==1.8.5",
            "MouseInfo==0.1.3",
            "requests-oauthlib==2.0.0",
            "rich==14.2.0",
            "rpds-py==0.29.0",
            "rsa==4.9.1",
            "scikit_build_core==0.11.6",
            "scipy==1.16.3",
            "sentencepiece==0.2.1",
            "setuptools==70.2.0",
            "shellingham==1.5.4",
            "shtab==1.8.0",
            "sniffio==1.3.1",
            "soupsieve==2.8",
            "starlette==0.49.3",
            "tenacity==9.1.2",
            "tiktoken==0.12.0",
            "torch==2.5.1+cu121",
            "torchaudio==2.5.1+cu121",
            "torchvision==0.20.1+cu121",
            "trl==0.24.0",
            "typeguard==4.4.4",
            "typer-slim==0.20.0",
            "typing-inspection==0.4.2",
            "tyro==0.9.35",
            "ultralytics==8.3.230",
            "ultralytics-thop==2.0.18",
            "uritemplate==4.2.0",
            "uv==0.9.7",
            "uvicorn==0.38.0",
            "websockets==15.0.1",
            "Werkzeug==3.1.3",
            "wheel==0.45.1",
            "zipp==3.23.0",
            "git+https://github.com/ggml-org/llama.cpp.git#subdirectory=gguf-py",
            "transformers==4.57.3",
        ],
    ];

    // Track installation status
    let mut failed_packages = Vec::new();
    let total_steps = commands.len();

    for (idx, args) in commands.iter().enumerate() {
        let step_num = idx + 1;
        let cmd_str = format!("python {}", args.join(" "));

        window
            .emit(
                "log",
                format!("📦 [{}/{}] {}", step_num, total_steps, cmd_str),
            )
            .unwrap();
        debug!("Running command: {:?}", args);

        let mut cmd = TokioCommand::new(&python_exe);
        cmd.args(args)
            .current_dir(&work_dir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                let err_msg = format!("Failed to spawn pip: {}", e);
                window.emit("log", format!("⚠️ {}", err_msg)).unwrap_or(());
                failed_packages.push((cmd_str.clone(), err_msg));
                continue; // Continue with next package
            }
        };

        let stdout = child.stdout.take();
        let stderr = child.stderr.take();

        // Stream stdout
        if let Some(stdout) = stdout {
            let win_c = window.clone();
            tokio::spawn(async move {
                let mut reader = BufReader::new(stdout);
                let mut line = String::new();
                while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
                    win_c
                        .emit("log", format!("PIP: {}", line.trim()))
                        .unwrap_or(());
                    line.clear();
                }
            });
        }

        // Stream and capture stderr
        let stderr_handle = if let Some(stderr) = stderr {
            let win_c = window.clone();
            Some(tokio::spawn(async move {
                let mut reader = BufReader::new(stderr);
                let mut line = String::new();
                let mut captured = String::new();
                while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
                    let trimmed = line.trim();
                    // Only emit non-WARNING messages to reduce noise
                    if !trimmed.starts_with("WARNING:") {
                        win_c.emit("log", format!("PIP: {}", trimmed)).unwrap_or(());
                    }
                    captured.push_str(trimmed);
                    captured.push('\n');
                    line.clear();
                }
                captured
            }))
        } else {
            None
        };

        let status_result = child.wait().await;

        let captured_stderr = if let Some(handle) = stderr_handle {
            handle.await.unwrap_or_else(|_| String::new())
        } else {
            String::new()
        };

        // Check if command succeeded
        let success = status_result.as_ref().map(|s| s.success()).unwrap_or(false);

        if !success {
            let error_summary = if captured_stderr.contains("No matching distribution found") {
                "Package version not available in the index"
            } else if captured_stderr.contains("ERROR: Could not find a version") {
                "Package not found or incompatible version"
            } else if captured_stderr.contains("network") || captured_stderr.contains("timed out") {
                "Network error - please check your internet connection"
            } else {
                "Installation failed"
            };

            let err_msg = format!(
                "{}: {}",
                error_summary,
                captured_stderr.lines().next().unwrap_or("Unknown error")
            );

            window
                .emit(
                    "log",
                    format!(
                        "⚠️ Step {}/{} failed: {}",
                        step_num, total_steps, error_summary
                    ),
                )
                .unwrap_or(());

            failed_packages.push((cmd_str, err_msg));
            // Continue with next package instead of failing entirely
            continue;
        } else {
            window
                .emit(
                    "log",
                    format!("✅ Step {}/{} complete", step_num, total_steps),
                )
                .unwrap_or(());
        }
    }

    // Final summary
    if failed_packages.is_empty() {
        window
            .emit("log", "✅ Python environment setup complete!".to_string())
            .unwrap();
        Ok("Setup complete".to_string())
    } else {
        let failed_count = failed_packages.len();
        let success_count = total_steps - failed_count;

        window
            .emit(
                "log",
                format!(
                    "⚠️ Setup completed with {} successes and {} failures",
                    success_count, failed_count
                ),
            )
            .unwrap();

        for (cmd, err) in &failed_packages {
            window
                .emit("log", format!("❌ Failed: {} - {}", cmd, err))
                .unwrap_or(());
        }

        // Return partial success if at least core packages installed
        if success_count >= 2 {
            window
                .emit(
                    "log",
                    "⚠️ Core packages installed. You may continue, but some features might not work.".to_string(),
                )
                .unwrap();
            Ok(format!(
                "Partial setup: {}/{} packages installed",
                success_count, total_steps
            ))
        } else {
            Err(format!(
                "Setup failed: Only {}/{} packages installed successfully",
                success_count, total_steps
            ))
        }
    }
}

// ============================================================================
// PYTHON STANDALONE INSTALLATION (Windows Native)
// ============================================================================

/// Helper function to get the Python standalone directory path
fn get_python_standalone_dir(_app_handle: &AppHandle) -> PathBuf {
    let src_tauri_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let project_root_dir = src_tauri_dir
        .parent()
        .expect("Failed to get project root directory")
        .to_path_buf();
    project_root_dir.join("python")
}

/// Check if Python standalone is already installed
#[tauri::command]
async fn check_python_standalone_command(app_handle: AppHandle) -> Result<bool, String> {
    let python_dir = get_python_standalone_dir(&app_handle);
    let python_exe = python_dir.join("python.exe");

    debug!("Checking for Python standalone at: {:?}", python_exe);
    let exists = python_exe.exists();
    debug!("Python standalone exists: {}", exists);

    Ok(exists)
}

/// Download and install Python standalone for Windows
#[tauri::command]
async fn download_python_standalone_command(
    window: Window,
    app_handle: AppHandle,
) -> Result<String, String> {
    #[cfg(not(target_os = "windows"))]
    {
        return Err("Python standalone download is only supported on Windows".to_string());
    }

    #[cfg(target_os = "windows")]
    {
        use tauri::Emitter;

        window
            .emit(
                "setup_progress",
                serde_json::json!({
                    "step": "init",
                    "message": "Starting Python download...",
                    "progress": 0
                }),
            )
            .unwrap_or(());

        // Use Python 3.11 standalone build
        let python_version = "20241016";
        let build_name = "cpython-3.11.10+20241016-x86_64-pc-windows-msvc-install_only";
        let download_url = format!(
            "https://github.com/indygreg/python-build-standalone/releases/download/{}/{}.tar.gz",
            python_version, build_name
        );

        debug!("Downloading Python from: {}", download_url);

        // Download the file
        let client = reqwest::Client::new();
        let response = client
            .get(&download_url)
            .send()
            .await
            .map_err(|e| format!("Failed to download Python: {}", e))?
            .error_for_status()
            .map_err(|e| format!("Download error: {}", e))?;

        let total_size = response.content_length().unwrap_or(0);
        let content = response
            .bytes()
            .await
            .map_err(|e| format!("Failed to read download: {}", e))?;

        window
            .emit(
                "setup_progress",
                serde_json::json!({
                    "step": "download",
                    "message": "Download complete. Extracting...",
                    "progress": 50
                }),
            )
            .unwrap_or(());

        // Target directory: .../python
        let target_dir = get_python_standalone_dir(&app_handle);
        let project_root = target_dir
            .parent()
            .ok_or("Failed to get project root")?
            .to_path_buf();

        debug!("Extracting to project root: {:?}", project_root);

        // Extract in a blocking task
        let project_root_clone = project_root.clone();

        tokio::task::spawn_blocking(move || {
            use flate2::read::GzDecoder;
            use tar::Archive;

            let tar = GzDecoder::new(std::io::Cursor::new(content));
            let mut archive = Archive::new(tar);

            // Unpack to project root. This creates "python" directory directly.
            archive
                .unpack(&project_root_clone)
                .map_err(|e| format!("Failed to extract: {}", e))?;

            Ok::<(), String>(())
        })
        .await
        .map_err(|e| format!("Task join error: {}", e))??;

        // Verify installation
        let python_exe = target_dir.join("python.exe");
        if !python_exe.exists() {
            return Err(format!(
                "Python extraction completed but python.exe not found at {:?}",
                python_exe
            ));
        }

        window
            .emit(
                "setup_progress",
                serde_json::json!({
                    "step": "complete",
                    "message": "Python installed successfully!",
                    "progress": 100
                }),
            )
            .unwrap_or(());

        Ok("Python standalone installation complete".to_string())
    }
}

#[tauri::command]
async fn stop_training_command(
    python_process_state: State<'_, Arc<PythonProcessState>>,
) -> Result<String, String> {
    // Try to gracefully stop the TensorBoard child if we started it
    {
        let mut guard = python_process_state.tensorboard_child.lock().await;
        if let Some(child) = guard.as_mut() {
            if let Err(e) = child.kill().await {
                error!("Failed to kill TensorBoard child: {}", e);
            } else {
                debug!("TensorBoard child killed successfully.");
            }
            *guard = None;
        }
    }

    // Fallback: Kill generic python/tensorboard processes (platform-specific)
    #[cfg(target_os = "windows")]
    {
        let _ = StdCommand::new("taskkill")
            .args(&["/F", "/IM", "python.exe"])
            .output();
        let _ = StdCommand::new("taskkill")
            .args(&["/F", "/IM", "tensorboard.exe"])
            .output();
    }

    #[cfg(not(target_os = "windows"))]
    {
        let _ = StdCommand::new("pkill").arg("python").output();
        let _ = StdCommand::new("pkill").arg("tensorboard").output();
    }

    Ok("Training stopped".to_string())
}

// Update the run() function to include all new commands
pub fn run() {
    env_logger::init();

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(Arc::new(PythonProcessState::default()))
        .manage(Arc::new(ModelsState {
            configs: Mutex::new(
                tokio::runtime::Runtime::new()
                    .unwrap()
                    .block_on(async { ModelsState::load_from_disk().await.unwrap_or_default() }),
            ),
            active_downloads: Mutex::new(HashMap::new()),
        }))
        .manage(Arc::new(LlamaChatContext::default()))
        .setup(|app| {
            // Spawn a background task to start TensorBoard at app startup and keep it running.
            let app_handle = app.handle().clone();

            tauri::async_runtime::spawn(async move {
                // Retrieve state via the AppHandle (avoids borrowing `app` across the async boundary)
                let python_state = app_handle.state::<Arc<PythonProcessState>>().clone();

                // If already started, skip
                let already = { let g = python_state.tensorboard_port.lock().await; *g };
                if already.is_some() {
                    return;
                }

                // Reserve a free port
                let listener = match std::net::TcpListener::bind("127.0.0.1:0") {
                    Ok(l) => l,
                    Err(e) => {
                        if let Some(win) = app_handle.get_webview_window("main") {
                            win.emit("log", format!("TENSORBOARD START ERROR: could not bind port: {}", e)).ok();
                        }
                        return;
                    }
                };
                let port = match listener.local_addr() {
                    Ok(addr) => addr.port(),
                    Err(_) => return,
                };
                drop(listener);

                let (python_exe, work_dir) = match get_python_command() {
                    Ok(result) => result,
                    Err(e) => {
                        if let Some(win) = app_handle.get_webview_window("main") {
                            win.emit("log", format!("ERROR: Python not found: {}", e)).ok();
                        }
                        return;
                    }
                };
                // On Windows, start TensorBoard in a detached new terminal window so it runs independently.
                #[cfg(target_os = "windows")]
                {
                    let port_str = port.to_string();
                    // cmd /C start "" <python> -m tensorboard --logdir <dir> --port <port> --host 127.0.0.1
                    // We pass "" as the first argument to start to act as the window title, ensuring python_exe is treated as the command.
                    let spawn_res = StdCommand::new("cmd")
                        .args(&[
                            "/C",
                            "start",
                            "", 
                            &python_exe,
                            "-m",
                            "tensorboard",
                            "--logdir",
                            "data/outputs",
                            "--port",
                            &port_str,
                            "--host",
                            "127.0.0.1",
                        ])
                        .current_dir(&work_dir)
                        .spawn();

                    match spawn_res {
                        Ok(_proc) => {
                            // We can't capture the child easily when using 'start'. Save only the port and poll readiness.
                            let mut pguard = python_state.tensorboard_port.lock().await;
                            *pguard = Some(port);

                            if let Some(win) = app_handle.get_webview_window("main") {
                                let w3 = win.clone();
                                let check_port = port;
                                tokio::spawn(async move {
                                    let client = reqwest::Client::new();
                                    let url = format!("http://127.0.0.1:{}/", check_port);
                                    let mut attempts = 0;
                                    while attempts < 60 {
                                        if client.get(&url).send().await.is_ok() {
                                            w3.emit("log", format!("STATUS: TensorBoard available at http://localhost:{}", check_port)).ok();
                                            return;
                                        }
                                        attempts += 1;
                                        tokio::time::sleep(std::time::Duration::from_millis(1000)).await;
                                    }
                                    w3.emit("log", format!("TENSORBOARD: failed to respond on http://localhost:{} after timeout", check_port)).ok();
                                });
                            }
                        }
                        Err(e) => {
                            if let Some(win) = app_handle.get_webview_window("main") {
                                win.emit("log", format!("TENSORBOARD START FAILED (detached): {}", e)).ok();
                            }
                        }
                    }
                }

                // Non-Windows: spawn normal background process and capture logs
                #[cfg(not(target_os = "windows"))]
                {
                    let mut tb_cmd = TokioCommand::new(&python_exe);
                    tb_cmd
                        .arg("-m")
                        .arg("tensorboard")
                        .arg("--logdir")
                        .arg("../data/outputs")
                        .arg("--port")
                        .arg(port.to_string())
                        .arg("--host")
                        .arg("0.0.0.0")
                        .current_dir(&work_dir)
                        .stdout(Stdio::piped())
                        .stderr(Stdio::piped());

                    match tb_cmd.spawn() {
                        Ok(mut child) => {
                            let tb_stdout = child.stdout.take();
                            let tb_stderr = child.stderr.take();

                            // store child and port
                            {
                                let mut guard = python_state.tensorboard_child.lock().await;
                                *guard = Some(child);
                                let mut pguard = python_state.tensorboard_port.lock().await;
                                *pguard = Some(port);
                            }

                            // stream logs and poll as before
                            if let Some(win) = app_handle.get_webview_window("main") {
                                if let Some(out) = tb_stdout {
                                    let w = win.clone();
                                    tokio::spawn(async move {
                                        let mut reader = BufReader::new(out);
                                        let mut line = String::new();
                                        while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
                                            let trimmed = line.trim();
                                            if !trimmed.is_empty() {
                                                w.emit("log", format!("TENSORBOARD: {}", trimmed)).ok();
                                            }
                                            line.clear();
                                        }
                                    });
                                }

                                if let Some(err) = tb_stderr {
                                    let w2 = win.clone();
                                    tokio::spawn(async move {
                                        let mut reader = BufReader::new(err);
                                        let mut line = String::new();
                                        while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
                                            let trimmed = line.trim();
                                            if !trimmed.is_empty() {
                                                w2.emit("log", format!("TENSORBOARD: {}", trimmed)).ok();
                                            }
                                            line.clear();
                                        }
                                    });
                                }

                                let w3 = win.clone();
                                let check_port = port;
                                tokio::spawn(async move {
                                    let client = reqwest::Client::new();
                                    let url = format!("http://127.0.0.1:{}/", check_port);
                                    let mut attempts = 0;
                                    while attempts < 15 {
                                        if client.get(&url).send().await.is_ok() {
                                            w3.emit("log", format!("STATUS: TensorBoard available at http://localhost:{}", check_port)).ok();
                                            return;
                                        }
                                        attempts += 1;
                                        tokio::time::sleep(std::time::Duration::from_millis(1000)).await;
                                    }
                                    w3.emit("log", format!("TENSORBOARD: failed to respond on http://localhost:{} after timeout", check_port)).ok();
                                });
                            }
                        }
                        Err(e) => {
                            if let Some(win) = app_handle.get_webview_window("main") {
                                win.emit("log", format!("TENSORBOARD START FAILED: {}", e)).ok();
                            }
                        }
                    }
                }
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            // Existing commands
            save_model_config_command,
            load_model_configs_command,
            delete_model_config_command,
            list_model_folders_command,
            list_finetuning_models_command,
            list_training_projects_command,
            list_loras_by_project_command,
            download_models_command,
            take_screenshot_path_command,
            save_annotation_crop_command,
            download_datasets_command,
            list_dataset_folders_command,
            list_gguf_models_command,
            list_lora_adapters_command,
            stop_llama_server_command,
            send_chat_message_command,
            send_chat_message_streaming_command,
            clear_chat_history_command,
            get_chat_response_command,
            update_llama_server_params_command,
            start_llama_server_command,
            check_llama_binary_command,
            download_llama_binary_command,
            debug_python_path_command,
            check_llama_server_status_command,
            list_all_resources_command,
            delete_resource_command,
            import_resource_command,
            load_presets_command,
            save_preset_command,
            convert_hf_to_gguf_command,
            convert_lora_to_gguf_command,
            convert_dataset_command,
            list_hf_repo_files_command,
            download_hf_model_command,
            download_hf_dataset_command,
            cancel_download_command,
            start_training_command,
            stop_training_command,
            get_hf_token_command,
            save_hf_token_command,
            search_huggingface_command,
            export_resources_command,
            check_python_installed_command,
            setup_python_env_command,
            // Python standalone installation
            check_python_standalone_command,
            download_python_standalone_command,

        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

#[tauri::command]
async fn update_llama_server_params_command(
    window: Window,
    host: String,
    port: u16,
    temperature: f64,
    top_p: f64,
    top_k: u64,
    system_prompt: String,
    ctx_size: u64, // Also allow updating context size
) -> Result<String, String> {
    debug!("Received update_llama_server_params_command. Host: {}, Port: {}, Temp: {}, TopP: {}, TopK: {}, CtxSize: {}",
        host, port, temperature, top_p, top_k, ctx_size);

    #[derive(Serialize)]
    struct LlamaServerParamsRequest {
        temperature: f64,
        top_p: f64,
        top_k: u64,
        system_prompt: String,
        n_ctx: u64, // This matches the llama.cpp server's n_ctx parameter
    }

    let client = reqwest::Client::new();
    let url = format!("http://{}:{}/model/parameters", host, port); // Assuming a /model/parameters endpoint

    let request_body = LlamaServerParamsRequest {
        temperature,
        top_p,
        top_k,
        system_prompt,
        n_ctx: ctx_size,
    };

    window
        .emit(
            "log",
            format!("Updating Llama server parameters for {}:{}...", host, port),
        )
        .map_err(|e| e.to_string())?;

    let res = client
        .post(&url)
        .json(&request_body)
        .send()
        .await
        .map_err(|e| {
            error!("Failed to update Llama server parameters: {}", e);
            e.to_string()
        })?;

    if res.status().is_success() {
        debug!("Llama server parameters updated successfully.");
        Ok("Llama server parameters updated.".to_string())
    } else {
        let status = res.status();
        let text = res
            .text()
            .await
            .unwrap_or_else(|_| "No response text".to_string());
        error!(
            "Llama server returned an error when updating parameters: Status: {}, Response: {}",
            status, text
        );
        Err(format!(
            "Llama server parameter update error: {} - {}",
            status, text
        ))
    }
}
