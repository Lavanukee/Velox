use futures_util::StreamExt;
use log::{debug, error};
use reqwest;
use serde::{Deserialize, Serialize};
use std::io::Cursor;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use tauri::{AppHandle, Emitter, Manager, State, Window};
use tokio::fs;
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::constants::{DOWNLOAD_TASKS_FILE, HF_TOKEN_FILE, PRESETS_FILE};
use crate::hardware::{detect_backend, ComputeBackend};
use crate::models::{
    CheckpointInfo, ModelConfig, ModelsState, PersistentDownloadTask, ProjectLoraInfo,
};
use crate::python::{get_python_command, get_script_path, LlamaChatContext, PythonProcessState};
use crate::utils::{
    check_dir_for_gguf, copy_dir_all, create_hidden_command, create_hidden_std_command,
    detect_dataset_format, detect_quantization, get_data_dir, get_dir_size, get_file_size,
};

// --- Helper Functions Local to Commands ---

fn get_binaries_dir(app_handle: &AppHandle) -> PathBuf {
    get_data_dir(app_handle).join("binaries")
}

#[tauri::command]
pub async fn load_persistent_downloads_command(
    app_handle: AppHandle,
) -> Result<Vec<PersistentDownloadTask>, String> {
    let data_dir = get_data_dir(&app_handle);
    let tasks_file = data_dir.join(DOWNLOAD_TASKS_FILE);

    if !tasks_file.exists() {
        return Ok(vec![]);
    }

    let contents = fs::read_to_string(&tasks_file)
        .await
        .map_err(|e| e.to_string())?;
    let tasks: Vec<PersistentDownloadTask> =
        serde_json::from_str(&contents).unwrap_or_else(|_| vec![]);
    Ok(tasks)
}

async fn save_persistent_task(
    app_handle: &AppHandle,
    task: PersistentDownloadTask,
) -> Result<(), String> {
    let data_dir = get_data_dir(app_handle);
    let tasks_file = data_dir.join(DOWNLOAD_TASKS_FILE);

    let mut tasks = if tasks_file.exists() {
        let contents = fs::read_to_string(&tasks_file).await.unwrap_or_default();
        serde_json::from_str::<Vec<PersistentDownloadTask>>(&contents).unwrap_or_default()
    } else {
        vec![]
    };

    // Check if exists, update or add
    if let Some(existing) = tasks.iter_mut().find(|t| t.id == task.id) {
        *existing = task;
    } else {
        tasks.push(task);
    }

    let serialized = serde_json::to_string(&tasks).map_err(|e| e.to_string())?;
    fs::write(tasks_file, serialized)
        .await
        .map_err(|e| e.to_string())?;
    Ok(())
}

async fn remove_persistent_task(app_handle: &AppHandle, task_id: &str) -> Result<(), String> {
    let data_dir = get_data_dir(app_handle);
    let tasks_file = data_dir.join(DOWNLOAD_TASKS_FILE);

    if !tasks_file.exists() {
        return Ok(());
    }

    let contents = fs::read_to_string(&tasks_file).await.unwrap_or_default();
    let mut tasks =
        serde_json::from_str::<Vec<PersistentDownloadTask>>(&contents).unwrap_or_default();

    tasks.retain(|t| t.id != task_id);

    let serialized = serde_json::to_string(&tasks).map_err(|e| e.to_string())?;
    fs::write(tasks_file, serialized)
        .await
        .map_err(|e| e.to_string())?;
    Ok(())
}

// --- COMMANDS ---

// LLAMA_VERSION constant removed in favor of dynamic check
async fn fetch_latest_llama_tag() -> Result<String, String> {
    let client = reqwest::Client::new();
    let resp = client
        .get("https://api.github.com/repos/ggml-org/llama.cpp/releases/latest")
        .header("User-Agent", "velox-client")
        .send()
        .await
        .map_err(|e| format!("Network error: {}", e))?;

    if !resp.status().is_success() {
        return Err(format!("GitHub API error: {}", resp.status()));
    }

    let json: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
    json["tag_name"]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| "No tag_name in release response".to_string())
}

#[tauri::command]
pub async fn check_python_installed_command(app_handle: AppHandle) -> Result<bool, String> {
    match get_python_command(&app_handle) {
        Ok((python_exe, _)) => {
            // Verify Python runs AND critical dependencies are importable
            // We check for 'unsloth' and 'torch' as the primary indicators of a valid environment
            let output = create_hidden_command(&python_exe)
                .arg("-c")
                .arg("import unsloth; import torch; print('Backend Ready')")
                .output()
                .await
                .map_err(|e| e.to_string())?;

            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                Ok(stdout.contains("Backend Ready"))
            } else {
                debug!(
                    "Python dependency check failed: {:?}",
                    String::from_utf8_lossy(&output.stderr)
                );
                Ok(false)
            }
        }
        Err(_) => Ok(false),
    }
}

#[tauri::command]
pub async fn check_python_minimal_command(app_handle: AppHandle) -> Result<bool, String> {
    match get_python_command(&app_handle) {
        Ok((python_exe, _)) => {
            // Fast check: Just ensure Python executable works
            let output = create_hidden_command(&python_exe)
                .arg("-c")
                .arg("print('Minimal Ready')")
                .output()
                .await
                .map_err(|e| e.to_string())?;

            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                Ok(stdout.contains("Minimal Ready"))
            } else {
                Ok(false)
            }
        }
        Err(_) => Ok(false),
    }
}

#[tauri::command]
pub async fn check_llama_binary_command(app_handle: AppHandle) -> Result<bool, String> {
    let bin_dir = get_binaries_dir(&app_handle);
    debug!("Binaries directory: {:?}", bin_dir);
    let name = if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    };
    let path = bin_dir.join(name);
    debug!("Checking for llama binary at: {:?}", path);

    let mut exists = path.exists();

    // Check version
    let version_file = bin_dir.join("version.txt");
    if exists {
        // Try to fetch latest version from GitHub
        match fetch_latest_llama_tag().await {
            Ok(latest_tag) => {
                if let Ok(installed_version) = tokio::fs::read_to_string(&version_file).await {
                    let installed = installed_version.trim();
                    if installed != latest_tag {
                        debug!(
                            "Update available. Installed: {}, Latest: {}. Triggering update.",
                            installed, latest_tag
                        );
                        exists = false;
                    } else {
                        debug!("Llama binary is up to date: {}", latest_tag);
                    }
                } else {
                    debug!("version.txt missing, triggering update to {}", latest_tag);
                    exists = false;
                }
            }
            Err(e) => {
                debug!(
                    "Failed to check for updates: {}. Skipping version check.",
                    e
                );
                // If we can't check online, but binary exists, assume it's usable.
            }
        }
    }

    // On Windows with CUDA, verify the DLL exists too
    if exists && cfg!(windows) {
        if let ComputeBackend::Cuda = detect_backend() {
            let cuda_dll = bin_dir.join("ggml-cuda.dll");
            if !cuda_dll.exists() {
                debug!(
                    "Llama binary exists but ggml-cuda.dll is missing at {:?}. Triggering update.",
                    cuda_dll
                );
                exists = false;
            }
        }
    }

    debug!("Llama binary valid: {}", exists);
    Ok(exists)
}

#[tauri::command]
pub async fn check_llama_binary_exists_command(app_handle: AppHandle) -> Result<bool, String> {
    let bin_dir = get_binaries_dir(&app_handle);
    let name = if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    };
    let path = bin_dir.join(name);
    Ok(path.exists())
}

#[tauri::command]
pub async fn download_llama_binary_command(
    window: Window,
    app_handle: AppHandle,
) -> Result<String, String> {
    let backend = detect_backend();
    let msg = format!("Detected compute backend: {:?}", backend);
    debug!("{}", msg);
    window.emit("log", msg).ok();

    let (main_keywords, dep_keywords): (Vec<&str>, Vec<&str>) =
        match (std::env::consts::OS, &backend) {
            ("windows", ComputeBackend::Cuda) => {
                (vec!["cuda", "win", "x64"], vec!["cudart", "win"])
            }
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

    let client = reqwest::Client::new();

    // Fetch dynamic latest version
    let latest_tag = fetch_latest_llama_tag()
        .await
        .map_err(|e| format!("Failed to resolve latest version: {}", e))?;
    let msg = format!("Resolving latest engine version: {}", latest_tag);
    debug!("{}", msg);
    window.emit("log", msg).ok();

    let release_url = format!(
        "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{}",
        latest_tag
    );

    let resp = client
        .get(&release_url)
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

    let find_asset_url = |keywords: &Vec<&str>, exclude: Option<&str>| -> Option<String> {
        assets
            .iter()
            .find(|a| {
                let name = a["name"].as_str().unwrap_or("");
                let matches_keywords =
                    name.ends_with(".zip") && keywords.iter().all(|k| name.contains(k));
                let not_excluded = match exclude {
                    Some(ex) => !name.contains(ex),
                    None => true,
                };
                matches_keywords && not_excluded
            })
            .and_then(|a| a["browser_download_url"].as_str().map(|s| s.to_string()))
    };

    let exclude_filter = if cfg!(windows) && matches!(backend, ComputeBackend::Cuda) {
        Some("cudart")
    } else {
        None
    };

    let main_url = match find_asset_url(&main_keywords, exclude_filter) {
        Some(url) => url,
        None => {
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

    let bin_dir = get_binaries_dir(&app_handle);
    if !bin_dir.exists() {
        fs::create_dir_all(&bin_dir)
            .await
            .map_err(|e| e.to_string())?;
    }

    let downloads = if let Some(dep) = dep_url {
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

        let total_size = response.content_length().unwrap_or(0);
        let mut downloaded: u64 = 0;
        let mut buffer = Vec::new();

        let mut stream = response.bytes_stream();
        while let Some(item) = stream.next().await {
            let chunk = item.map_err(|e| format!("Error while downloading: {}", e))?;
            buffer.extend_from_slice(&chunk);
            downloaded += chunk.len() as u64;

            // Emit progress
            if total_size > 0 {
                let progress = (downloaded as f64 / total_size as f64 * 100.0) as u8;
                window
                    .emit(
                        "setup_progress",
                        serde_json::json!({
                            "step": "download",
                            "message": format!("Downloading {}...", label),
                            "progress": progress,
                            "loaded": downloaded,
                            "total": total_size
                        }),
                    )
                    .ok();
            }
        }

        let content = bytes::Bytes::from(buffer);

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

    let server_path = bin_dir.join(if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    });
    if server_path.exists() {
        // Save version file
        let version_file = bin_dir.join("version.txt");
        if let Err(e) = tokio::fs::write(&version_file, &latest_tag).await {
            window
                .emit(
                    "log",
                    format!("Warning: Failed to save version file: {}", e),
                )
                .unwrap();
        }

        window
            .emit("log", "Engine installed successfully!".to_string())
            .unwrap();
        Ok("Installation complete.".into())
    } else {
        Err("Installation finished, but llama-server binary is missing.".into())
    }
}

#[tauri::command]
pub async fn save_project_config_command(
    app_handle: AppHandle,
    project_name: String,
    config: serde_json::Value,
) -> Result<String, String> {
    let data_dir = get_data_dir(&app_handle);
    let project_dir = data_dir.join("data").join("outputs").join(&project_name);

    if !project_dir.exists() {
        tokio::fs::create_dir_all(&project_dir)
            .await
            .map_err(|e| format!("Failed to create project directory: {}", e))?;
    }

    let config_path = project_dir.join("training_config.json");
    let json_str = serde_json::to_string_pretty(&config)
        .map_err(|e| format!("Failed to serialize config: {}", e))?;

    tokio::fs::write(&config_path, json_str)
        .await
        .map_err(|e| format!("Failed to write config file: {}", e))?;

    Ok("Config saved".to_string())
}

#[tauri::command]
pub async fn load_project_config_command(
    app_handle: AppHandle,
    project_name: String,
) -> Result<serde_json::Value, String> {
    let data_dir = get_data_dir(&app_handle);
    let config_path = data_dir
        .join("data")
        .join("outputs")
        .join(&project_name)
        .join("training_config.json");

    if !config_path.exists() {
        return Ok(serde_json::json!({}));
    }

    let content = tokio::fs::read_to_string(&config_path)
        .await
        .map_err(|e| format!("Failed to read config file: {}", e))?;

    let json: serde_json::Value = serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse config file: {}", e))?;

    Ok(json)
}

#[tauri::command]
pub async fn save_model_config_command(
    models_state: State<'_, Arc<ModelsState>>,
    model_config: ModelConfig,
) -> Result<String, String> {
    debug!(
        "Attempting to save model config for ID: {}",
        model_config.id
    );
    let mut configs = models_state.configs.lock().await;

    let src_tauri_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let project_root_dir = src_tauri_dir
        .parent()
        .expect("Failed to get project root")
        .to_path_buf();
    let lora_base_dir = project_root_dir.join("data").join("loras");

    let mut processed_model_config = model_config.clone();
    if !processed_model_config.lora_path.is_empty() {
        let current_lora_path = PathBuf::from(&processed_model_config.lora_path);
        if current_lora_path.is_absolute() {
            if let Ok(relative_path) = current_lora_path.strip_prefix(&lora_base_dir) {
                processed_model_config.lora_path = relative_path.to_string_lossy().to_string();
            } else {
                error!(
                    "Absolute lora_path '{}' is not within the expected data/loras directory.",
                    processed_model_config.lora_path
                );
                processed_model_config.lora_path = "".to_string();
            }
        }
    }

    let models_base_dir = project_root_dir.join("data").join("models");
    if !processed_model_config.llama_model_path.is_empty() {
        let current_model_path = PathBuf::from(&processed_model_config.llama_model_path);
        if current_model_path.is_absolute() {
            if let Ok(relative_path) = current_model_path.strip_prefix(&models_base_dir) {
                processed_model_config.llama_model_path =
                    relative_path.to_string_lossy().to_string();
            }
        }
    }

    configs.insert(processed_model_config.id.clone(), processed_model_config);
    ModelsState::save_to_disk(&configs).await?;
    debug!("Model config saved successfully.");
    Ok("Saved.".to_string())
}

#[tauri::command]
pub async fn load_model_configs_command(
    models_state: State<'_, Arc<ModelsState>>,
) -> Result<Vec<ModelConfig>, String> {
    debug!("Loading model configurations.");
    let configs = models_state.configs.lock().await;
    Ok(configs.values().cloned().collect())
}

#[tauri::command]
pub async fn delete_model_config_command(
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
pub async fn list_model_folders_command(app_handle: AppHandle) -> Result<Vec<String>, String> {
    debug!("Listing model folders.");
    let data_dir = get_data_dir(&app_handle);
    let models_dir = data_dir.join("data").join("models");
    let mut folders = Vec::new();
    if models_dir.exists() && models_dir.is_dir() {
        if let Ok(mut entries) = fs::read_dir(models_dir).await {
            while let Ok(Some(entry)) = entries.next_entry().await {
                if entry
                    .file_type()
                    .await
                    .ok()
                    .map(|ft| ft.is_dir())
                    .unwrap_or(false)
                {
                    folders.push(entry.file_name().to_string_lossy().to_string());
                }
            }
        }
    }
    Ok(folders)
}

#[tauri::command]
pub async fn list_dataset_folders_command(app_handle: AppHandle) -> Result<Vec<String>, String> {
    debug!("Listing dataset folders/files.");
    let data_dir = get_data_dir(&app_handle);
    let datasets_dir = data_dir.join("data").join("datasets");
    let mut resources = Vec::new();

    if datasets_dir.exists() && datasets_dir.is_dir() {
        if let Ok(mut entries) = fs::read_dir(datasets_dir).await {
            while let Ok(Some(entry)) = entries.next_entry().await {
                let file_name = entry.file_name().to_string_lossy().to_string();
                let path = entry.path();

                let is_dir = entry
                    .file_type()
                    .await
                    .ok()
                    .map(|ft| ft.is_dir())
                    .unwrap_or(false);
                let ext = path.extension().map(|e| e.to_string_lossy().to_lowercase());

                // Include directories
                if is_dir {
                    resources.push(file_name);
                }
                // Include specific dataset files
                else if let Some(e) = ext {
                    if ["jsonl", "json", "csv", "parquet", "arrow"].contains(&e.as_str()) {
                        resources.push(file_name);
                    }
                }
            }
        }
    }
    Ok(resources)
}

#[tauri::command]
pub async fn list_finetuning_models_command(app_handle: AppHandle) -> Result<Vec<String>, String> {
    debug!("Listing models suitable for fine-tuning.");
    let data_dir = get_data_dir(&app_handle);
    let models_dir = data_dir.join("data").join("models");
    let mut models = Vec::new();

    if models_dir.exists() && models_dir.is_dir() {
        if let Ok(mut entries) = fs::read_dir(models_dir).await {
            while let Ok(Some(entry)) = entries.next_entry().await {
                if entry
                    .file_type()
                    .await
                    .ok()
                    .map(|ft| ft.is_dir())
                    .unwrap_or(false)
                {
                    let folder_name = entry.file_name().to_string_lossy().to_string();
                    models.push(folder_name);
                }
            }
        }
    }
    Ok(models)
}

/// Reads the config.json from a model directory to get metadata for resource estimation.
#[tauri::command]
pub async fn get_model_config_command(
    app_handle: AppHandle,
    model_path: String,
) -> Result<serde_json::Value, String> {
    let data_dir = get_data_dir(&app_handle);

    // Resolve path (could be absolute or relative)
    let resolved_path = if PathBuf::from(&model_path).is_absolute() {
        PathBuf::from(&model_path)
    } else {
        data_dir.join("data").join("models").join(&model_path)
    };

    let config_path = resolved_path.join("config.json");

    if !config_path.exists() {
        return Err(format!("config.json not found at {:?}", config_path));
    }

    let content = fs::read_to_string(&config_path)
        .await
        .map_err(|e| format!("Failed to read config.json: {}", e))?;

    let json: serde_json::Value = serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse config.json: {}", e))?;

    Ok(json)
}

#[tauri::command]
pub async fn list_training_projects_command(app_handle: AppHandle) -> Result<Vec<String>, String> {
    debug!("Listing existing training projects.");
    let data_dir = get_data_dir(&app_handle);
    let outputs_dir = data_dir.join("data").join("outputs");
    let mut projects = Vec::new();

    if outputs_dir.exists() && outputs_dir.is_dir() {
        if let Ok(mut entries) = fs::read_dir(outputs_dir).await {
            while let Ok(Some(entry)) = entries.next_entry().await {
                if entry
                    .file_type()
                    .await
                    .ok()
                    .map(|ft| ft.is_dir())
                    .unwrap_or(false)
                {
                    let folder_name = entry.file_name().to_string_lossy().to_string();
                    if !folder_name.starts_with('.') {
                        projects.push(folder_name);
                    }
                }
            }
        }
    }
    projects.sort();
    Ok(projects)
}

#[tauri::command]
pub async fn list_loras_by_project_command(
    app_handle: AppHandle,
) -> Result<Vec<ProjectLoraInfo>, String> {
    debug!("Listing LoRAs organized by training project.");
    let data_dir = get_data_dir(&app_handle);
    let outputs_dir = data_dir.join("data").join("outputs");
    let mut projects_info = Vec::new();

    if outputs_dir.exists() && outputs_dir.is_dir() {
        if let Ok(mut entries) = fs::read_dir(&outputs_dir).await {
            while let Ok(Some(entry)) = entries.next_entry().await {
                if entry
                    .file_type()
                    .await
                    .ok()
                    .map(|ft| ft.is_dir())
                    .unwrap_or(false)
                {
                    let project_name = entry.file_name().to_string_lossy().to_string();
                    let project_path = entry.path();

                    // Read project_metadata.json if it exists
                    let base_model_from_metadata = {
                        let metadata_path = project_path.join("project_metadata.json");
                        if metadata_path.exists() {
                            if let Ok(c) = fs::read_to_string(metadata_path).await {
                                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&c) {
                                    json.get("base_model")
                                        .and_then(|v| v.as_str())
                                        .map(|s| s.to_string())
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    };

                    let mut checkpoints = Vec::new();

                    if let Ok(mut sub_entries) = fs::read_dir(&project_path).await {
                        while let Ok(Some(sub)) = sub_entries.next_entry().await {
                            if sub
                                .file_type()
                                .await
                                .ok()
                                .map(|ft| ft.is_dir())
                                .unwrap_or(false)
                            {
                                let sub_name = sub.file_name().to_string_lossy().to_string();
                                if sub_name == "final_model" || sub_name.starts_with("checkpoint-")
                                {
                                    let mut step_number = None;
                                    if sub_name.starts_with("checkpoint-") {
                                        if let Ok(num) =
                                            sub_name.replace("checkpoint-", "").parse::<i32>()
                                        {
                                            step_number = Some(num);
                                        }
                                    }

                                    let gguf_path = {
                                        let mut found = None;
                                        // Look for common patterns: checkpoint.q8_0.gguf, checkpoint.adapter.q8_0.gguf, etc.
                                        if let Ok(mut project_sub) =
                                            fs::read_dir(&project_path).await
                                        {
                                            while let Ok(Some(file_entry)) =
                                                project_sub.next_entry().await
                                            {
                                                let file_name = file_entry
                                                    .file_name()
                                                    .to_string_lossy()
                                                    .to_string();
                                                if file_name.starts_with(&sub_name)
                                                    && file_name.ends_with(".gguf")
                                                {
                                                    found = Some(format!(
                                                        "data/outputs/{}/{}",
                                                        project_name, file_name
                                                    ));
                                                    break;
                                                }
                                            }
                                        }
                                        found
                                    };

                                    checkpoints.push(CheckpointInfo {
                                        name: sub_name,
                                        path: format!(
                                            "data/outputs/{}/{}",
                                            project_name,
                                            sub.file_name().to_string_lossy()
                                        ),
                                        is_final: entry.file_name().to_string_lossy()
                                            == "final_model",
                                        step_number,
                                        base_model_name: {
                                            let config_path =
                                                sub.path().join("adapter_config.json");
                                            if config_path.exists() {
                                                if let Ok(c) = std::fs::read_to_string(config_path)
                                                {
                                                    if let Ok(json) =
                                                        serde_json::from_str::<serde_json::Value>(
                                                            &c,
                                                        )
                                                    {
                                                        json.get("base_model_name_or_path")
                                                            .and_then(|v| v.as_str())
                                                            .map(|s| s.to_string())
                                                    } else {
                                                        None
                                                    }
                                                } else {
                                                    None
                                                }
                                            } else {
                                                None
                                            }
                                        },
                                        gguf_path,
                                    });
                                }
                            }
                        }
                    }

                    if !checkpoints.is_empty() {
                        checkpoints.sort_by(|a, b| {
                            let a_val = if a.name == "final_model" {
                                i32::MAX
                            } else {
                                a.step_number.unwrap_or(0)
                            };
                            let b_val = if b.name == "final_model" {
                                i32::MAX
                            } else {
                                b.step_number.unwrap_or(0)
                            };
                            b_val.cmp(&a_val)
                        });

                        projects_info.push(ProjectLoraInfo {
                            project_name,
                            checkpoints,
                            base_model: base_model_from_metadata,
                        });
                    }
                }
            }
        }
    }

    // Also include manually placed LoRAs in data/loras as a "Manual/External" project
    let loras_dir = data_dir.join("data").join("loras");
    if loras_dir.exists() && loras_dir.is_dir() {
        let mut manual_checkpoints = Vec::new();
        if let Ok(mut entries) = fs::read_dir(&loras_dir).await {
            while let Ok(Some(entry)) = entries.next_entry().await {
                let path = entry.path();
                if path.is_file() || path.is_dir() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    manual_checkpoints.push(CheckpointInfo {
                        name: name.clone(),
                        path: format!("data/loras/{}", name),
                        is_final: false,
                        step_number: None,
                        base_model_name: {
                            let config_path = path.join("adapter_config.json");
                            if config_path.exists() {
                                if let Ok(c) = std::fs::read_to_string(config_path) {
                                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(&c)
                                    {
                                        json.get("base_model_name_or_path")
                                            .and_then(|v| v.as_str())
                                            .map(|s| s.to_string())
                                    } else {
                                        None
                                    }
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        },
                        gguf_path: {
                            let mut found = None;
                            let lora_file_name = entry.file_name().to_string_lossy().to_string();
                            if let Ok(mut lora_dir_search) = fs::read_dir(&loras_dir).await {
                                while let Ok(Some(fe)) = lora_dir_search.next_entry().await {
                                    let fn_str = fe.file_name().to_string_lossy().to_string();
                                    if fn_str.starts_with(&lora_file_name)
                                        && fn_str.ends_with(".gguf")
                                    {
                                        found = Some(format!("data/loras/{}", fn_str));
                                        break;
                                    }
                                }
                            }
                            found
                        },
                    });
                }
            }
        }
        if !manual_checkpoints.is_empty() {
            projects_info.push(ProjectLoraInfo {
                project_name: "Manual/External".to_string(),
                checkpoints: manual_checkpoints,
                base_model: None,
            });
        }
    }

    Ok(projects_info)
}

#[tauri::command]
pub async fn start_llama_server_command(
    window: Window,
    python_process_state: State<'_, Arc<PythonProcessState>>,
    app_handle: AppHandle,
    model_path: String,
    port: u16,
    gpu_layers: Option<u32>,
    ctx_size: u32,
    batch_size: u32,
    ubatch_size: u32,
    threads: Option<u32>,
    flash_attn: bool,
    no_mmap: bool,
    mmproj_path: String,
    lora_path: Option<String>,
    slot_id: Option<u8>, // 0=Primary, 1=Secondary
) -> Result<String, String> {
    let slot = slot_id.unwrap_or(0);

    // Kill existing if any using the correct slot
    {
        let mut child_guard = if slot == 1 {
            python_process_state.llama_secondary_child.lock().await
        } else {
            python_process_state.llama_server_child.lock().await
        };

        if let Some(child) = child_guard.as_mut() {
            let _ = child.kill().await;
            *child_guard = None;
        }
    }

    let bin_dir = get_binaries_dir(&app_handle);
    let llama_server_name = if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    };
    let llama_server_path = bin_dir.join(llama_server_name);

    if !llama_server_path.exists() {
        return Err("Llama server binary not found. Please download it first.".to_string());
    }

    let resolved_model_path = {
        let p = PathBuf::from(&model_path);
        if p.is_absolute() {
            p
        } else {
            let data_dir = get_data_dir(&app_handle);
            // If the path already starts with "data/" or looks like a relative path from app data dir,
            // join it directly. Otherwise, default to data/models.
            if model_path.starts_with("data") || model_path.contains("data\\") {
                data_dir.join(&model_path)
            } else {
                data_dir.join("data/models").join(&model_path)
            }
        }
    };

    // Check for chat template files and attempt fallback download if missing
    let model_parent_dir = if resolved_model_path.is_file() {
        resolved_model_path
            .parent()
            .unwrap_or(&bin_dir)
            .to_path_buf()
    } else {
        resolved_model_path.clone()
    };

    // Attempt to extract repo_id from folder name (e.g. "author--repo")
    // If user renamed folder, this might fail, which is fine (we just skip fallback)
    if let Some(folder_name) = model_parent_dir.file_name() {
        let folder_str = folder_name.to_string_lossy();
        if folder_str.contains("--") {
            // Simple heuristic for HF folders
            let repo_id = folder_str.replace("--", "/");
            let files_to_check = vec![
                "tokenizer_config.json",
                "special_tokens_map.json",
                "chat_template.json",
                "tokenizer.json",
            ];

            let missing_files: Vec<String> = files_to_check
                .iter()
                .filter(|&&f| !model_parent_dir.join(f).exists())
                .map(|&f| f.to_string())
                .collect();

            if !missing_files.is_empty() {
                window.emit("log", format!("Missing chat template files: {:?}. Attempting auto-download from {}...", missing_files, repo_id)).ok();

                let (python_exe, work_dir) = get_python_command(&app_handle).unwrap_or_default();
                if !python_exe.is_empty() {
                    let script = get_script_path(&app_handle, "huggingface_manager.py");
                    // We run this synchronously to ensure they exist before server start
                    // We use the same 'huggingface_manager.py' as the download command
                    let mut cmd = create_hidden_command(&python_exe);
                    cmd.arg(&script)
                        .arg("download")
                        .arg("--repo_id")
                        .arg(&repo_id)
                        .arg("--output")
                        .arg(model_parent_dir.to_string_lossy().to_string())
                        .arg("--files")
                        .arg(missing_files.join(","));

                    cmd.current_dir(&work_dir);

                    match cmd.output().await {
                        Ok(o) => {
                            if o.status.success() {
                                window
                                    .emit(
                                        "log",
                                        "Auto-download of chat templates successful.".to_string(),
                                    )
                                    .ok();
                            } else {
                                let err_msg = String::from_utf8_lossy(&o.stderr);
                                window
                                    .emit(
                                        "log",
                                        format!("Auto-download failed (non-critical): {}", err_msg),
                                    )
                                    .ok();
                            }
                        }
                        Err(e) => {
                            window
                                .emit("log", format!("Failed to run auto-download command: {}", e))
                                .ok();
                        }
                    }
                }
            }
        }
    }

    // Construct args
    let mut args = vec![
        "--model".to_string(),
        resolved_model_path.to_string_lossy().to_string(),
        "--port".to_string(),
        port.to_string(),
        "--ctx-size".to_string(),
        ctx_size.to_string(),
        "--batch-size".to_string(),
        batch_size.to_string(),
        "--ubatch-size".to_string(),
        ubatch_size.to_string(),
        "-ngl".to_string(),
        // User requested to ignore 'fit' flag and default to GPU.
        // If gpu_layers is 0 or user passed auto-fit (which passes None/0), we default to offloading ALL layers (using a high number like 300).
        // This ensures -ngl is always passed.
        match gpu_layers {
            Some(n) if n > 0 => n.to_string(),
            _ => "999".to_string(), // Default to max offload (increased to 999 for safety)
        },
        "--host".to_string(),
        "0.0.0.0".to_string(),
    ];

    // Log the constructed arguments for debugging
    log::info!("Starting Llama Server with args: {:?}", args);

    // Auto-detect mmproj if not provided
    let mut effective_mmproj_path = mmproj_path.clone();
    if effective_mmproj_path.is_empty() {
        let path_obj = PathBuf::from(&model_path);
        if let Some(parent) = path_obj.parent() {
            if let Ok(mut entries) = fs::read_dir(parent).await {
                while let Ok(Some(entry)) = entries.next_entry().await {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if name.contains("mmproj") && name.ends_with(".gguf") {
                        effective_mmproj_path = entry.path().to_string_lossy().to_string();
                        debug!("Auto-detected mmproj: {}", effective_mmproj_path);
                        break;
                    }
                }
            }
        }
    }

    if !effective_mmproj_path.is_empty() {
        args.push("--mmproj".to_string());
        args.push(effective_mmproj_path);
    }
    if let Some(lp) = lora_path {
        if !lp.is_empty() {
            args.push("--lora".to_string());
            args.push(lp);
        }
    }
    if flash_attn {
        args.push("--flash-attn".to_string());
        args.push("on".to_string());
    }
    if no_mmap {
        args.push("--no-mmap".to_string());
    }
    if let Some(t) = threads {
        args.push("--threads".to_string());
        args.push(t.to_string());
    }

    // Stop sequences are handled at the request level in the chat commands
    // to ensure compatibility with different llama-server versions that may not support --stop or --stop-sequence CLI flags.

    let mut child_guard = if slot == 1 {
        python_process_state.llama_secondary_child.lock().await
    } else {
        python_process_state.llama_server_child.lock().await
    };

    // Windows usually needs hidden command, but for server we want to capture stdout/stderr manually.
    let mut cmd = if cfg!(windows) {
        let mut c = tokio::process::Command::new(&llama_server_path);
        // use std::os::windows::process::CommandExt; // Unused import warning fix
        c.creation_flags(0x08000000);
        c
    } else {
        tokio::process::Command::new(&llama_server_path)
    };

    cmd.args(args).stdout(Stdio::piped()).stderr(Stdio::piped());

    let mut child = cmd
        .spawn()
        .map_err(|e| format!("Failed to spawn Llama server: {}", e))?;
    let stdout = child.stdout.take().ok_or("No stdout for llama server")?;
    let stderr = child.stderr.take().ok_or("No stderr for llama server")?;

    let win_c = window.clone();
    tokio::spawn(async move {
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();
        while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
            if !line.trim().is_empty() {
                let prefix = if slot == 1 {
                    "LLAMA_SERVER_SEC"
                } else {
                    "LLAMA_SERVER"
                };
                win_c
                    .emit("log", format!("{}: {}", prefix, line.trim()))
                    .unwrap_or(());
            }
            line.clear();
        }
    });

    // We can also log stderr.
    let win_c_err = window.clone();
    tokio::spawn(async move {
        let mut reader = BufReader::new(stderr);
        let mut line = String::new();
        while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
            if !line.trim().is_empty() {
                win_c_err
                    .emit("log", format!("LLAMA_SERVER_ERR: {}", line.trim()))
                    .unwrap_or(());
            }
            line.clear();
        }
    });

    *child_guard = Some(child);
    Ok("Llama server started.".to_string())
}

#[tauri::command]
pub async fn stop_llama_server_command(
    python_process_state: State<'_, Arc<PythonProcessState>>,
    slot_id: Option<u8>,
) -> Result<String, String> {
    let slot = slot_id.unwrap_or(0);
    let mut child_guard = if slot == 1 {
        python_process_state.llama_secondary_child.lock().await
    } else {
        python_process_state.llama_server_child.lock().await
    };

    if let Some(child) = child_guard.as_mut() {
        match child.kill().await {
            Ok(_) => {
                *child_guard = None;
                Ok(format!("Llama server (slot {}) stopped.", slot))
            }
            Err(e) => Err(format!(
                "Failed to stop Llama server (slot {}): {}",
                slot, e
            )),
        }
    } else {
        Err(format!("Llama server (slot {}) is not running.", slot).into())
    }
}

#[tauri::command]
pub async fn check_llama_server_status_command(
    python_process_state: State<'_, Arc<PythonProcessState>>,
    slot_id: Option<u8>,
) -> Result<bool, String> {
    let slot = slot_id.unwrap_or(0);
    let child_guard = if slot == 1 {
        python_process_state.llama_secondary_child.lock().await
    } else {
        python_process_state.llama_server_child.lock().await
    };
    Ok(child_guard.is_some())
}

#[tauri::command]
pub async fn start_transformers_server_command(
    window: Window,
    app_handle: AppHandle,
    python_process_state: State<'_, Arc<PythonProcessState>>,
    model_path: String,
    port: u16,
) -> Result<String, String> {
    // Kill existing if any
    {
        let mut child_guard = python_process_state.transformers_child.lock().await;
        if let Some(child) = child_guard.as_mut() {
            let _ = child.kill().await;
            *child_guard = None;
        }
    }

    let (python_exe, work_dir) = get_python_command(&app_handle)?;
    let script = get_script_path(&app_handle, "transformers_server.py");

    // Check if model path is absolute, if not try to find it in data/models or data/outputs
    let resolved_model_path = {
        let p = PathBuf::from(&model_path);
        if p.is_absolute() {
            model_path
        } else {
            let data_dir = get_data_dir(&app_handle);
            if model_path.starts_with("data") || model_path.contains("data\\") {
                data_dir.join(&model_path).to_string_lossy().to_string()
            } else {
                data_dir
                    .join("data/models")
                    .join(&model_path)
                    .to_string_lossy()
                    .to_string()
            }
        }
    };

    let mut cmd = create_hidden_command(&python_exe);
    #[cfg(target_os = "windows")]
    {
        // use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x08000000);
    }

    cmd.arg(&script)
        .arg("--model")
        .arg(&resolved_model_path)
        .arg("--port")
        .arg(port.to_string())
        .arg("--host")
        .arg("0.0.0.0")
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
                .emit("log", format!("TF_SERVER: {}", line.trim()))
                .ok();
            line.clear();
        }
    });

    let win_c2 = window.clone();
    tokio::spawn(async move {
        let mut reader = BufReader::new(stderr);
        let mut line = String::new();
        while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
            win_c2
                .emit("log", format!("TF_SERVER_ERR: {}", line.trim()))
                .ok();
            line.clear();
        }
    });

    let mut child_guard = python_process_state.transformers_child.lock().await;
    *child_guard = Some(child);

    Ok("Transformers server started".to_string())
}

#[tauri::command]
pub async fn stop_transformers_server_command(
    python_process_state: State<'_, Arc<PythonProcessState>>,
) -> Result<String, String> {
    let mut child_guard = python_process_state.transformers_child.lock().await;
    if let Some(child) = child_guard.as_mut() {
        let _ = child.kill().await;
        *child_guard = None;
        Ok("Transformers server stopped".to_string())
    } else {
        Err("Transformers server is not running".to_string())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceInfo {
    pub name: String,
    pub size: String,
    pub path: String,
    pub r#type: String, // "model", "gguf", "lora", "dataset"
    pub quantization: Option<String>,
    pub is_mmproj: bool,
    pub is_processed: Option<bool>,
    pub dataset_format: Option<String>,
    pub format_error: Option<String>, // Error if format detection failed
    pub count: Option<u64>,
    pub modalities: Option<Vec<String>>,
    pub base_model: Option<String>,
}

#[tauri::command]
pub async fn list_all_resources_command(
    app_handle: AppHandle,
) -> Result<Vec<ResourceInfo>, String> {
    let mut resources = Vec::new();
    let data_dir = get_data_dir(&app_handle);

    // HF Models
    let models_path = data_dir.join("data").join("models");
    if let Ok(mut entries) = fs::read_dir(&models_path).await {
        while let Ok(Some(entry)) = entries.next_entry().await {
            if entry
                .file_type()
                .await
                .ok()
                .map(|ft| ft.is_dir())
                .unwrap_or(false)
            {
                let name = entry.file_name().to_string_lossy().to_string();
                let path_obj = entry.path();
                let has_gguf = check_dir_for_gguf(&path_obj).await;

                if !has_gguf {
                    resources.push(ResourceInfo {
                        name: name.clone(),
                        size: get_dir_size(&path_obj).await.unwrap_or("Unknown".into()),
                        path: format!("data/models/{}", name),
                        r#type: "model".to_string(),
                        quantization: None,
                        is_mmproj: false,
                        is_processed: None,
                        dataset_format: None,
                        format_error: None,
                        count: None,
                        modalities: None,
                        base_model: None,
                    });
                } else {
                    if let Ok(mut sub) = fs::read_dir(&path_obj).await {
                        while let Ok(Some(gguf_entry)) = sub.next_entry().await {
                            let gguf_path = gguf_entry.path();
                            if gguf_path.extension().map_or(false, |e| e == "gguf") {
                                let gguf_name =
                                    gguf_entry.file_name().to_string_lossy().to_string();
                                if gguf_name.contains("mmproj") {
                                    // Skip showing mmproj files separately as per user request
                                    continue;
                                }
                                resources.push(ResourceInfo {
                                    name: gguf_name.clone(),
                                    size: get_file_size(&gguf_path)
                                        .await
                                        .unwrap_or("Unknown".into()),
                                    path: format!("data/models/{}/{}", name, gguf_name),
                                    r#type: "gguf".to_string(),
                                    quantization: Some(detect_quantization(&gguf_name)),
                                    is_mmproj: false,
                                    is_processed: None,
                                    dataset_format: None,
                                    format_error: None,
                                    count: None,
                                    modalities: None,
                                    base_model: None,
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    // GGUFs in data/outputs (Converted LoRAs or Merged models)
    let outputs_path = data_dir.join("data").join("outputs");
    if outputs_path.exists() {
        if let Ok(mut project_dirs) = fs::read_dir(&outputs_path).await {
            while let Ok(Some(project_entry)) = project_dirs.next_entry().await {
                if project_entry
                    .file_type()
                    .await
                    .map_or(false, |ft| ft.is_dir())
                {
                    let project_name = project_entry.file_name().to_string_lossy().to_string();
                    let project_dir = project_entry.path();

                    if let Ok(mut files) = fs::read_dir(&project_dir).await {
                        while let Ok(Some(file_entry)) = files.next_entry().await {
                            let file_path = file_entry.path();
                            if file_path.extension().map_or(false, |e| e == "gguf") {
                                let file_name =
                                    file_entry.file_name().to_string_lossy().to_string();
                                if file_name.contains("mmproj") {
                                    continue;
                                }

                                resources.push(ResourceInfo {
                                    name: format!("{}: {}", project_name, file_name),
                                    size: get_file_size(&file_path)
                                        .await
                                        .unwrap_or("Unknown".into()),
                                    path: format!("data/outputs/{}/{}", project_name, file_name),
                                    r#type: "gguf".to_string(),
                                    quantization: Some(detect_quantization(&file_name)),
                                    is_mmproj: false,
                                    is_processed: None,
                                    dataset_format: None,
                                    format_error: None,
                                    count: None,
                                    modalities: None,
                                    base_model: None,
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    // Datasets
    let datasets_path = data_dir.join("data").join("datasets");
    println!("BACKEND: scanning datasets at {:?}", datasets_path);
    if let Ok(mut entries) = fs::read_dir(&datasets_path).await {
        while let Ok(Some(entry)) = entries.next_entry().await {
            if entry
                .file_type()
                .await
                .ok()
                .map(|ft| ft.is_dir())
                .unwrap_or(false)
            {
                let name = entry.file_name().to_string_lossy().to_string();
                let path = entry.path();
                let is_processed = path.join("processed_data").exists();
                let mut fmt = None;
                if let Ok(mut files) = fs::read_dir(&path).await {
                    while let Ok(Some(f)) = files.next_entry().await {
                        if f.path().is_file() {
                            if let Some(d) = detect_dataset_format(&f.path()) {
                                fmt = Some(d);
                                break;
                            }
                        }
                    }
                }
                // Check for format error and metadata
                let mut format_error = None;
                let mut dataset_count: Option<u64> = None;
                let mut dataset_modalities: Option<Vec<String>> = None;

                let format_info_path = path.join("processed_data").join("format_info.json");
                if format_info_path.exists() {
                    if let Ok(content) = std::fs::read_to_string(&format_info_path) {
                        if let Ok(info) = serde_json::from_str::<serde_json::Value>(&content) {
                            if info.get("success").and_then(|v| v.as_bool()) == Some(false) {
                                format_error = info
                                    .get("error")
                                    .and_then(|v| v.as_str())
                                    .map(|s| s.to_string());
                            }
                            // Also get detected format for display
                            if let Some(detected) =
                                info.get("detected_format").and_then(|v| v.as_str())
                            {
                                if fmt.is_none() {
                                    fmt = Some(detected.to_string());
                                }
                            }
                            // Get count
                            if let Some(c) = info.get("num_examples").and_then(|v| v.as_u64()) {
                                dataset_count = Some(c);
                            }
                            // Get modalities
                            if let Some(mods) = info.get("modalities").and_then(|v| v.as_array()) {
                                let mod_strings: Vec<String> = mods
                                    .iter()
                                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                                    .collect();
                                if !mod_strings.is_empty() {
                                    dataset_modalities = Some(mod_strings);
                                }
                            }
                        }
                    }
                }

                resources.push(ResourceInfo {
                    name,
                    size: get_dir_size(&path).await.unwrap_or("Unknown".into()),
                    path: format!("data/datasets/{}", entry.file_name().to_string_lossy()),
                    r#type: "dataset".to_string(),
                    quantization: None,
                    is_mmproj: false,
                    is_processed: Some(is_processed),
                    dataset_format: fmt,
                    format_error,
                    count: dataset_count,
                    modalities: dataset_modalities,
                    base_model: None,
                });
            }
        }
    }

    // LoRAs
    let loras_path = data_dir.join("data").join("loras");
    if let Ok(mut entries) = fs::read_dir(&loras_path).await {
        while let Ok(Some(entry)) = entries.next_entry().await {
            let name = entry.file_name().to_string_lossy().to_string();
            let lora_path = entry.path();
            let mut base_model = None;

            // If it's a directory, check for adapter_config.json
            if lora_path.is_dir() {
                let config_path = lora_path.join("adapter_config.json");
                if config_path.exists() {
                    if let Ok(c) = std::fs::read_to_string(config_path) {
                        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&c) {
                            base_model = json
                                .get("base_model_name_or_path")
                                .and_then(|v| v.as_str())
                                .map(|s| s.to_string());
                        }
                    }
                }
            }

            resources.push(ResourceInfo {
                name: name.clone(),
                size: if lora_path.is_dir() {
                    get_dir_size(&lora_path).await.unwrap_or("Unknown".into())
                } else {
                    get_file_size(&lora_path).await.unwrap_or("Unknown".into())
                },
                path: format!("data/loras/{}", name),
                r#type: "lora".to_string(),
                quantization: None,
                is_mmproj: false,
                is_processed: None,
                dataset_format: None,
                format_error: None,
                count: None,
                modalities: None,
                base_model,
            });
        }
    }

    Ok(resources)
}

#[tauri::command]
pub async fn send_chat_message_command(
    _window: Window,
    llama_chat_context: State<'_, Arc<LlamaChatContext>>,
    host: String,
    port: u16,
    message: serde_json::Value,
    system_prompt: String,
    temperature: f64,
    top_p: f64,
    top_k: u64,
    _ctx_size: u64,
) -> Result<String, String> {
    debug!(
        "Received send_chat_message_command with message: {:?}",
        message
    );
    let mut chat_history = llama_chat_context.chat_history.lock().await;

    let user_message = serde_json::json!({
        "role": "user",
        "content": message
    });
    chat_history.push(user_message);

    let mut messages_for_server = Vec::new();
    if chat_history.len() == 1
        || chat_history.last().map_or(false, |m| {
            m.get("role")
                .and_then(|v| v.as_str())
                .map_or(false, |r| r == "user")
        })
    {
        let system_message = serde_json::json!({
            "role": "system",
            "content": system_prompt
        });
        messages_for_server.push(system_message);
    }
    messages_for_server.extend(chat_history.iter().cloned());

    #[derive(Serialize)]
    struct LlamaChatRequest {
        messages: Vec<serde_json::Value>,
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

    let res = client
        .post(&url)
        .json(&request_body)
        .send()
        .await
        .map_err(|e| e.to_string())?;

    if res.status().is_success() {
        let val: serde_json::Value = res.json().await.map_err(|e| e.to_string())?;
        let bot_response = val["choices"][0]["message"]["content"]
            .as_str()
            .ok_or("Failed to parse content")?
            .to_string();

        let bot_message = serde_json::json!({
            "role": "assistant",
            "content": bot_response
        });
        chat_history.push(bot_message);
        Ok(bot_response)
    } else {
        Err(format!("Llama server error: {}", res.status()))
    }
}

#[tauri::command]
pub async fn send_chat_message_streaming_command(
    window: Window,
    llama_chat_context: State<'_, Arc<LlamaChatContext>>,
    host: String,
    port: u16,
    message: serde_json::Value,
    system_prompt: String,
    temperature: f64,
    top_p: f64,
    top_k: u64,
    _ctx_size: u64,
    label: Option<String>,
    full_history: Option<Vec<serde_json::Value>>,
) -> Result<String, String> {
    debug!("Streaming chat message (label: {:?}): {:?}", label, message);

    let mut chat_history = llama_chat_context.chat_history.lock().await;

    // If full history is provided, replace existing history (stateless mode for frontend)
    if let Some(history) = full_history {
        *chat_history = history;
        // User message is assumed to be in the history or we don't add it separately?
        // Wait, the frontend might send HISTORY + NEW MESSAGE separately?
        // Logic: If full_history is passed, we assume it INCLUDES the latest message, OR we append `message` to it?
        // Let's assume full_history is everything UP TO the new message, and we append `message`.
        // OR filtering: Frontend sends EVERYTHING.

        // Let's check how I plan to use it.
        // I want Frontend to send "current state of chat".
        // So `full_history` should be "everything including latest".
        // But `message` param exists.

        // Revised Logic:
        // If `full_history` is present, use it as the base.
        // Push `message` to it?
        // No, `message` is the "prompt" for completion.
        // It's cleaner if `full_history` replaces the *past*, and we push `message` as user message.
        // This allows `message` to be the trigger.
    }

    // Construct the user message
    let user_message = serde_json::json!({
        "role": "user",
        "content": message
    });
    chat_history.push(user_message);

    let mut messages_for_server = Vec::new();
    if chat_history.len() == 1
        || chat_history.last().map_or(false, |m| {
            m.get("role")
                .and_then(|v| v.as_str())
                .map_or(false, |r| r == "user")
        })
    {
        let system_message = serde_json::json!({
            "role": "system",
            "content": system_prompt
        });
        messages_for_server.push(system_message);
    }
    messages_for_server.extend(chat_history.iter().cloned());

    #[derive(Serialize)]
    struct StreamOptions {
        include_usage: bool,
    }

    #[derive(Serialize)]
    struct LlamaChatStreamRequest {
        messages: Vec<serde_json::Value>,
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

    let event_chunk = label
        .as_ref()
        .map(|l| format!("chat-stream-chunk-{}", l))
        .unwrap_or_else(|| "chat-stream-chunk".to_string());
    let event_done = label
        .as_ref()
        .map(|l| format!("chat-stream-done-{}", l))
        .unwrap_or_else(|| "chat-stream-done".to_string());

    window
        .emit(
            "log",
            format!(
                "Sending message to Llama server ({}): '{}'",
                event_chunk, message
            ),
        )
        .ok();

    let start_time = std::time::Instant::now();
    let mut first_token_time: Option<std::time::Instant> = None;
    let mut token_count = 0;

    let res = client
        .post(&url)
        .json(&request_body)
        .send()
        .await
        .map_err(|e| e.to_string())?;

    let mut stream = res.bytes_stream();
    let mut full_response = String::new();

    while let Some(item) = stream.next().await {
        let chunk = item.map_err(|e| e.to_string())?;
        let s = String::from_utf8_lossy(&chunk);

        // DEBUG: Log raw chunk from server
        debug!("Raw stream chunk: {}", s);

        for line in s.lines() {
            if line.starts_with("data: ") {
                let data = line.trim_start_matches("data: ").trim();
                if data == "[DONE]" {
                    break;
                }

                // DEBUG: Log parsed data
                debug!("Parsed data: {}", data);

                if let Ok(val) = serde_json::from_str::<serde_json::Value>(data) {
                    // DEBUG: Log the full parsed JSON
                    debug!("Parsed JSON: {:?}", val);

                    if let Some(content) = val["choices"][0]["delta"]["content"].as_str() {
                        if first_token_time.is_none() {
                            first_token_time = Some(std::time::Instant::now());
                        }
                        token_count += 1;
                        full_response.push_str(content);
                        window
                            .emit(
                                &event_chunk,
                                StreamChunk {
                                    content: content.to_string(),
                                },
                            )
                            .ok();
                    } else {
                        // DEBUG: Log when content is not found
                        debug!(
                            "No content found in delta. choices[0]: {:?}",
                            val["choices"][0]
                        );
                    }

                    // Check for usage/metrics at the end
                    if let Some(usage) = val.get("usage") {
                        let total_tokens =
                            usage["total_tokens"].as_u64().unwrap_or(token_count as u64);

                        let prompt_eval_time_ms = first_token_time
                            .map(|t| t.duration_since(start_time).as_secs_f64() * 1000.0)
                            .unwrap_or(0.0);
                        let eval_time_ms = if let Some(first) = first_token_time {
                            std::time::Instant::now()
                                .duration_since(first)
                                .as_secs_f64()
                                * 1000.0
                        } else {
                            0.0
                        };
                        let tps = if eval_time_ms > 0.0 {
                            (token_count as f64) / (eval_time_ms / 1000.0)
                        } else {
                            0.0
                        };

                        window
                            .emit(
                                &event_done,
                                StreamMetrics {
                                    prompt_eval_time_ms,
                                    eval_time_ms,
                                    tokens_per_second: tps,
                                    total_tokens,
                                },
                            )
                            .ok();
                    }
                }
            }
        }
    }

    let bot_message = serde_json::json!({
        "role": "assistant",
        "content": full_response
    });
    chat_history.push(bot_message);

    Ok("Stream complete".to_string())
}

// ... Additional simple commands ...

#[tauri::command]
pub async fn clear_chat_history_command(
    llama_chat_context: State<'_, Arc<LlamaChatContext>>,
) -> Result<(), String> {
    let mut chat_history = llama_chat_context.chat_history.lock().await;
    chat_history.clear();
    Ok(())
}

#[tauri::command]
pub async fn get_hf_token_command() -> Result<String, String> {
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
pub async fn save_hf_token_command(token: String) -> Result<String, String> {
    fs::write(HF_TOKEN_FILE, token.trim())
        .await
        .map_err(|e| e.to_string())?;
    Ok("Token saved".to_string())
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
pub async fn get_chat_response_command(
    _window: Window,
    llama_chat_context: State<'_, Arc<LlamaChatContext>>,
) -> Result<Vec<serde_json::Value>, String> {
    let chat_history = llama_chat_context.chat_history.lock().await;
    Ok(chat_history.clone())
}

#[tauri::command]
pub async fn export_resources_command(
    app_handle: AppHandle,
    resource_paths: Vec<String>,
    destination: String,
) -> Result<String, String> {
    let data_dir = get_data_dir(&app_handle);
    let dest_path = PathBuf::from(&destination);

    if !dest_path.exists() {
        return Err("Destination path does not exist".to_string());
    }

    let count = resource_paths.len();
    for resource_path in resource_paths {
        // Resolve source path relative to data dir
        let source = data_dir.join(&resource_path);

        if !source.exists() {
            error!("Export failed: source path does not exist: {:?}", source);
            return Err(format!("Source path does not exist: {}", resource_path));
        }

        let file_name = source
            .file_name()
            .ok_or("Invalid source path")?
            .to_string_lossy()
            .to_string();

        let dest = dest_path.join(&file_name);

        debug!("Exporting {:?} to {:?}", source, dest);

        if source.is_dir() {
            copy_dir_all(&source, &dest).await?;
        } else {
            fs::copy(&source, &dest).await.map_err(|e| e.to_string())?;
        }
    }
    Ok(format!("Exported {} resources", count))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HFSearchResult {
    pub id: String,
    pub name: String,
    pub author: Option<String>,
    pub downloads: u64,
    pub likes: u64,
}

#[tauri::command]
pub async fn search_huggingface_command(
    app_handle: AppHandle,
    query: String,
    resource_type: String,
    author: Option<String>,
    modalities: Option<String>,
    size_range: Option<String>,
) -> Result<Vec<HFSearchResult>, String> {
    let (python_exe, work_dir) = get_python_command(&app_handle)?;
    let script = get_script_path(&app_handle, "huggingface_manager.py");
    let token = if let Ok(t) = fs::read_to_string(HF_TOKEN_FILE).await {
        Some(t.trim().to_string())
    } else {
        None
    };

    let mut cmd = create_hidden_command(&python_exe);
    cmd.arg(&script)
        .arg("search")
        .arg("--query")
        .arg(&query)
        .arg("--type")
        .arg(&resource_type)
        .current_dir(&work_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    if let Some(a) = author {
        if !a.is_empty() {
            cmd.arg("--author").arg(a);
        }
    }
    if let Some(m) = modalities {
        if !m.is_empty() {
            cmd.arg("--modalities").arg(m);
        }
    }
    if let Some(s) = size_range {
        if !s.is_empty() {
            cmd.arg("--size").arg(s);
        }
    }

    if let Some(t) = token {
        if !t.is_empty() {
            cmd.arg("--token").arg(t);
        }
    }

    let child = cmd.spawn().map_err(|e| e.to_string())?;
    let output = child.wait_with_output().await.map_err(|e| e.to_string())?;

    if !output.status.success() {
        return Err(format!(
            "Search script failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let results: Vec<HFSearchResult> =
        serde_json::from_str(&stdout).map_err(|e| format!("Parse error: {}", e))?;
    Ok(results)
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HFFile {
    pub path: String,
    pub size: Option<u64>,
    pub lfs: Option<serde_json::Value>,
    pub file_type: String,
    pub quantization: Option<String>,
    pub is_mmproj: bool,
}

#[tauri::command]
pub async fn list_hf_repo_files_command(
    app_handle: AppHandle,
    repo_id: String,
    token: Option<String>,
    resource_type: Option<String>,
) -> Result<Vec<HFFile>, String> {
    let (python_exe, work_dir) = get_python_command(&app_handle)?;
    let script = get_script_path(&app_handle, "huggingface_manager.py");
    let repo_type = resource_type.as_deref().unwrap_or("model");

    let mut cmd = create_hidden_command(&python_exe);
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
        return Err(format!(
            "List failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let files: Vec<HFFile> =
        serde_json::from_str(&stdout).map_err(|e| format!("Parse error: {}", e))?;
    Ok(files)
}

#[tauri::command]
pub async fn list_directory_command(
    app_handle: AppHandle,
    path: String,
) -> Result<Vec<String>, String> {
    let path_buf = PathBuf::from(&path);
    let target_path = if path_buf.is_absolute() {
        path_buf
    } else {
        get_data_dir(&app_handle).join(path)
    };

    let mut files = Vec::new();
    let read_dir = fs::read_dir(target_path).await.map_err(|e| e.to_string())?;
    let mut entries = tokio_stream::wrappers::ReadDirStream::new(read_dir);

    while let Some(entry) = entries.next().await {
        let entry = entry.map_err(|e| e.to_string())?;
        if let Ok(name) = entry.file_name().into_string() {
            files.push(name);
        }
    }
    Ok(files)
}

#[tauri::command]
pub async fn download_hf_model_command(
    window: Window,
    app_handle: AppHandle,
    models_state: State<'_, Arc<ModelsState>>,
    model_id: String,
    files: Option<Vec<String>>,
    token: Option<String>,
    task_id: Option<String>,
) -> Result<String, String> {
    let (python_exe, work_dir) = get_python_command(&app_handle)?;
    let script = get_script_path(&app_handle, "huggingface_manager.py");
    let data_dir = get_data_dir(&app_handle);
    let sanitized_repo_id = model_id.replace('/', "--");
    let base_output_folder = data_dir
        .join("data")
        .join("models")
        .join(&sanitized_repo_id);

    let mut cmd = create_hidden_command(&python_exe);
    cmd.arg(&script)
        .arg("download")
        .arg("--repo_id")
        .arg(&model_id);

    if let Some(f) = &files {
        if !f.is_empty() {
            cmd.arg("--output")
                .arg(base_output_folder.to_string_lossy().to_string())
                .arg("--files")
                .arg(f.join(","));
        }
    }
    cmd.current_dir(&work_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(t) = &token {
        if !t.is_empty() {
            cmd.arg("--token").arg(t);
        }
    }

    let mut child = cmd.spawn().map_err(|e| e.to_string())?;

    // Persist task for resumption
    if let Some(tid) = &task_id {
        let p_task = PersistentDownloadTask {
            id: tid.clone(),
            name: model_id.clone(),
            type_: "model".to_string(),
            repo_id: model_id.clone(),
            files: files.clone(),
        };
        let _ = save_persistent_task(&app_handle, p_task).await;
    }

    if let Some(tid) = &task_id {
        if let Some(pid) = child.id() {
            models_state
                .active_downloads
                .lock()
                .await
                .insert(tid.clone(), pid);
        }
    }

    let stdout = child.stdout.take().ok_or("No stdout")?;
    let stderr = child.stderr.take().ok_or("No stderr")?;

    let win_c = window.clone();
    tokio::spawn(async move {
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();
        while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
            win_c.emit("log", format!("HF: {}", line.trim())).ok();
            line.clear();
        }
    });

    let win_c = window.clone();
    let task_id_c = task_id.clone();
    tokio::spawn(async move {
        let mut reader = BufReader::new(stderr);
        let mut line = String::new();

        let mut start_time = std::time::Instant::now();
        let mut last_emit_time = std::time::Instant::now();
        let mut first_byte_received = false;

        while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
            let trimmed = line.trim();
            if trimmed.starts_with("BYTES_UPDATE:") {
                if let Some(payload) = trimmed.strip_prefix("BYTES_UPDATE:") {
                    let parts: Vec<&str> = payload.split('/').collect();
                    if parts.len() == 2 {
                        let current = parts[0].parse::<u64>().unwrap_or(0);
                        let total = parts[1].parse::<u64>().unwrap_or(0);

                        if !first_byte_received && current > 0 {
                            start_time = std::time::Instant::now();
                            first_byte_received = true;
                        }

                        let now = std::time::Instant::now();
                        // Throttle progress emission to 200ms for performance
                        if now.duration_since(last_emit_time).as_millis() > 200 || current == total
                        {
                            let elapsed = start_time.elapsed().as_secs_f64();
                            let speed = if elapsed > 0.0 {
                                (current as f64) / elapsed
                            } else {
                                0.0
                            };
                            let progress = if total > 0 {
                                (current as f64 / total as f64) * 100.0
                            } else {
                                0.0
                            };
                            let eta = if speed > 0.1 && total > current {
                                (total.saturating_sub(current) as f64) / speed
                            } else {
                                0.0
                            };

                            if let Some(tid) = &task_id_c {
                                win_c
                                    .emit(
                                        "download_progress",
                                        serde_json::json!({
                                            "id": tid,
                                            "progress": progress,
                                            "downloaded_bytes": current,
                                            "total_bytes": total,
                                            "speed_bps": speed,
                                            "eta_seconds": eta
                                        }),
                                    )
                                    .ok();
                            }
                            last_emit_time = now;
                        }
                    }
                }
            } else if trimmed.starts_with("PROGRESS:") {
                if let Some(value) = trimmed.strip_prefix("PROGRESS:") {
                    if let Ok(percentage) = value.parse::<u8>() {
                        if let Some(tid) = &task_id_c {
                            win_c
                                .emit(
                                    "download_progress",
                                    serde_json::json!({ "id": tid, "progress": percentage }),
                                )
                                .ok();
                        } else {
                            win_c.emit("download_progress", percentage).ok();
                        }
                    }
                }
            } else {
                win_c.emit("log", format!("HF: {}", trimmed)).ok();
            }
            line.clear();
        }
    });

    let win_c = window.clone();
    let task_id_c = task_id.clone();
    let models_state_c = (*models_state).clone();
    let model_id_c = model_id.clone();
    let output_folder_c = base_output_folder.clone();
    let app_handle_c = app_handle.clone();

    tokio::spawn(async move {
        let status = child.wait().await;
        // Emit global event on success
        let success = status.map(|s| s.success()).unwrap_or(false);
        if success {
            win_c.emit("model_downloaded", &model_id_c).ok();

            // Save metadata for healing
            let meta_path = output_folder_c.join("metadata.json");
            let meta = serde_json::json!({
                "source_repo": model_id_c
            });
            if let Ok(content) = serde_json::to_string_pretty(&meta) {
                let _ = tokio::fs::write(&meta_path, content).await;
            }
        }

        if let Some(tid) = &task_id_c {
            let _ = remove_persistent_task(&app_handle_c, tid).await;
            models_state_c.active_downloads.lock().await.remove(tid);
            let status_str = if success { "completed" } else { "error" };
            win_c
                .emit(
                    "download_status",
                    serde_json::json!({ "id": tid, "status": status_str }),
                )
                .ok();
        }
    });

    Ok("Download started".to_string())
}

#[tauri::command]
pub async fn download_hf_dataset_command(
    window: Window,
    app_handle: AppHandle,
    models_state: State<'_, Arc<ModelsState>>,
    dataset_id: String,
    files: Option<Vec<String>>,
    token: Option<String>,
    task_id: Option<String>,
) -> Result<String, String> {
    let (python_exe, work_dir) = get_python_command(&app_handle)?;
    let script = get_script_path(&app_handle, "huggingface_manager.py");
    let data_dir = get_data_dir(&app_handle);
    let sanitized_id = dataset_id.replace('/', "--");
    let output_folder = data_dir.join("data").join("datasets").join(&sanitized_id);

    let mut cmd = create_hidden_command(&python_exe);
    cmd.arg(&script)
        .arg("download")
        .arg("--repo_id")
        .arg(&dataset_id)
        .arg("--repo_type")
        .arg("dataset");

    if let Some(f) = &files {
        if !f.is_empty() {
            cmd.arg("--output")
                .arg(output_folder.to_string_lossy().to_string())
                .arg("--files")
                .arg(f.join(","));
        }
    }
    cmd.current_dir(&work_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(t) = &token {
        if !t.is_empty() {
            cmd.arg("--token").arg(t);
        }
    }

    let mut child = cmd.spawn().map_err(|e| e.to_string())?;

    // Persist task for resumption
    if let Some(tid) = &task_id {
        let p_task = PersistentDownloadTask {
            id: tid.clone(),
            name: dataset_id.clone(),
            type_: "dataset".to_string(),
            repo_id: dataset_id.clone(),
            files: files.clone(),
        };
        let _ = save_persistent_task(&app_handle, p_task).await;
    }

    if let Some(tid) = &task_id {
        if let Some(pid) = child.id() {
            models_state
                .active_downloads
                .lock()
                .await
                .insert(tid.clone(), pid);
        }
    }

    let stdout = child.stdout.take().ok_or("No stdout")?;
    let stderr = child.stderr.take().ok_or("No stderr")?;

    let win_c = window.clone();
    tokio::spawn(async move {
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();
        while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
            win_c.emit("log", format!("HF_DS: {}", line.trim())).ok();
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
                let val = trimmed.strip_prefix("PROGRESS:").unwrap_or("0");
                if let Ok(p) = val.parse::<u8>() {
                    if let Some(tid) = &task_id_c {
                        win_c
                            .emit(
                                "download_progress",
                                serde_json::json!({ "id": tid, "progress": p }),
                            )
                            .ok();
                    } else {
                        win_c.emit("download_progress", p).ok();
                    }
                }
            } else {
                win_c.emit("log", format!("HF_DS: {}", trimmed)).ok();
            }
            line.clear();
        }
    });

    let win_c = window.clone();
    let task_id_c = task_id.clone();
    let models_state_c = (*models_state).clone();
    let dataset_id_c = dataset_id.clone();
    let app_handle_c = app_handle.clone();
    tokio::spawn(async move {
        let status = child.wait().await;
        let success = status.map(|s| s.success()).unwrap_or(false);
        if success {
            win_c.emit("dataset_downloaded", &dataset_id_c).ok();
        }

        if let Some(tid) = &task_id_c {
            let _ = remove_persistent_task(&app_handle_c, tid).await;
            models_state_c.active_downloads.lock().await.remove(tid);
            let status_str = if success { "completed" } else { "error" };
            win_c
                .emit(
                    "download_status",
                    serde_json::json!({ "id": tid, "status": status_str }),
                )
                .ok();
        }
    });
    Ok("Download started".to_string())
}

#[tauri::command]
pub async fn convert_dataset_command(
    window: Window,
    app_handle: AppHandle,
    models_state: State<'_, Arc<ModelsState>>,
    source_path: String,
    destination_path: String,
    task_id: Option<String>,
) -> Result<String, String> {
    let (python_exe, work_dir) = get_python_command(&app_handle)?;
    let script = get_script_path(&app_handle, "huggingface_manager.py");
    let data_dir = get_data_dir(&app_handle);

    let source_abs = if PathBuf::from(&source_path).is_absolute() {
        source_path
    } else {
        data_dir.join(&source_path).to_string_lossy().to_string()
    };
    let dest_abs = if PathBuf::from(&destination_path).is_absolute() {
        destination_path
    } else {
        data_dir
            .join(&destination_path)
            .to_string_lossy()
            .to_string()
    };

    let mut cmd = create_hidden_command(&python_exe);
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

    // Track active task for cancellation/progress
    if let Some(tid) = &task_id {
        if let Some(pid) = child.id() {
            models_state
                .active_downloads
                .lock()
                .await
                .insert(tid.clone(), pid);
        }
    }

    let stdout = child.stdout.take().ok_or("No stdout")?;
    let stderr = child.stderr.take().ok_or("No stderr")?;

    let win_c = window.clone();
    tokio::spawn(async move {
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();
        while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
            if !line.trim().is_empty() {
                win_c.emit("log", format!("CONVERT: {}", line.trim())).ok();
            }
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
                // Reuse download_progress event logic
                if let Some(value) = trimmed.strip_prefix("PROGRESS:") {
                    if let Ok(percentage) = value.parse::<u8>() {
                        if let Some(tid) = &task_id_c {
                            win_c
                                .emit(
                                    "download_progress",
                                    serde_json::json!({ "id": tid, "progress": percentage }),
                                )
                                .ok();
                        }
                    }
                }
            } else if trimmed.starts_with("Map") && trimmed.contains('%') {
                // Parse "Map ... 52% ..." for tqdm progress
                if let Some(idx) = trimmed.find('%') {
                    let prefix = &trimmed[..idx];
                    let mut start_idx = idx;
                    for (i, c) in prefix.char_indices().rev() {
                        if !c.is_ascii_digit() {
                            break;
                        }
                        start_idx = i;
                    }
                    if start_idx < idx {
                        if let Ok(percentage) = prefix[start_idx..idx].parse::<u8>() {
                            if let Some(tid) = &task_id_c {
                                win_c
                                    .emit(
                                        "download_progress",
                                        serde_json::json!({ "id": tid, "progress": percentage }),
                                    )
                                    .ok();
                            }
                        }
                    }
                }
                // Log the map line as well for detail
                win_c.emit("log", format!("CONVERT: {}", trimmed)).ok();
            } else if !trimmed.is_empty() {
                win_c.emit("log", format!("CONVERT: {}", trimmed)).ok();
            }
            line.clear();
        }
    });

    let status = child.wait().await.map_err(|e| e.to_string())?;

    if let Some(tid) = &task_id {
        models_state.active_downloads.lock().await.remove(tid);
        let status_str = if status.success() {
            "completed"
        } else {
            "error"
        };
        window
            .emit(
                "download_status",
                serde_json::json!({ "id": tid, "status": status_str }),
            )
            .ok();
    }

    if status.success() {
        Ok("Dataset converted successfully".to_string())
    } else {
        Err("Dataset conversion failed".to_string())
    }
}

// fix_dataset_command removed: superseded by new implementation at the end of file.

#[tauri::command]
pub async fn process_vlm_dataset_command(
    window: Window,
    app_handle: AppHandle,
    dataset_dir: String,
    model_name: String,
) -> Result<String, String> {
    let (python_exe, work_dir) = get_python_command(&app_handle)?;
    let script = get_script_path(&app_handle, "process_data.py");
    let data_dir = get_data_dir(&app_handle);

    let source_abs = if PathBuf::from(&dataset_dir).is_absolute() {
        dataset_dir.clone()
    } else {
        data_dir.join(&dataset_dir).to_string_lossy().to_string()
    };

    // Output dir is usually specific for VLM processing, e.g. source_dir/processed_vlm
    let output_dir = PathBuf::from(&source_abs)
        .join("processed_data")
        .to_string_lossy()
        .to_string();

    let mut cmd = create_hidden_command(&python_exe);
    cmd.arg(&script)
        .arg("--dataset_dir")
        .arg(&source_abs)
        .arg("--output_dir")
        .arg(&output_dir)
        .arg("--model_name")
        .arg(&model_name)
        .arg("--force") // Always force overwrite for cleanliness
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
                win_c.emit("log", format!("VLM_PROC: {}", line.trim())).ok();
            }
            line.clear();
        }
    });

    let win_c2 = window.clone();
    tokio::spawn(async move {
        let mut reader = BufReader::new(stderr);
        let mut line = String::new();
        while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
            if !line.trim().is_empty() {
                win_c2
                    .emit("log", format!("VLM_PROC ERR: {}", line.trim()))
                    .ok();
            }
            line.clear();
        }
    });

    if child.wait().await.map_err(|e| e.to_string())?.success() {
        Ok("VLM Dataset processing completed".to_string())
    } else {
        Err("VLM Dataset processing failed".to_string())
    }
}

#[tauri::command]
pub async fn delete_resource_command(
    app_handle: AppHandle,
    resource_type: String,
    resource_path: String,
) -> Result<String, String> {
    let data_dir = get_data_dir(&app_handle);

    // Resolve base path based on resource type
    let base_dir = match resource_type.as_str() {
        "model" | "gguf" => data_dir.join("data/models"),
        "lora" => data_dir.join("data/loras"),
        "dataset" => data_dir.join("data/datasets"),
        _ => return Err("Invalid resource type".to_string()),
    };

    let full_path = base_dir.join(&resource_path);

    if !full_path.exists() {
        return Err(format!("Resource not found at: {:?}", full_path));
    }

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
pub async fn rename_resource_command(
    app_handle: AppHandle,
    resource_type: String,
    current_name: String,
    new_name: String,
) -> Result<String, String> {
    let data_dir = get_data_dir(&app_handle);
    let base_dir = match resource_type.as_str() {
        "model" | "gguf" => data_dir.join("data/models"),
        "lora" => data_dir.join("data/loras"),
        "dataset" => data_dir.join("data/datasets"),
        _ => return Err("Invalid resource type".to_string()),
    };

    let current_path = base_dir.join(&current_name);
    let new_path = base_dir.join(&new_name);

    if !current_path.exists() {
        return Err("Source resource does not exist".to_string());
    }
    if new_path.exists() {
        return Err("A resource with the new name already exists".to_string());
    }

    // Attempt rename
    fs::rename(&current_path, &new_path)
        .await
        .map_err(|e| format!("Rename failed: {}", e))?;

    Ok(format!("Renamed to {}", new_name))
}

#[tauri::command]
pub async fn import_resource_command(
    app_handle: AppHandle,
    resource_type: String,
    source_path: String,
) -> Result<String, String> {
    let source = PathBuf::from(&source_path);
    let data_dir = get_data_dir(&app_handle);
    let dest_base = match resource_type.as_str() {
        "model" | "gguf" => data_dir.join("data/models"),
        "lora" => data_dir.join("data/loras"),
        "dataset" => data_dir.join("data/datasets"),
        _ => return Err("Invalid resource type".to_string()),
    };

    if !dest_base.exists() {
        fs::create_dir_all(&dest_base)
            .await
            .map_err(|e| e.to_string())?;
    }

    let file_name = source
        .file_name()
        .ok_or("Invalid source")?
        .to_string_lossy()
        .to_string();

    // Special handling for single dataset files: Containerize them
    if resource_type == "dataset" && source.is_file() {
        let file_stem = source
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or(file_name.clone());
        let container_path = dest_base.join(&file_stem);

        if !container_path.exists() {
            fs::create_dir_all(&container_path)
                .await
                .map_err(|e| e.to_string())?;
        }

        // Standardize naming if JSONL
        let dest_file_name = if file_name.to_lowercase().ends_with(".jsonl") {
            "training_data.jsonl".to_string()
        } else {
            file_name.clone()
        };

        let dest = container_path.join(dest_file_name);
        fs::copy(&source, &dest).await.map_err(|e| e.to_string())?;

        return Ok(format!("Dataset imported into folder: {}", file_stem));
    }

    let dest = dest_base.join(&file_name);

    if source.is_dir() {
        copy_dir_all(&source, &dest).await?;
    } else {
        fs::copy(&source, &dest).await.map_err(|e| e.to_string())?;
    }
    Ok(format!("{} imported successfully", resource_type))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Preset {
    pub id: String,
    pub name: String,
    pub model: String,
    pub lora: String,
}

#[tauri::command]
pub async fn load_presets_command() -> Result<Vec<Preset>, String> {
    let path = PathBuf::from(PRESETS_FILE);
    if !path.exists() {
        return Ok(Vec::new());
    }
    let c = fs::read_to_string(&path).await.map_err(|e| e.to_string())?;
    Ok(serde_json::from_str(&c).unwrap_or_default())
}

#[tauri::command]
pub async fn save_preset_command(preset: Preset) -> Result<String, String> {
    let mut presets = load_presets_command().await.unwrap_or_default();
    presets.push(preset);
    let s = serde_json::to_string_pretty(&presets).map_err(|e| e.to_string())?;
    fs::write(PRESETS_FILE, s)
        .await
        .map_err(|e| e.to_string())?;
    Ok("Preset saved".to_string())
}

#[tauri::command]
pub async fn convert_hf_to_gguf_command(
    window: Window,
    source_path: String,
    output_path: Option<String>,
    quantization_type: String,
) -> Result<String, String> {
    let (python_exe, work_dir) = get_python_command(window.app_handle())?;
    let script = get_script_path(window.app_handle(), "convert_hf_to_gguf.py");

    let data_dir = get_data_dir(window.app_handle());
    let resolved_source = if PathBuf::from(&source_path).is_absolute() {
        source_path.clone()
    } else {
        data_dir.join(&source_path).to_string_lossy().to_string()
    };

    let mut cmd = create_hidden_command(&python_exe);
    cmd.arg(&script);
    if let Some(out) = &output_path {
        cmd.arg("--outfile").arg(out);
    }
    cmd.arg("--outtype").arg(&quantization_type);
    cmd.arg(&resolved_source);
    cmd.current_dir(&work_dir)
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
            let t = line.trim();
            win_c.emit("log", format!("CONVERT: {}", t)).ok();

            // Heuristic progress parsing for convert_hf_to_gguf.py
            if t.contains("Loading model:") {
                win_c.emit("conversion_progress", 5).ok();
            } else if t.contains("indexing model part") {
                win_c.emit("conversion_progress", 20).ok();
            } else if t.contains("Exporting model...") {
                win_c.emit("conversion_progress", 50).ok();
            } else if t.contains("successfully exported to") {
                win_c.emit("conversion_progress", 100).ok();
            }

            line.clear();
        }
    });

    let win_c2 = window.clone();
    let stderr_handle = tokio::spawn(async move {
        let mut reader = BufReader::new(stderr);
        let mut line = String::new();
        let mut cap = String::new();
        while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
            let t = line.trim();
            if !t.is_empty() {
                win_c2.emit("log", format!("CONVERT ERR: {}", t)).ok();
                cap.push_str(t);
                cap.push('\n');
            }
            line.clear();
        }
        cap
    });

    let status = child.wait().await.map_err(|e| e.to_string())?;
    let err = stderr_handle.await.map_err(|e| e.to_string())?;

    if status.success() {
        Ok("Conversion complete".to_string())
    } else {
        // Auto-fix: Check for ModuleNotFoundError: No module named 'gguf'
        if err.contains("ModuleNotFoundError: No module named 'gguf'") {
            let win_c3 = window.clone();
            win_c3
                .emit(
                    "log",
                    "GGUF module missing. Attempting auto-fix (installing from git)...",
                )
                .ok();

            // Run pip install
            let install_cmd = create_hidden_command(&python_exe)
                .arg("-m")
                .arg("pip")
                .arg("install")
                .arg("gguf @ git+https://github.com/ggerganov/llama.cpp.git#subdirectory=gguf-py")
                .current_dir(&work_dir)
                .output()
                .await
                .map_err(|e| e.to_string())?;

            if install_cmd.status.success() {
                win_c3
                    .emit("log", "GGUF installed successfully. Retrying conversion...")
                    .ok();

                // Retry the original command
                let mut retry_cmd = create_hidden_command(&python_exe);
                retry_cmd.arg(&script);
                if let Some(out) = output_path {
                    retry_cmd.arg("--outfile").arg(out);
                }
                retry_cmd.arg("--outtype").arg(&quantization_type);
                retry_cmd.arg(&source_path);
                retry_cmd
                    .current_dir(&work_dir)
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped());

                let mut retry_child = retry_cmd.spawn().map_err(|e| e.to_string())?;
                let retry_stdout = retry_child.stdout.take().ok_or("No stdout on retry")?;
                let retry_stderr = retry_child.stderr.take().ok_or("No stderr on retry")?;

                // Stream retry output
                let win_c4 = window.clone();
                tokio::spawn(async move {
                    let mut reader = BufReader::new(retry_stdout);
                    let mut line = String::new();
                    while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
                        win_c4
                            .emit("log", format!("CONVERT (RETRY): {}", line.trim()))
                            .ok();
                        line.clear();
                    }
                });

                let win_c5 = window.clone();
                let retry_stderr_handle = tokio::spawn(async move {
                    let mut reader = BufReader::new(retry_stderr);
                    let mut line = String::new();
                    let mut cap = String::new();
                    while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
                        let t = line.trim();
                        if !t.is_empty() {
                            win_c5
                                .emit("log", format!("CONVERT ERR (RETRY): {}", t))
                                .ok();
                            cap.push_str(t);
                            cap.push('\n');
                        }
                        line.clear();
                    }
                    cap
                });

                let retry_status = retry_child.wait().await.map_err(|e| e.to_string())?;
                let retry_err_msg = retry_stderr_handle.await.map_err(|e| e.to_string())?;

                if retry_status.success() {
                    return Ok("Conversion complete (after auto-fix)".to_string());
                } else {
                    return Err(format!(
                        "Conversion failed after auto-fix: {}",
                        retry_err_msg.trim()
                    ));
                }
            } else {
                let install_err = String::from_utf8_lossy(&install_cmd.stderr);
                win_c3
                    .emit("log", format!("Auto-fix failed: {}", install_err))
                    .ok();
                return Err(format!(
                    "Missing GGUF module and auto-fix failed: {}",
                    install_err
                ));
            }
        }

        Err(format!("Conversion failed: {}", err.trim()))
    }
}

#[tauri::command]
pub async fn convert_lora_to_gguf_command(
    window: Window,
    lora_path: String,
    base_path: String,
    output_path: Option<String>,
    quantization_type: String,
) -> Result<String, String> {
    let (python_exe, work_dir) = get_python_command(window.app_handle())?;
    let script = get_script_path(window.app_handle(), "convert_lora_to_gguf.py");

    let data_dir = get_data_dir(window.app_handle());
    let resolved_base = if PathBuf::from(&base_path).is_absolute() {
        base_path.clone()
    } else {
        // Try direct first, then models subdir
        let direct = data_dir.join(&base_path);
        if direct.exists() {
            direct.to_string_lossy().to_string()
        } else {
            let model_path = data_dir.join("data").join("models").join(&base_path);
            if model_path.exists() {
                model_path.to_string_lossy().to_string()
            } else {
                direct.to_string_lossy().to_string()
            }
        }
    };
    let resolved_lora = if PathBuf::from(&lora_path).is_absolute() {
        lora_path.clone()
    } else {
        data_dir.join(&lora_path).to_string_lossy().to_string()
    };

    let mut cmd = create_hidden_command(&python_exe);
    // Note: convert_lora_to_gguf.py expects: lora_path (positional), --base, --outfile, --outtype
    cmd.arg(&script)
        .arg(&resolved_lora) // Positional argument first
        .arg("--base")
        .arg(&resolved_base);
    if let Some(out) = output_path {
        cmd.arg("--outfile").arg(out);
    }
    cmd.arg("--outtype").arg(quantization_type);
    cmd.current_dir(&work_dir)
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
            let t = line.trim();
            win_c.emit("log", format!("CONVERT_LORA: {}", t)).ok();

            // Heuristic progress parsing for convert_lora_to_gguf.py
            if t.contains("Loading base model") {
                win_c.emit("conversion_progress", 10).ok();
            } else if t.contains("Exporting model...") {
                win_c.emit("conversion_progress", 50).ok();
            } else if t.contains("Model successfully exported to") {
                win_c.emit("conversion_progress", 100).ok();
            }

            line.clear();
        }
    });

    let win_c2 = window.clone();
    let stderr_handle = tokio::spawn(async move {
        let mut reader = BufReader::new(stderr);
        let mut line = String::new();
        let mut cap = String::new();
        while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
            let t = line.trim();
            if !t.is_empty() {
                win_c2.emit("log", format!("CONVERT_LORA ERR: {}", t)).ok();
                cap.push_str(t);
                cap.push('\n');
            }
            line.clear();
        }
        cap
    });

    let status = child.wait().await.map_err(|e| e.to_string())?;
    let err = stderr_handle.await.map_err(|e| e.to_string())?;

    if status.success() {
        Ok("LoRA conversion complete".to_string())
    } else {
        Err(format!("LoRA conversion failed: {}", err.trim()))
    }
}

#[tauri::command]
pub async fn cancel_download_command(
    models_state: State<'_, Arc<ModelsState>>,
    task_id: String,
) -> Result<String, String> {
    let mut downloads = models_state.active_downloads.lock().await;
    if let Some(pid) = downloads.get(&task_id) {
        #[cfg(target_os = "windows")]
        let _ = create_hidden_std_command("taskkill")
            .args(&["/F", "/PID", &pid.to_string()])
            .output();
        #[cfg(not(target_os = "windows"))]
        let _ = create_hidden_std_command("kill")
            .args(&["-9", &pid.to_string()])
            .output();

        downloads.remove(&task_id);
        Ok(format!("Download {} cancelled", task_id))
    } else {
        Ok(format!(
            "Download {} not found or already finished",
            task_id
        ))
    }
}

#[tauri::command]
pub async fn convert_unsloth_gguf_command(
    window: Window,
    app_handle: AppHandle,
    source_path: String,
    output_path: String,
    quantization_type: String,
    lora_path: Option<String>,
) -> Result<String, String> {
    let (python_exe, work_dir) = get_python_command(&app_handle)?;
    let script = get_script_path(&app_handle, "convert_unsloth_gguf.py");

    let data_dir = get_data_dir(&app_handle);
    let resolved_source = if PathBuf::from(&source_path).is_absolute() {
        source_path.clone()
    } else {
        // Try direct first, then models subdir
        let direct = data_dir.join(&source_path);
        if direct.exists() {
            direct.to_string_lossy().to_string()
        } else {
            let model_path = data_dir.join("data").join("models").join(&source_path);
            if model_path.exists() {
                model_path.to_string_lossy().to_string()
            } else {
                // Fallback to direct path for unsloth to potentially handle as HF ID
                direct.to_string_lossy().to_string()
            }
        }
    };

    let mut cmd = create_hidden_command(&python_exe);
    cmd.arg(&script)
        .arg("--model")
        .arg(&resolved_source)
        .arg("--output")
        .arg(&output_path)
        .arg("--quant")
        .arg(&quantization_type);

    if let Some(lora) = &lora_path {
        if !lora.is_empty() {
            let resolved_lora = if PathBuf::from(lora).is_absolute() {
                lora.clone()
            } else {
                data_dir.join(lora).to_string_lossy().to_string()
            };
            cmd.arg("--lora").arg(resolved_lora);
        }
    }

    cmd.current_dir(&work_dir)
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
            let t = line.trim();
            win_c.emit("log", format!("UNSLOTH_GGUF: {}", t)).ok();

            // Parse progress: "PROGRESS: 10% - ..."
            if t.contains("PROGRESS:") {
                if let Some(p_part) = t.split("PROGRESS:").nth(1) {
                    if let Some(percent_str) = p_part.split('%').next() {
                        if let Ok(percent) = percent_str.trim().parse::<u32>() {
                            win_c.emit("conversion_progress", percent).ok();
                        }
                    }
                }
            }
            line.clear();
        }
    });

    let win_c2 = window.clone();
    let stderr_handle = tokio::spawn(async move {
        let mut reader = BufReader::new(stderr);
        let mut line = String::new();
        let mut cap = String::new();
        while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
            let t = line.trim();
            if !t.is_empty() {
                win_c2.emit("log", format!("UNSLOTH_GGUF: {}", t)).ok();
                cap.push_str(t);
                cap.push('\n');
            }
            line.clear();
        }
        cap
    });

    let status = child.wait().await.map_err(|e| e.to_string())?;
    let err = stderr_handle.await.map_err(|e| e.to_string())?;

    if status.success() {
        Ok("Unsloth GGUF conversion complete".to_string())
    } else {
        Err(format!("Unsloth GGUF conversion failed: {}", err.trim()))
    }
}

#[tauri::command]
pub async fn start_training_command(
    window: Window,
    app_handle: AppHandle,
    python_process_state: State<'_, Arc<PythonProcessState>>,
    project_name: String,
    model_path: String,
    dataset_paths: Vec<String>, // Changed from dataset_path: String to support multiple
    // Removed dataset known weights
    num_epochs: u32,
    batch_size: u32,
    learning_rate: f64,
    lora_r: u32,
    lora_alpha: u32,
    max_seq_length: u32,
    training_method: Option<String>, // sft, dpo, orpo
    adapter_type: Option<String>,    // lora, dora, full
    gradient_accumulation_steps: Option<u32>,
    warmup_ratio: Option<f64>,
    weight_decay: Option<f64>,
    optimizer: Option<String>,
    lr_scheduler_type: Option<String>,
    // New parameters
    eval_split: Option<f64>,
    eval_steps: Option<u32>,
    use_cpu_offload: Option<bool>,
    use_paged_optimizer: Option<bool>,
    use_gradient_checkpointing: Option<bool>,
    hybrid_training: Option<bool>,
    gpu_layers: Option<u32>,
    offload_optimizer: Option<bool>,
    use_deepspeed: Option<bool>,
) -> Result<serde_json::Value, String> {
    let (python_exe, work_dir) = get_python_command(&app_handle)?;
    let data_dir = get_data_dir(&app_handle);

    // Resolve multiple dataset paths
    let datasets_abs: Vec<String> = dataset_paths
        .iter()
        .map(|dataset_path| {
            if PathBuf::from(&dataset_path).is_absolute() {
                dataset_path.clone()
            } else if dataset_path.contains("data/datasets")
                || dataset_path.contains("data\\datasets")
            {
                data_dir.join(&dataset_path).to_string_lossy().to_string()
            } else {
                data_dir
                    .join("data/datasets")
                    .join(&dataset_path)
                    .to_string_lossy()
                    .to_string()
            }
        })
        .collect();

    // Join paths with comma for --datasets argument
    let datasets_arg = datasets_abs.join(",");

    // Tensorboard logic (Simplified for length: assuming already handled by setup or similar if we want robustness,
    // but honestly we should probably include the minimal TB logic if it's critical).
    // I'll assume TB is handled or use a simplified approach as I'm running out of space?
    // No, I'll include the port finding logic.

    // Check if TB is running. If not, start it.
    let mut tb_port_guard = python_process_state.tensorboard_port.lock().await;
    let tensorboard_port = if let Some(p) = *tb_port_guard {
        p
    } else {
        // Find free port
        let listener = std::net::TcpListener::bind("127.0.0.1:0").map_err(|e| e.to_string())?;
        let port = listener.local_addr().map_err(|e| e.to_string())?.port();

        // Start TensorBoard
        // Better: use python -m tensorboard.main
        let mut tb_cmd = create_hidden_command(&python_exe);

        // Note: We use data/outputs as logdir
        let logdir = data_dir
            .join("data")
            .join("outputs")
            .to_string_lossy()
            .to_string();

        #[cfg(target_os = "windows")]
        {
            // use std::os::windows::process::CommandExt; // already imported or handled by hidden command helper
            tb_cmd.creation_flags(0x08000000);
        }

        tb_cmd
            .arg("-m")
            .arg("tensorboard.main")
            .arg("--logdir")
            .arg(&logdir)
            .arg("--port")
            .arg(port.to_string())
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--reload_interval")
            .arg("15")
            .arg("--reload_multifile")
            .arg("true")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        match tb_cmd.spawn() {
            Ok(child) => {
                let mut child_guard = python_process_state.tensorboard_child.lock().await;
                *child_guard = Some(child);
                *tb_port_guard = Some(port);
                let _ = app_handle.emit("log", format!("TensorBoard started on port {}", port));
                port
            }
            Err(e) => return Err(format!("Failed to start TensorBoard: {}", e)),
        }
    };

    let script = get_script_path(&app_handle, "train.py");
    let mut cmd = create_hidden_command(&python_exe);
    #[cfg(target_os = "windows")]
    {
        cmd.creation_flags(0x08000000);
    }

    let output_dir_abs = data_dir
        .join("data")
        .join("outputs")
        .join(&project_name)
        .to_string_lossy()
        .to_string();

    // Ensure output directory exists (Tensorboard might need it or script will create it)
    // We create it here to save metadata immediately
    let output_path_buf = PathBuf::from(&output_dir_abs);
    if let Err(e) = std::fs::create_dir_all(&output_path_buf) {
        log::warn!("Failed to create output dir for metadata: {}", e);
    } else {
        // Save metadata
        let metadata_path = output_path_buf.join("project_metadata.json");
        let metadata = serde_json::json!({
            "base_model": model_path,
            "training_started_at": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs()
        });
        if let Ok(s) = serde_json::to_string_pretty(&metadata) {
            if let Err(e) = std::fs::write(metadata_path, s) {
                log::warn!("Failed to write project metadata: {}", e);
            }
        }
    }

    cmd.arg(&script)
        .arg("--model")
        .arg(&model_path)
        .arg("--datasets")
        .arg(&datasets_arg)
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
        .arg(&output_dir_abs);

    // Add optional training method and adapter type
    if let Some(tm) = &training_method {
        cmd.arg("--training_method").arg(tm);
    }

    // Handle Adapter Type correctly for train.py
    if let Some(at) = &adapter_type {
        if at == "dora" {
            // train.py expects --use_dora for DoRA, not --adapter_type dora
            cmd.arg("--use_dora");
        } else {
            // For 'lora' or 'full', pass as standard arg
            cmd.arg("--adapter_type").arg(at);
        }
    }
    if let Some(gas) = gradient_accumulation_steps {
        cmd.arg("--gradient_accumulation_steps")
            .arg(gas.to_string());
    }
    if let Some(wr) = warmup_ratio {
        cmd.arg("--warmup_ratio").arg(wr.to_string());
    }
    if let Some(wd) = weight_decay {
        cmd.arg("--weight_decay").arg(wd.to_string());
    }
    if let Some(opt) = &optimizer {
        cmd.arg("--optimizer").arg(opt);
    }
    if let Some(lrs) = &lr_scheduler_type {
        cmd.arg("--lr_scheduler_type").arg(lrs);
    }

    // New Params
    if let Some(es) = eval_split {
        cmd.arg("--eval_split").arg(es.to_string());
    }
    if let Some(steps) = eval_steps {
        cmd.arg("--eval_steps").arg(steps.to_string());
    }
    if let Some(true) = use_cpu_offload {
        cmd.arg("--use_cpu_offload");
    }
    if let Some(true) = use_paged_optimizer {
        cmd.arg("--use_paged_optimizer");
    }
    if let Some(true) = use_gradient_checkpointing {
        cmd.arg("--use_gradient_checkpointing");
    }

    // Hybrid Settings
    if let Some(true) = hybrid_training {
        cmd.arg("--hybrid_training");
    }
    if let Some(layers) = gpu_layers {
        cmd.arg("--gpu_layers").arg(layers.to_string());
    }
    if let Some(true) = offload_optimizer {
        cmd.arg("--offload_optimizer");
    }
    if let Some(true) = use_deepspeed {
        cmd.arg("--use_deepspeed");
    }

    cmd.current_dir(&work_dir)
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
            win_c.emit("log", format!("TRAIN: {}", line.trim())).ok();
            line.clear();
        }
    });

    let win_c2 = window.clone();
    tokio::spawn(async move {
        let mut reader = BufReader::new(stderr);
        let mut line = String::new();
        while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
            win_c2.emit("log", format!("TRAIN: {}", line.trim())).ok();
            line.clear();
        }
    });

    // Monitor process exit
    let win_c3 = window.clone();
    tokio::spawn(async move {
        let status = child.wait().await;
        // Emit global event regarding training status
        let success = status.as_ref().map(|s| s.success()).unwrap_or(false);
        let code = status.as_ref().ok().and_then(|s| s.code());

        // Log the final result
        if success {
            win_c3
                .emit("log", "TRAIN: Training finished successfully.")
                .ok();
        } else {
            win_c3
                .emit(
                    "log",
                    format!("TRAIN: Training failed with exit code {:?}", code),
                )
                .ok();
        }

        win_c3
            .emit(
                "training_finished",
                serde_json::json!({
                    "success": success,
                    "code": code
                }),
            )
            .ok();
    });

    let mut res = serde_json::Map::new();
    res.insert(
        "tensorboard_port".to_string(),
        serde_json::json!(tensorboard_port),
    );
    Ok(serde_json::Value::Object(res))
}

#[tauri::command]
pub async fn stop_training_command() -> Result<String, String> {
    #[cfg(target_os = "windows")]
    {
        let _ = create_hidden_std_command("taskkill")
            .args(&["/F", "/IM", "python.exe"])
            .output();
        let _ = create_hidden_std_command("taskkill")
            .args(&["/F", "/IM", "tensorboard.exe"])
            .output();
    }
    #[cfg(not(target_os = "windows"))]
    {
        let _ = create_hidden_std_command("pkill").arg("python").output();
        let _ = create_hidden_std_command("pkill")
            .arg("tensorboard")
            .output();
    }
    Ok("Training stopped".to_string())
}

#[allow(dead_code)]
fn get_python_standalone_dir(app_handle: &AppHandle) -> PathBuf {
    let data_dir = get_data_dir(app_handle);
    data_dir.join("python")
}

#[tauri::command]
pub async fn check_python_standalone_command(app_handle: AppHandle) -> Result<bool, String> {
    let python_dir = get_python_standalone_dir(&app_handle);
    Ok(python_dir.join("python.exe").exists())
}

#[tauri::command]
pub async fn download_python_standalone_command(
    window: Window,
    app_handle: AppHandle,
) -> Result<String, String> {
    #[cfg(not(target_os = "windows"))]
    return Err("Only supported on Windows".to_string());

    #[cfg(target_os = "windows")]
    {
        window.emit("setup_progress", serde_json::json!({ "step": "init", "message": "Starting Python download...", "progress": 0 })).ok();

        let url = "https://github.com/indygreg/python-build-standalone/releases/download/20241016/cpython-3.11.10+20241016-x86_64-pc-windows-msvc-install_only.tar.gz";
        let client = reqwest::Client::new();
        let mut retries = 3;
        let mut resp = None;
        while retries > 0 {
            let res = client.get(url).send().await;
            match res {
                Ok(r) => {
                    if r.status().is_success() {
                        resp = Some(r);
                        break;
                    } else if r.status().as_u16() == 503 {
                        window.emit("log", "GitHub 503 error, retrying...").ok();
                        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                        retries -= 1;
                    } else {
                        return Err(format!("Download failed with status: {}", r.status()));
                    }
                }
                Err(e) => {
                    window
                        .emit("log", format!("Network error: {}, retrying...", e))
                        .ok();
                    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                    retries -= 1;
                }
            }
        }
        let resp = resp.ok_or("Failed to download python after retries")?;

        // Streaming download logic
        let total_size = resp.content_length().unwrap_or(0);
        let mut stream = resp.bytes_stream();
        let mut downloaded: u64 = 0;
        let mut buffer = Vec::new();
        let mut last_emit = std::time::Instant::now();

        while let Some(item) = stream.next().await {
            let chunk = item.map_err(|e| format!("Download error: {}", e))?;
            downloaded += chunk.len() as u64;
            buffer.extend_from_slice(&chunk);

            // Throttle updates to ~100ms to avoid flooding frontend
            if last_emit.elapsed().as_millis() > 100 {
                let progress_percent = if total_size > 0 {
                    (downloaded as f64 / total_size as f64) * 100.0
                } else {
                    0.0
                };

                // Emit detailed stats payload
                window
                    .emit(
                        "setup_progress",
                        serde_json::json!({
                            "step": "download",
                            "message": "Downloading Python...",
                            "progress": progress_percent,
                            "loaded": downloaded,
                            "total": total_size
                        }),
                    )
                    .ok();

                last_emit = std::time::Instant::now();
            }
        }

        // Final download progress event
        window
            .emit(
                "setup_progress",
                serde_json::json!({
                    "step": "download",
                    "message": "Download complete. Extracting...",
                    "progress": 100.0,
                    "loaded": downloaded,
                    "total": total_size
                }),
            )
            .ok();

        let content = bytes::Bytes::from(buffer);

        window.emit("setup_progress", serde_json::json!({ "step": "download", "message": "Preparing extraction...", "progress": 100 })).ok();

        let target_dir = get_python_standalone_dir(&app_handle);
        let project_root = target_dir.parent().ok_or("No root")?.to_path_buf();

        // Delete existing Python directory if it exists
        if target_dir.exists() {
            window.emit("setup_progress", serde_json::json!({ "step": "cleanup", "message": "Cleaning up old Python installation...", "progress": 100 })).ok();

            // Kill any running python/tensorboard processes to release file locks
            #[cfg(target_os = "windows")]
            {
                let _ = create_hidden_std_command("taskkill")
                    .args(&["/F", "/IM", "python.exe"])
                    .output();
                let _ = create_hidden_std_command("taskkill")
                    .args(&["/F", "/IM", "tensorboard.exe"])
                    .output();
                // Give OS time to release handles
                tokio::time::sleep(tokio::time::Duration::from_millis(2000)).await;
            }
            #[cfg(not(target_os = "windows"))]
            {
                let _ = create_hidden_std_command("pkill").arg("python").output();
                let _ = create_hidden_std_command("pkill")
                    .arg("tensorboard")
                    .output();
                tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;
            }

            if let Err(e) = std::fs::remove_dir_all(&target_dir) {
                // Retry once after a delay
                tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;
                if let Err(e2) = std::fs::remove_dir_all(&target_dir) {
                    window
                        .emit(
                            "log",
                            format!(
                                "Warning: Could not fully remove old Python directory: {}. {}",
                                e, e2
                            ),
                        )
                        .ok();
                }
            }
        }

        window.emit("setup_progress", serde_json::json!({ "step": "extract", "message": "Extracting Python...", "progress": 100 })).ok();

        tokio::task::spawn_blocking(move || {
            let tar = flate2::read::GzDecoder::new(std::io::Cursor::new(content));
            let mut archive = tar::Archive::new(tar);
            archive.unpack(&project_root).map_err(|e| e.to_string())
        })
        .await
        .map_err(|e| e.to_string())??;

        window
            .emit(
                "setup_progress",
                serde_json::json!({ "step": "complete", "message": "Installed!", "progress": 100 }),
            )
            .ok();
        Ok("Installed".to_string())
    }
}

#[tauri::command]
pub async fn debug_python_path_command(app_handle: AppHandle) -> Result<String, String> {
    let (python_exe, work_dir) = get_python_command(&app_handle)?;
    Ok(format!(
        "Python Exe: {}, Working Dir: {:?}",
        python_exe, work_dir
    ))
}

#[tauri::command]
pub async fn take_screenshot_path_command(
    app_handle: AppHandle,
    output_dir: String,
    _delay: f64,
) -> Result<String, String> {
    // Calls the python screenshot tool
    let (python_exe, work_dir) = get_python_command(&app_handle)?;
    let script_path = get_script_path(&app_handle, "screenshot_tool.py");

    let mut cmd = create_hidden_command(&python_exe);
    cmd.arg(&script_path)
        .arg("--output_dir")
        .arg(&output_dir)
        .current_dir(&work_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let output = cmd.output().await.map_err(|e| e.to_string())?;

    // The script prints the relative path to stderr for us to capture
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Look for "INFO: Screenshot saved to "
    for line in stderr.lines().chain(stdout.lines()) {
        if let Some(path) = line.trim().strip_prefix("INFO: Screenshot saved to ") {
            return Ok(path.trim().to_string());
        }
    }

    Err(format!(
        "Screenshot failed. Output: {}\nError: {}",
        stdout, stderr
    ))
}

#[derive(Serialize, Deserialize)]
struct AnnotationData {
    dataset_name: String,
    image_path: String,
    prompt: String,
    x: f64,
    y: f64,
    width: f64,
    height: f64,
    screen_width: f64,
    screen_height: f64,
}

#[tauri::command]
pub async fn save_annotation_crop_command(
    app_handle: AppHandle,
    data: String,
) -> Result<String, String> {
    let annotation: AnnotationData = serde_json::from_str(&data).map_err(|e| e.to_string())?;

    let data_dir = get_data_dir(&app_handle);
    let dataset_dir = data_dir
        .join("data/datasets")
        .join(&annotation.dataset_name);
    let processed_dir = dataset_dir.join("processed_data");
    let dataset_file = processed_dir.join("dataset.jsonl");

    // Ensure processed_dir exists
    if !processed_dir.exists() {
        fs::create_dir_all(&processed_dir)
            .await
            .map_err(|e| e.to_string())?;
    }

    // Normalize coordinates like data_collecter.py does (1-1000 range)
    let norm_x = (annotation.x / annotation.screen_width * 1000.0).clamp(1.0, 1000.0) as i32;
    let norm_y = (annotation.y / annotation.screen_height * 1000.0).clamp(1.0, 1000.0) as i32;

    // Construct the entry
    let entry = serde_json::json!({
        "image": annotation.image_path, // Path relative to src-tauri root or absolute?
        // Ideally should be relative to dataset root or something standard.
        // For now trusting what frontend/screenshot tool gave us.
        "conversations": [
            {
                "role": "user",
                "content": format!("\n{}", annotation.prompt)
            },
            {
                "role": "assistant",
                "content": format!("<tool_call>{{\"name\": \"computer_use\", \"arguments\": {{\"action\": \"left_click\", \"coordinate\": [{}, {}]}}}}</tool_call>", norm_x, norm_y)
            }
        ],
        "metadata": {
            "timestamp": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
                .to_string(),
            "coordinates_normalized": [norm_x, norm_y],
            "coordinates_raw": [annotation.x, annotation.y],
            "screen_size": [annotation.screen_width, annotation.screen_height]
        }
    });

    // Append to jsonl
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&dataset_file)
        .await
        .map_err(|e| e.to_string())?;

    use tokio::io::AsyncWriteExt;
    file.write_all(format!("{}\n", entry.to_string()).as_bytes())
        .await
        .map_err(|e| e.to_string())?;

    Ok("Annotation saved".to_string())
}

#[tauri::command]
pub async fn get_tensorboard_url_command(
    python_process_state: State<'_, Arc<PythonProcessState>>,
) -> Result<String, String> {
    let port_guard = python_process_state.tensorboard_port.lock().await;
    if let Some(port) = *port_guard {
        Ok(format!("http://127.0.0.1:{}", port))
    } else {
        Err("TensorBoard is not running".to_string())
    }
}

#[tauri::command]
pub async fn start_data_collector_command(
    python_process_state: State<'_, Arc<PythonProcessState>>,
    app_handle: AppHandle,
    output_dir: String,
    dataset_file: String,
) -> Result<String, String> {
    let (python_exe, work_dir) = get_python_command(&app_handle)?;
    let script_path = get_script_path(&app_handle, "data_collecter.py"); // Ensure filename matches script on disk

    // Kill existing if any
    {
        let mut guard = python_process_state.data_collector_child.lock().await;
        if let Some(child) = guard.as_mut() {
            let _ = child.kill().await;
            *guard = None;
        }
    }

    // Start new process
    let mut cmd = create_hidden_command(&python_exe);
    cmd.arg(&script_path)
        .arg("--output_dir")
        .arg(&output_dir)
        .arg("--dataset_file")
        .arg(&dataset_file)
        .current_dir(&work_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let child = cmd.spawn().map_err(|e| e.to_string())?;

    let mut guard = python_process_state.data_collector_child.lock().await;
    *guard = Some(child);

    Ok("Data collector started".to_string())
}

#[tauri::command]
pub async fn list_gguf_models_command(app_handle: AppHandle) -> Result<Vec<String>, String> {
    // Reusing list_all_resources logic or simple dir list
    let data_dir = get_data_dir(&app_handle);
    let models_dir = data_dir.join("data/models");
    let mut res = Vec::new();
    if let Ok(mut entries) = fs::read_dir(models_dir).await {
        while let Ok(Some(entry)) = entries.next_entry().await {
            let path = entry.path();
            if path.extension().map_or(false, |e| e == "gguf") {
                res.push(entry.file_name().to_string_lossy().to_string());
            } else if path.is_dir() {
                if check_dir_for_gguf(&path).await {
                    // It's a GGUF folder? or HF folder with GGUF?
                    // Just list the folder name
                    res.push(entry.file_name().to_string_lossy().to_string());
                }
            }
        }
    }
    Ok(res)
}

#[tauri::command]
pub async fn list_lora_adapters_command(app_handle: AppHandle) -> Result<Vec<String>, String> {
    let data_dir = get_data_dir(&app_handle);
    let mut res = Vec::new();

    // 1. Standard loras directory
    let loras_dir = data_dir.join("data/loras");
    if let Ok(mut entries) = fs::read_dir(&loras_dir).await {
        while let Ok(Some(entry)) = entries.next_entry().await {
            res.push(entry.file_name().to_string_lossy().to_string());
        }
    }

    // 2. Training outputs (projects)
    let outputs_dir = data_dir.join("data/outputs");
    if let Ok(mut entries) = fs::read_dir(&outputs_dir).await {
        while let Ok(Some(entry)) = entries.next_entry().await {
            let path = entry.path();
            if path.is_dir() {
                // If final_model exists, it's a LoRA output
                let final_model = path.join("final_model");
                if final_model.exists() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    res.push(format!("outputs/{}", name));
                }
            }
        }
    }

    Ok(res)
}

#[tauri::command]
pub async fn delete_project_command(
    app_handle: AppHandle,
    project_name: String,
) -> Result<(), String> {
    let data_dir = get_data_dir(&app_handle);
    let project_dir = data_dir.join("data/outputs").join(&project_name);
    if project_dir.exists() {
        fs::remove_dir_all(project_dir)
            .await
            .map_err(|e| e.to_string())?;
        Ok(())
    } else {
        Err("Project not found".to_string())
    }
}

#[tauri::command]
pub async fn setup_python_env_command(
    window: Window,
    app_handle: AppHandle,
) -> Result<String, String> {
    let (python_exe, work_dir) = get_python_command(&app_handle)?;

    // Helper to run pip commands
    let run_pip = |args: Vec<&str>, step_name: &str, progress: u8| {
        let window = window.clone();
        let python_exe = python_exe.clone();
        let work_dir = work_dir.clone();
        let args: Vec<String> = args.into_iter().map(|s| s.to_string()).collect();
        let step_name = step_name.to_string();

        async move {
            window.emit("setup_progress", serde_json::json!({ "step": "install", "message": format!("Installing {}...", step_name), "progress": progress })).ok();

            let mut cmd = create_hidden_command(&python_exe);
            cmd.arg("-m").arg("pip").arg("install").args(&args);
            cmd.current_dir(&work_dir)
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
                    win_c.emit("log", format!("SETUP: {}", line.trim())).ok();
                    line.clear();
                }
            });

            let win_c = window.clone();
            tokio::spawn(async move {
                let mut reader = BufReader::new(stderr);
                let mut line = String::new();
                while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
                    win_c.emit("log", format!("SETUP: {}", line.trim())).ok();
                    line.clear();
                }
            });

            let status = child.wait().await.map_err(|e| e.to_string())?;
            if status.success() {
                Ok(())
            } else {
                Err(format!("Failed to install {}", step_name))
            }
        }
    };

    // Define dependencies based on OS
    let is_windows = cfg!(windows);

    // 1. PyTorch
    // torch 2.6.0 + torchvision 0.21.0 are required for Unsloth-zoo's int1 dtype
    // These WHLs are generally cross-platform (linux/win) at this index-url,
    // but Linux might prefer the linux-specific index if needed.
    // Usually download.pytorch.org/whl/cu124 works for both.
    run_pip(
        vec![
            "--upgrade",
            "--force-reinstall",
            "torch==2.6.0", // Remove +cu124 from name to let index resolve, or keep specific? +cu124 is safer
            "torchvision==0.21.0",
            "torchaudio==2.6.0",
            "--index-url",
            "https://download.pytorch.org/whl/cu124",
        ],
        "PyTorch (CUDA 12.4)",
        10,
    )
    .await?;

    // 2. Core Dependencies
    let mut core_deps = vec![
        "--upgrade",
        "cmake",
        "setuptools",
        "wheel",
        "trl",
        "peft",
        "accelerate",
        "bitsandbytes",
        "tensorboard",
        "huggingface_hub[hf_xet]",
        "requests",
        "uvicorn",
        "fastapi",
        "jinja2",
        "python-multipart",
        "gguf @ git+https://github.com/ggerganov/llama.cpp.git#subdirectory=gguf-py",
    ];

    if is_windows {
        // triton-windows 3.2.x is required for PyTorch 2.6 on Windows
        core_deps.push("triton-windows<3.3");
    } else {
        // Standard triton for Linux
        core_deps.push("triton");
    }

    run_pip(core_deps, "Core Dependencies", 40).await?;

    // 3. Unsloth
    let unsloth_pkg = if is_windows {
        "unsloth[windows] @ git+https://github.com/unslothai/unsloth.git"
    } else {
        "unsloth @ git+https://github.com/unslothai/unsloth.git"
    };

    run_pip(vec![unsloth_pkg], "Unsloth", 70).await?;

    // 4. Reinstall PyTorch CUDA
    // Unsloth might pull CPU versions on some systems, so we force reinstall GPU version just in case
    run_pip(
        vec![
            "--upgrade",
            "--force-reinstall",
            "torch==2.6.0",
            "torchvision==0.21.0",
            "torchaudio==2.6.0",
            "--index-url",
            "https://download.pytorch.org/whl/cu124",
        ],
        "PyTorch CUDA Restore",
        90,
    )
    .await?;

    window.emit("setup_progress", serde_json::json!({ "step": "complete", "message": "Environment Setup Complete!", "progress": 100 })).ok();
    Ok("Setup complete".to_string())
}

// --- TOOL EXECUTION ---

#[tauri::command]
pub async fn save_custom_tool_command(
    app_handle: tauri::AppHandle,
    tool_name: String,
    tool_code: String,
) -> Result<String, String> {
    use tauri::Manager;
    let resource_dir = app_handle
        .path()
        .resource_dir()
        .map_err(|e| e.to_string())?;

    // In built app, resources are flattened or strictly structure?
    // Usually resources are in 'resources' folder.
    // If 'scripts' is a resource, it should be in resource_dir/scripts.
    let scripts_dir = resource_dir.join("scripts");

    let custom_tools_dir = scripts_dir.join("custom_tools");
    if !custom_tools_dir.exists() {
        std::fs::create_dir_all(&custom_tools_dir).map_err(|e| e.to_string())?;
    }

    let file_path = custom_tools_dir.join(format!("{}.py", tool_name));
    std::fs::write(&file_path, tool_code).map_err(|e| e.to_string())?;

    Ok(format!("Tool {} saved successfully", tool_name))
}

#[tauri::command]
pub async fn execute_tool_command(
    app_handle: AppHandle,
    tool_name: String,
    args: serde_json::Value,
) -> Result<serde_json::Value, String> {
    use std::process::Stdio;

    let (python_exe, _cwd) = get_python_command(&app_handle)?;
    let script_path = get_script_path(&app_handle, "tools.py");

    debug!("Executing tool: {} with args: {:?}", tool_name, args);

    let mut cmd = create_hidden_command(&python_exe);
    cmd.arg(&script_path)
        .arg(&tool_name)
        .arg(serde_json::to_string(&args).unwrap_or_default())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let output = cmd
        .output()
        .await
        .map_err(|e| format!("Failed to execute tool: {}", e))?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    if !output.status.success() {
        error!("Tool execution failed: {}", stderr);
        return Err(format!("Tool execution failed: {}", stderr));
    }

    // Parse the JSON result
    let result: serde_json::Value = serde_json::from_str(&stdout)
        .map_err(|e| format!("Failed to parse tool result: {} - Output: {}", e, stdout))?;

    Ok(result)
}

#[tauri::command]
pub async fn check_server_health_command(port: u16) -> Result<bool, String> {
    let client = reqwest::Client::new();
    let res = client
        .get(format!("http://127.0.0.1:{}/health", port))
        .timeout(std::time::Duration::from_millis(500))
        .send()
        .await;

    match res {
        Ok(r) => Ok(r.status().is_success()),
        Err(_) => Ok(false),
    }
}

#[tauri::command]
pub async fn resolve_path_command(app_handle: AppHandle, path: String) -> Result<String, String> {
    // If absolute, return as is (normalized)
    let p = PathBuf::from(&path);
    if p.is_absolute() {
        return Ok(p.to_string_lossy().to_string());
    }

    let data_dir = get_data_dir(&app_handle);

    // Check various common locations
    let candidates = vec![
        data_dir.join(&path),
        data_dir.join("data").join("models").join(&path),
        data_dir.join("data").join("datasets").join(&path),
        data_dir.join("data").join("loras").join(&path),
        // Handling for direct data/ paths
        data_dir.join("data").join(&path),
    ];

    for candidate in candidates {
        if candidate.exists() {
            return Ok(candidate.to_string_lossy().to_string());
        }
    }

    // Sometimes path already contains "data/...", check parent of data_dir
    if path.starts_with("data") {
        if let Some(parent) = data_dir.parent() {
            let retry = parent.join(&path);
            if retry.exists() {
                return Ok(retry.to_string_lossy().to_string());
            }
        }
    }

    // Fallback: return best effort absolute path relative to data dir
    Ok(data_dir.join(path).to_string_lossy().to_string())
}

#[tauri::command]
pub async fn create_dataset_command(app_handle: AppHandle, name: String) -> Result<String, String> {
    println!("BACKEND: create_dataset_command called with name: {}", name);
    let data_dir = get_data_dir(&app_handle);
    // Sanitize name
    let safe_name = name.replace(|c: char| !c.is_alphanumeric() && c != '-' && c != '_', "_");

    // Create folder: data/datasets/{safe_name}
    let dataset_folder = data_dir.join("data/datasets").join(&safe_name);
    println!("BACKEND: Creating dataset folder at: {:?}", dataset_folder);

    if dataset_folder.exists() {
        println!("BACKEND: Folder already exists!");
        return Err(format!("Dataset folder '{}' already exists", safe_name));
    }

    fs::create_dir_all(&dataset_folder)
        .await
        .map_err(|e| e.to_string())?;

    // Create empty file: dataset.jsonl
    let file_path = dataset_folder.join("dataset.jsonl");
    fs::File::create(&file_path)
        .await
        .map_err(|e| e.to_string())?;

    let relative_path = format!("data/datasets/{}", safe_name);
    println!(
        "BACKEND: Success. Returning relative path: {}",
        relative_path
    );
    // Return relative path for UI selection match
    Ok(relative_path)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DatasetPreviewResult {
    pub rows: Vec<serde_json::Value>,
    #[serde(rename = "totalCount")]
    pub total_count: i64,
    pub columns: Vec<String>,
    pub format: String,
    pub error: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DatasetFixedResult {
    pub success: bool,
    pub output_path: Option<String>,
    pub error: Option<String>,
}

#[tauri::command]
pub async fn fix_dataset_command(
    app_handle: AppHandle,
    dataset_dir: String,
) -> Result<DatasetFixedResult, String> {
    let (python_exe, work_dir) = get_python_command(&app_handle)?;
    let script = get_script_path(&app_handle, "fix_dataset.py");

    let resolved_dir = resolve_path_command(app_handle.clone(), dataset_dir.clone()).await?;

    let mut cmd = create_hidden_std_command(&python_exe);
    cmd.arg(&script)
        .arg("--dataset_dir")
        .arg(&resolved_dir)
        .current_dir(&work_dir);

    let output = cmd
        .output()
        .map_err(|e| format!("Failed to run fix_dataset: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        // It might print to stdout even on error, so capture that too
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Ok(DatasetFixedResult {
            success: false,
            output_path: None,
            error: Some(format!("Error: {}\nOutput: {}", stderr, stdout)),
        });
    }

    // The script prints the output file path as the last line usually?
    // Or we just assume it's created. The script in fix_dataset.py returns the output path or prints it?
    // Looking at fix_dataset.py: it prints stats and output file.
    // We might want to adjust fix_dataset.py to print JSON result for better parsing,
    // OR just assume success and return the expected path.
    // For now, let's assume it worked if exit code 0.

    Ok(DatasetFixedResult {
        success: true,
        output_path: Some(resolved_dir + "/dataset_fixed.jsonl"), // Simple assumption based on script default
        error: None,
    })
}

#[tauri::command]
pub async fn bulk_edit_dataset_command(
    app_handle: AppHandle,
    dataset_path: String,
    operation: String, // JSON
) -> Result<DatasetSaveResult, String> {
    let (python_exe, work_dir) = get_python_command(&app_handle)?;
    let script = get_script_path(&app_handle, "dataset_loader.py");

    let resolved_path = resolve_path_command(app_handle.clone(), dataset_path.clone()).await?;

    let mut cmd = create_hidden_std_command(&python_exe);
    cmd.arg(&script)
        .arg("bulk_edit")
        .arg(&resolved_path)
        .arg("--operation")
        .arg(&operation)
        .current_dir(&work_dir);

    let output = cmd
        .output()
        .map_err(|e| format!("Failed to run dataset_loader bulk_edit: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Bulk edit error: {}", stderr));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let result: DatasetSaveResult = serde_json::from_str(&stdout).map_err(|e| {
        format!(
            "Failed to parse bulk edit result: {} - Output: {}",
            e, stdout
        )
    })?;

    Ok(result)
}

#[tauri::command]
pub async fn load_dataset_preview_command(
    app_handle: AppHandle,
    dataset_path: String,
    offset: i64,
    limit: i64,
    split: Option<String>,
) -> Result<DatasetPreviewResult, String> {
    let (python_exe, work_dir) = get_python_command(&app_handle)?;
    let script = get_script_path(&app_handle, "dataset_loader.py");

    // Resolve dataset path
    let resolved_path = {
        let p = PathBuf::from(&dataset_path);
        if p.is_absolute() {
            p
        } else {
            let data_dir = get_data_dir(&app_handle);
            if dataset_path.starts_with("data") || dataset_path.contains("data\\") {
                data_dir.join(&dataset_path)
            } else {
                data_dir.join("data/datasets").join(&dataset_path)
            }
        }
    };

    let mut cmd = create_hidden_std_command(&python_exe);
    cmd.arg(&script)
        .arg("load")
        .arg(resolved_path.to_string_lossy().to_string())
        .arg("--offset")
        .arg(offset.to_string())
        .arg("--limit")
        .arg(limit.to_string())
        .arg("--split")
        .arg(split.unwrap_or_else(|| "train".to_string()))
        .current_dir(&work_dir);

    let output = cmd
        .output()
        .map_err(|e| format!("Failed to run dataset_loader: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Dataset loader error: {}", stderr));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let result: DatasetPreviewResult = serde_json::from_str(&stdout).map_err(|e| {
        format!(
            "Failed to parse dataset loader output: {} - Output: {}",
            e, stdout
        )
    })?;

    Ok(result)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DatasetSaveResult {
    pub success: bool,
    pub count: Option<i64>,
    pub error: Option<String>,
}

#[tauri::command]
pub async fn save_dataset_command(
    app_handle: AppHandle,
    dataset_path: String,
    rows: String,
) -> Result<DatasetSaveResult, String> {
    let (python_exe, work_dir) = get_python_command(&app_handle)?;
    let script = get_script_path(&app_handle, "dataset_loader.py");

    // Resolve path (similar logic to load)
    let resolved_path = resolve_path_command(app_handle.clone(), dataset_path.clone()).await?;

    let mut cmd = create_hidden_std_command(&python_exe);
    cmd.arg(&script)
        .arg("save")
        .arg(&resolved_path)
        .arg("--data")
        .arg(&rows)
        .current_dir(&work_dir);

    let output = cmd
        .output()
        .map_err(|e| format!("Failed to run dataset_loader save: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Dataset save error: {}", stderr));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let result: DatasetSaveResult = serde_json::from_str(&stdout)
        .map_err(|e| format!("Failed to parse save result: {} - Output: {}", e, stdout))?;

    Ok(result)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DatasetEdit {
    #[serde(rename = "rowIndex")]
    pub row_index: i64,
    pub data: Option<serde_json::Value>, // None means delete
}

#[tauri::command]
pub async fn apply_dataset_edits_command(
    app_handle: AppHandle,
    dataset_path: String,
    edits: String, // JSON string of DatasetEdit[]
) -> Result<DatasetSaveResult, String> {
    let (python_exe, work_dir) = get_python_command(&app_handle)?;
    let script = get_script_path(&app_handle, "dataset_loader.py");

    let resolved_path = resolve_path_command(app_handle.clone(), dataset_path.clone()).await?;

    let mut cmd = create_hidden_std_command(&python_exe);
    cmd.arg(&script)
        .arg("edit")
        .arg(&resolved_path)
        .arg("--edits")
        .arg(&edits)
        .current_dir(&work_dir);

    let output = cmd
        .output()
        .map_err(|e| format!("Failed to run dataset_loader edit: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Dataset edit error: {}", stderr));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let result: DatasetSaveResult = serde_json::from_str(&stdout)
        .map_err(|e| format!("Failed to parse edit result: {} - Output: {}", e, stdout))?;

    Ok(result)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GenerationResult {
    pub success: bool,
    pub output_path: String,
    pub error: Option<String>,
}

#[tauri::command]
pub async fn generate_dataset_command(
    app_handle: AppHandle,
    recipe: String, // JSON string
    output_path: String,
) -> Result<GenerationResult, String> {
    let (python_exe, work_dir) = get_python_command(&app_handle)?;
    let script = get_script_path(&app_handle, "generation_engine.py");

    // Resolve output path
    // If relative, put in data/datasets
    let target_path = if std::path::Path::new(&output_path).is_absolute() {
        output_path.clone()
    } else {
        get_data_dir(&app_handle)
            .join("datasets")
            .join(&output_path)
            .to_string_lossy()
            .to_string()
    };

    let mut cmd = create_hidden_std_command(&python_exe);
    cmd.arg(&script)
        .arg("--recipe")
        .arg(&recipe)
        .arg("--output")
        .arg(&target_path)
        .current_dir(&work_dir);

    // TODO: Ideally stream output. For now wait.
    let output = cmd
        .output()
        .map_err(|e| format!("Failed to run generation engine: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        // Also check stdout for JSON error from script
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(format!("Generation error: {} \nOutput: {}", stderr, stdout));
    }

    Ok(GenerationResult {
        success: true,
        output_path: target_path,
        error: None,
    })
}

// --- Drop Analysis Types ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DropDetails {
    pub is_split_model: bool,
    pub has_vision: bool,
    pub file_count: u32,
    pub total_size: String,
    pub detected_format: Option<String>,
    pub shard_info: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DropAnalysis {
    pub path: String,
    pub name: String,
    pub resource_type: String, // "model", "gguf", "lora", "dataset", "unknown"
    pub confidence: String,    // "high", "medium", "low"
    pub details: DropDetails,
}

// --- Drop Analysis Command ---

#[tauri::command]
pub async fn analyze_drop_command(paths: Vec<String>) -> Result<Vec<DropAnalysis>, String> {
    let mut results = Vec::new();

    for path_str in paths {
        let path = PathBuf::from(&path_str);
        if !path.exists() {
            continue;
        }

        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "Unknown".to_string());

        let analysis = if path.is_file() {
            analyze_file(&path, &name).await
        } else {
            analyze_directory(&path, &name).await
        };

        results.push(analysis);
    }

    Ok(results)
}

async fn analyze_file(path: &PathBuf, name: &str) -> DropAnalysis {
    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default();

    let file_size = fs::metadata(path).await.map(|m| m.len()).unwrap_or(0);

    let size_str = crate::utils::format_size(file_size);

    // Check for GGUF files
    if ext == "gguf" {
        let is_mmproj = name.to_lowercase().contains("mmproj");
        let is_split = is_split_model_file(name);
        let shard_info = if is_split {
            parse_shard_info(name)
        } else {
            None
        };

        return DropAnalysis {
            path: path.to_string_lossy().to_string(),
            name: name.to_string(),
            resource_type: if is_mmproj {
                "mmproj".to_string()
            } else {
                "gguf".to_string()
            },
            confidence: "high".to_string(),
            details: DropDetails {
                is_split_model: is_split,
                has_vision: is_mmproj,
                file_count: 1,
                total_size: size_str,
                detected_format: Some("gguf".to_string()),
                shard_info,
            },
        };
    }

    // Check for dataset files
    let dataset_exts = ["jsonl", "parquet", "arrow", "csv", "json"];
    if dataset_exts.contains(&ext.as_str()) {
        return DropAnalysis {
            path: path.to_string_lossy().to_string(),
            name: name.to_string(),
            resource_type: "dataset".to_string(),
            confidence: "high".to_string(),
            details: DropDetails {
                is_split_model: false,
                has_vision: false,
                file_count: 1,
                total_size: size_str,
                detected_format: Some(ext),
                shard_info: None,
            },
        };
    }

    // Check for LoRA files
    let lora_exts = ["safetensors", "bin", "pt"];
    if lora_exts.contains(&ext.as_str()) && name.to_lowercase().contains("adapter") {
        return DropAnalysis {
            path: path.to_string_lossy().to_string(),
            name: name.to_string(),
            resource_type: "lora".to_string(),
            confidence: "medium".to_string(),
            details: DropDetails {
                is_split_model: false,
                has_vision: false,
                file_count: 1,
                total_size: size_str,
                detected_format: Some(ext),
                shard_info: None,
            },
        };
    }

    // Unknown file type
    DropAnalysis {
        path: path.to_string_lossy().to_string(),
        name: name.to_string(),
        resource_type: "unknown".to_string(),
        confidence: "low".to_string(),
        details: DropDetails {
            is_split_model: false,
            has_vision: false,
            file_count: 1,
            total_size: size_str,
            detected_format: Some(ext),
            shard_info: None,
        },
    }
}

async fn analyze_directory(path: &PathBuf, name: &str) -> DropAnalysis {
    let mut gguf_count = 0u32;
    let mut mmproj_count = 0u32;
    let mut safetensor_count = 0u32;
    let mut dataset_file_count = 0u32;
    let mut has_config_json = false;
    let mut has_adapter_config = false;
    let mut total_size = 0u64;
    let mut file_count = 0u32;
    let mut is_split = false;
    let mut shard_info: Option<String> = None;

    // Scan directory contents
    if let Ok(mut entries) = fs::read_dir(path).await {
        while let Ok(Some(entry)) = entries.next_entry().await {
            let entry_name = entry.file_name().to_string_lossy().to_lowercase();
            let entry_path = entry.path();

            if let Ok(metadata) = entry.metadata().await {
                if metadata.is_file() {
                    file_count += 1;
                    total_size += metadata.len();

                    let ext = entry_path
                        .extension()
                        .map(|e| e.to_string_lossy().to_lowercase())
                        .unwrap_or_default();

                    // Check file types
                    if ext == "gguf" {
                        if entry_name.contains("mmproj") {
                            mmproj_count += 1;
                        } else {
                            gguf_count += 1;
                            if is_split_model_file(&entry_name) {
                                is_split = true;
                                if shard_info.is_none() {
                                    shard_info = parse_shard_info(&entry_name);
                                }
                            }
                        }
                    } else if ext == "safetensors" {
                        safetensor_count += 1;
                    } else if ["jsonl", "parquet", "arrow", "csv"].contains(&ext.as_str()) {
                        dataset_file_count += 1;
                    }

                    // Check for config files
                    if entry_name == "config.json" {
                        has_config_json = true;
                    }
                    if entry_name == "adapter_config.json"
                        || entry_name == "adapter_model.safetensors"
                    {
                        has_adapter_config = true;
                    }
                }
            }
        }
    }

    let size_str = crate::utils::format_size(total_size);
    let has_vision = mmproj_count > 0;

    // Determine resource type based on contents
    let (resource_type, confidence, detected_format) = if has_adapter_config {
        // LoRA adapter folder
        (
            "lora".to_string(),
            "high".to_string(),
            Some("lora".to_string()),
        )
    } else if gguf_count > 0 {
        // GGUF model folder (possibly with vision)
        if has_vision {
            (
                "model".to_string(),
                "high".to_string(),
                Some("gguf+vision".to_string()),
            )
        } else if is_split {
            (
                "model".to_string(),
                "high".to_string(),
                Some("gguf-split".to_string()),
            )
        } else {
            (
                "gguf".to_string(),
                "high".to_string(),
                Some("gguf".to_string()),
            )
        }
    } else if safetensor_count > 0 && has_config_json {
        // HuggingFace model folder
        (
            "model".to_string(),
            "high".to_string(),
            Some("safetensors".to_string()),
        )
    } else if dataset_file_count > 0 {
        // Dataset folder
        (
            "dataset".to_string(),
            "high".to_string(),
            Some("dataset-folder".to_string()),
        )
    } else if safetensor_count > 0 {
        // Could be LoRA or model, lower confidence
        (
            "model".to_string(),
            "medium".to_string(),
            Some("safetensors".to_string()),
        )
    } else {
        ("unknown".to_string(), "low".to_string(), None)
    };

    DropAnalysis {
        path: path.to_string_lossy().to_string(),
        name: name.to_string(),
        resource_type,
        confidence,
        details: DropDetails {
            is_split_model: is_split,
            has_vision,
            file_count,
            total_size: size_str,
            detected_format,
            shard_info,
        },
    }
}

fn is_split_model_file(filename: &str) -> bool {
    let lower = filename.to_lowercase();
    // Common patterns for split models:
    // model-00001-of-00003.gguf
    // model.gguf.part1of3
    // model-shard-001.gguf
    lower.contains("-of-")
        || lower.contains("part")
            && (lower.contains("of") || lower.chars().filter(|c| c.is_ascii_digit()).count() >= 2)
        || lower.contains("-shard-")
}

fn parse_shard_info(filename: &str) -> Option<String> {
    let lower = filename.to_lowercase();

    // Try to parse "00001-of-00003" pattern
    if let Some(of_idx) = lower.find("-of-") {
        // Find the number before "-of-"
        let before = &lower[..of_idx];
        let after = &lower[of_idx + 4..];

        // Extract current shard number (last digits before -of-)
        let current: String = before
            .chars()
            .rev()
            .take_while(|c| c.is_ascii_digit())
            .collect::<String>()
            .chars()
            .rev()
            .collect();
        // Extract total shards (digits right after -of-)
        let total: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();

        if let (Ok(c), Ok(t)) = (current.parse::<u32>(), total.parse::<u32>()) {
            return Some(format!("{} of {} shards", c, t));
        }
    }

    None
}
