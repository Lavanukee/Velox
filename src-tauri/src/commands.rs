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

use crate::constants::{HF_TOKEN_FILE, PRESETS_FILE};
use crate::hardware::{detect_backend, ComputeBackend};
use crate::models::{CheckpointInfo, ModelConfig, ModelsState, ProjectLoraInfo};
use crate::python::{get_python_command, get_script_path, LlamaChatContext, PythonProcessState};
use crate::utils::{
    check_dir_for_gguf, copy_dir_all, create_hidden_command, create_hidden_std_command,
    detect_dataset_format, detect_quantization, get_data_dir, get_dir_size, get_file_size,
};

// --- Helper Functions Local to Commands ---

fn get_binaries_dir(app_handle: &AppHandle) -> PathBuf {
    get_data_dir(app_handle).join("binaries")
}

// --- COMMANDS ---

#[tauri::command]
pub async fn check_python_installed_command(app_handle: AppHandle) -> Result<bool, String> {
    match get_python_command(&app_handle) {
        Ok((python_exe, _)) => {
            // Verify it actually runs
            let output = create_hidden_command(&python_exe)
                .arg("--version")
                .output()
                .await
                .map_err(|e| e.to_string())?;
            Ok(output.status.success())
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

    // On Windows with CUDA, verify the DLL exists too, otherwise we might be stuck with CPU-only binaries
    if exists && cfg!(windows) {
        if let ComputeBackend::Cuda = detect_backend() {
            let cuda_dll = bin_dir.join("ggml-cuda.dll");
            if !cuda_dll.exists() {
                debug!("Llama binary exists but ggml-cuda.dll is missing. Triggering re-download.");
                exists = false;
            }
        }
    }

    debug!("Llama binary valid: {}", exists);
    Ok(exists)
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
        window
            .emit("log", "Engine installed successfully!".to_string())
            .unwrap();
        Ok("Installation complete.".into())
    } else {
        Err("Installation finished, but llama-server binary is missing.".into())
    }
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
    debug!("Listing dataset folders.");
    let data_dir = get_data_dir(&app_handle);
    let datasets_dir = data_dir.join("data").join("datasets");
    let mut folders = Vec::new();
    if datasets_dir.exists() && datasets_dir.is_dir() {
        if let Ok(mut entries) = fs::read_dir(datasets_dir).await {
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
                    });
                }
            }
        }
        if !manual_checkpoints.is_empty() {
            projects_info.push(ProjectLoraInfo {
                project_name: "Manual/External".to_string(),
                checkpoints: manual_checkpoints,
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
) -> Result<String, String> {
    // Kill existing if any
    {
        let mut child_guard = python_process_state.llama_server_child.lock().await;
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

    // Resolve model path to absolute if needed
    let resolved_model_path = {
        let p = PathBuf::from(&model_path);
        if p.exists() && p.is_absolute() {
            p
        } else {
            let data_dir = get_data_dir(&app_handle);
            let check_p = data_dir.join("data").join("models").join(&model_path);
            if check_p.exists() {
                check_p
            } else {
                // Fallback: try relative to CWD just in case, or default to constructed path
                // to let llama-server error out with the full path if missing.
                check_p
            }
        }
    };

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
        gpu_layers.unwrap_or(0).to_string(),
        "--host".to_string(),
        "0.0.0.0".to_string(),
    ];

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
    // "threads" arg is often auto-detected or --threads, omitting if not strictly required or use defaults.

    let mut child_guard = python_process_state.llama_server_child.lock().await;

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
                win_c
                    .emit("log", format!("LLAMA_SERVER: {}", line.trim()))
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
) -> Result<String, String> {
    let mut child_guard = python_process_state.llama_server_child.lock().await;
    if let Some(child) = child_guard.as_mut() {
        match child.kill().await {
            Ok(_) => {
                *child_guard = None;
                Ok("Llama server stopped.".to_string())
            }
            Err(e) => Err(format!("Failed to stop Llama server: {}", e)),
        }
    } else {
        Err("Llama server is not running.".into())
    }
}

#[tauri::command]
pub async fn check_llama_server_status_command(
    python_process_state: State<'_, Arc<PythonProcessState>>,
) -> Result<bool, String> {
    let child_guard = python_process_state.llama_server_child.lock().await;
    Ok(child_guard.is_some())
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
                                    is_mmproj: false, // Since we skip them, this flag usage might change, but for now we won't show them at all.
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

    // Datasets
    let datasets_path = data_dir.join("data").join("datasets");
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
                resources.push(ResourceInfo {
                    name,
                    size: get_dir_size(&path).await.unwrap_or("Unknown".into()),
                    path: format!("data/datasets/{}", entry.file_name().to_string_lossy()),
                    r#type: "dataset".to_string(),
                    quantization: None,
                    is_mmproj: false,
                    is_processed: Some(is_processed),
                    dataset_format: fmt,
                });
            }
        }
    }

    // LoRAs
    let loras_path = data_dir.join("data").join("loras");
    if let Ok(mut entries) = fs::read_dir(&loras_path).await {
        while let Ok(Some(entry)) = entries.next_entry().await {
            if entry.path().is_file() {
                let name = entry.file_name().to_string_lossy().to_string();
                resources.push(ResourceInfo {
                    name: name.clone(),
                    size: get_file_size(&entry.path())
                        .await
                        .unwrap_or("Unknown".into()),
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

    Ok(resources)
}

#[tauri::command]
pub async fn send_chat_message_command(
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
) -> Result<String, String> {
    debug!(
        "Received send_chat_message_command with message: {:?}",
        message
    );
    let mut chat_history = llama_chat_context.chat_history.lock().await;

    // Use serde_json::json! macro for constructing messages
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

    window
        .emit(
            "log",
            format!("Sending message to Llama server: '{}'", message),
        )
        .ok();

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
) -> Result<String, String> {
    debug!("Streaming chat message: {:?}", message);

    let mut chat_history = llama_chat_context.chat_history.lock().await;

    // Use serde_json::json! macro for constructing messages
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

    window
        .emit("log", format!("Streaming chat message..."))
        .ok();

    // DEBUG: Log the request body to see what we are sending for Vision
    let debug_body_str = serde_json::to_string_pretty(&request_body).unwrap_or_default();
    debug!("Llama Request Body: {}", debug_body_str);
    // Be careful not to emit huge base64 strings to frontend logs affecting perf, but for debugging we need to see structure
    if debug_body_str.len() > 2000 {
        window
            .emit(
                "log",
                format!("Request body (truncated): {}", &debug_body_str[..2000]),
            )
            .ok();
    } else {
        window
            .emit("log", format!("Request body: {}", debug_body_str))
            .ok();
    }

    let start_time = std::time::Instant::now();
    let mut first_token_time: Option<std::time::Instant> = None;
    let mut token_count = 0;

    let response = client
        .post(&url)
        .json(&request_body)
        .send()
        .await
        .map_err(|e| format!("Failed to send streaming request: {}", e))?;

    if !response.status().is_success() {
        return Err(format!("Server error: {}", response.status()));
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
                                                window
                                                    .emit(
                                                        "chat-stream-chunk",
                                                        StreamChunk {
                                                            content: content_str.to_string(),
                                                        },
                                                    )
                                                    .ok();
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
            "chat-stream-done",
            StreamMetrics {
                prompt_eval_time_ms,
                eval_time_ms,
                tokens_per_second: tps,
                total_tokens: token_count,
            },
        )
        .ok();

    Ok("Stream complete".to_string())
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HFSearchResult {
    pub id: String,
    pub name: String,
    pub downloads: u64,
    pub likes: u64,
}

#[tauri::command]
pub async fn search_huggingface_command(
    app_handle: AppHandle,
    query: String,
    resource_type: String,
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

    if let Some(f) = files {
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
    if let Some(t) = token {
        if !t.is_empty() {
            cmd.arg("--token").arg(t);
        }
    }

    let mut child = cmd.spawn().map_err(|e| e.to_string())?;

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
        while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
            let trimmed = line.trim();
            if trimmed.starts_with("PROGRESS:") {
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
    tokio::spawn(async move {
        let status = child.wait().await;
        // Emit global event on success
        let success = status.map(|s| s.success()).unwrap_or(false);
        if success {
            win_c.emit("model_downloaded", &model_id_c).ok();
        }

        if let Some(tid) = &task_id_c {
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
        .arg("--output")
        .arg(output_folder.to_string_lossy().to_string())
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
    tokio::spawn(async move {
        let status = child.wait().await;
        let success = status.map(|s| s.success()).unwrap_or(false);
        if success {
            win_c.emit("dataset_downloaded", &dataset_id_c).ok();
        }

        if let Some(tid) = &task_id_c {
            models_state_c.active_downloads.lock().await.remove(tid);
            let s_str = if success { "completed" } else { "error" };
            win_c
                .emit(
                    "download_status",
                    serde_json::json!({ "id": tid, "status": s_str }),
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
    source_path: String,
    destination_path: String,
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

    let win_c2 = window.clone();
    tokio::spawn(async move {
        let mut reader = BufReader::new(stderr);
        let mut line = String::new();
        while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
            if !line.trim().is_empty() {
                win_c2.emit("log", format!("CONVERT: {}", line.trim())).ok();
            }
            line.clear();
        }
    });

    if child.wait().await.map_err(|e| e.to_string())?.success() {
        Ok("Dataset converted successfully".to_string())
    } else {
        Err("Dataset conversion failed".to_string())
    }
}

#[tauri::command]
pub async fn delete_resource_command(
    app_handle: AppHandle,
    resource_type: String,
    resource_path: String,
) -> Result<String, String> {
    let data_dir = get_data_dir(&app_handle);
    let full_path = data_dir.join(&resource_path);

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
    let file_name = source
        .file_name()
        .ok_or("Invalid source")?
        .to_string_lossy()
        .to_string();
    let dest = dest_base.join(&file_name);

    // Check if dest_base exists first? create_dir_all happens in copy_dir_all but not for single file copy
    if !dest_base.exists() {
        fs::create_dir_all(&dest_base)
            .await
            .map_err(|e| e.to_string())?;
    }

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

    let mut cmd = create_hidden_command(&python_exe);
    cmd.arg(&script);
    if let Some(out) = &output_path {
        cmd.arg("--outfile").arg(out);
    }
    cmd.arg("--outtype").arg(&quantization_type);
    cmd.arg(&source_path);
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
            win_c.emit("log", format!("CONVERT: {}", line.trim())).ok();
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

    let mut cmd = create_hidden_command(&python_exe);
    cmd.arg(&script)
        .arg("--base")
        .arg(&base_path)
        .arg("--lora")
        .arg(&lora_path);
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
            win_c
                .emit("log", format!("CONVERT_LORA: {}", line.trim()))
                .ok();
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
pub async fn start_training_command(
    window: Window,
    app_handle: AppHandle,
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
    let (python_exe, work_dir) = get_python_command(&app_handle)?;
    let data_dir = get_data_dir(&app_handle);
    let dataset_abs = if PathBuf::from(&dataset_path).is_absolute() {
        dataset_path
    } else {
        data_dir
            .join("data/datasets")
            .join(&dataset_path)
            .to_string_lossy()
            .to_string()
    };

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
    // std::fs::create_dir_all(&output_dir_abs).unwrap_or_default();
    // Actually let script handle it, or create parent.

    cmd.arg(&script)
        .arg("--model")
        .arg(&model_path)
        .arg("--dataset")
        .arg(&dataset_abs)
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
        .arg(&output_dir_abs)
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
    let loras_dir = data_dir.join("data/loras");
    let mut res = Vec::new();
    if let Ok(mut entries) = fs::read_dir(loras_dir).await {
        while let Ok(Some(entry)) = entries.next_entry().await {
            res.push(entry.file_name().to_string_lossy().to_string());
        }
    }
    Ok(res)
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

    // 1. PyTorch (CUDA 12.4) - PINNED VERSIONS for Unsloth compatibility
    // torch 2.6.0 + torchvision 0.21.0 are required for Unsloth-zoo's int1 dtype
    run_pip(
        vec![
            "--upgrade",
            "--force-reinstall",
            "torch==2.6.0+cu124",
            "torchvision==0.21.0+cu124",
            "torchaudio==2.6.0+cu124",
            "--index-url",
            "https://download.pytorch.org/whl/cu124",
        ],
        "PyTorch (CUDA 12.4)",
        10,
    )
    .await?;

    // 2. Core Dependencies + Triton (for Windows)
    // triton-windows 3.2.x is required for PyTorch 2.6 (3.3+ has breaking changes)
    run_pip(
        vec![
            "--upgrade",
            "triton-windows<3.3",
            "trl",
            "peft",
            "accelerate",
            "bitsandbytes",
            "tensorboard",
            "huggingface_hub",
            "requests",
            "gguf @ git+https://github.com/ggerganov/llama.cpp.git#subdirectory=gguf-py",
        ],
        "Core Dependencies",
        40,
    )
    .await?;

    // 3. Unsloth (Windows version)
    run_pip(
        vec!["unsloth[windows] @ git+https://github.com/unslothai/unsloth.git"],
        "Unsloth",
        70,
    )
    .await?;

    // 4. Reinstall PyTorch CUDA (Unsloth may have pulled in CPU version)
    run_pip(
        vec![
            "torch==2.6.0+cu124",
            "torchvision==0.21.0+cu124",
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
