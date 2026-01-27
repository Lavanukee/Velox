mod commands;
mod constants;
mod hardware;
mod models;
mod python;
mod utils;

pub use hardware::get_hardware_info_command;

use commands::*;
use models::ModelsState;
use python::{get_python_command, LlamaChatContext, PythonProcessState};
use std::collections::HashMap;
use std::process::Stdio;
use std::sync::Arc;
use tauri::{Emitter, Manager};
use tokio::sync::Mutex;

// Re-implement the TensorBoard startup logic inside run or a local helper
// Since we want to keep lib.rs clean, but we need this logic to run on startup.
// We can use the logic we saw in the previous file.

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    env_logger::init();

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_drag::init())
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
            let app_handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                // Try to start TensorBoard backend quietly on startup
                let python_state = app_handle.state::<Arc<PythonProcessState>>().clone();

                // If already started, skip
                let already = {
                    let g = python_state.tensorboard_port.lock().await;
                    *g
                };
                if already.is_some() {
                    return;
                }

                // Find free port
                let listener = match std::net::TcpListener::bind("127.0.0.1:0") {
                    Ok(l) => l,
                    Err(e) => {
                        app_handle
                            .emit("log", format!("TB Startup Error: Bind failed: {}", e))
                            .ok();
                        return;
                    }
                };
                let port = match listener.local_addr() {
                    Ok(a) => a.port(),
                    Err(_) => return,
                };
                drop(listener);

                let (python_exe, work_dir) = match get_python_command(&app_handle) {
                    Ok(r) => r,
                    Err(e) => {
                        app_handle
                            .emit("log", format!("TB Startup Error: Python not found: {}", e))
                            .ok();
                        return;
                    }
                };

                let logdir = "data/outputs"; // Default logdir base
                let _ = std::fs::create_dir_all(work_dir.join(logdir));

                // On Windows, use detached process logic similar to original lib.rs if possible,
                // or just spawn simple hidden command for now to ensure it works.
                // We'll use our utils helper if we can, but we are inside lib.rs.
                // We can access crate::utils::create_hidden_command.

                let mut cmd = crate::utils::create_hidden_command(&python_exe);
                #[cfg(target_os = "windows")]
                {
                    // Use specific bat file approach if we want to be exactly like before,
                    // or just trust create_hidden_command (creation_flags) works enough for background services.
                    // The original code used a .bat file to redirect output.
                    // For simplicity in refactor, we'll pipe logic directly.
                }

                cmd.arg("-m")
                    .arg("tensorboard.main")
                    .arg("--logdir")
                    .arg(logdir)
                    .arg("--port")
                    .arg(port.to_string())
                    .arg("--host")
                    .arg("127.0.0.1") // strictly local
                    .arg("--reload_interval")
                    .arg("15") // Refresh every 15 seconds for live training updates
                    .arg("--reload_multifile")
                    .arg("true") // Pick up new event files from SFTTrainer
                    .current_dir(&work_dir)
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped());

                match cmd.spawn() {
                    Ok(mut child) => {
                        let stdout = child.stdout.take();
                        let stderr = child.stderr.take();

                        // Spawn threads to pipe output to logs
                        if let Some(stdout) = stdout {
                            let app_handle_c = app_handle.clone();
                            tauri::async_runtime::spawn(async move {
                                use tokio::io::{AsyncBufReadExt, BufReader};
                                let reader = BufReader::new(stdout);
                                let mut lines = reader.lines();
                                while let Ok(Some(line)) = lines.next_line().await {
                                    app_handle_c.emit("log", format!("TB: {}", line)).ok();
                                }
                            });
                        }
                        if let Some(stderr) = stderr {
                            let app_handle_c = app_handle.clone();
                            tauri::async_runtime::spawn(async move {
                                use tokio::io::{AsyncBufReadExt, BufReader};
                                let reader = BufReader::new(stderr);
                                let mut lines = reader.lines();
                                while let Ok(Some(line)) = lines.next_line().await {
                                    app_handle_c.emit("log", format!("TB ERR: {}", line)).ok();
                                }
                            });
                        }

                        let mut guard = python_state.tensorboard_child.lock().await;
                        *guard = Some(child);
                        let mut pguard = python_state.tensorboard_port.lock().await;
                        *pguard = Some(port);

                        app_handle
                            .emit(
                                "log",
                                format!("TensorBoard background service started on port {}", port),
                            )
                            .ok();
                    }
                    Err(e) => {
                        app_handle
                            .emit("log", format!("Failed to start TensorBoard service: {}", e))
                            .ok();
                    }
                }
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            check_python_installed_command,
            check_python_minimal_command,
            check_llama_binary_command,
            check_llama_binary_exists_command,
            download_llama_binary_command,
            check_server_health_command,
            execute_tool_command,
            save_model_config_command,
            load_model_configs_command,
            delete_model_config_command,
            list_model_folders_command,
            list_dataset_folders_command,
            list_finetuning_models_command,
            get_model_config_command,
            list_training_projects_command,
            list_loras_by_project_command,
            save_project_config_command,
            load_project_config_command,
            start_llama_server_command,
            stop_llama_server_command,
            check_llama_server_status_command,
            start_transformers_server_command,
            stop_transformers_server_command,
            list_all_resources_command,
            send_chat_message_command,
            send_chat_message_streaming_command,
            get_chat_response_command,
            clear_chat_history_command,
            export_resources_command,
            search_huggingface_command,
            list_hf_repo_files_command,
            list_directory_command,
            download_hf_model_command,
            download_hf_dataset_command,
            convert_dataset_command,
            delete_resource_command,
            delete_project_command,
            rename_resource_command,
            import_resource_command,
            load_presets_command,
            save_preset_command,
            convert_hf_to_gguf_command,
            convert_lora_to_gguf_command,
            convert_unsloth_gguf_command,
            cancel_download_command,
            start_training_command,
            stop_training_command,
            check_python_standalone_command,
            download_python_standalone_command,
            debug_python_path_command,
            take_screenshot_path_command,
            save_annotation_crop_command,
            list_gguf_models_command,
            list_lora_adapters_command,
            setup_python_env_command,
            get_hf_token_command,
            save_hf_token_command,
            start_data_collector_command,
            get_tensorboard_url_command,
            get_hardware_info_command,
            execute_tool_command,
            save_custom_tool_command,
            fix_dataset_command,
            bulk_edit_dataset_command,
            process_vlm_dataset_command,
            resolve_path_command,
            load_dataset_preview_command,
            save_dataset_command,
            apply_dataset_edits_command,
            generate_dataset_command,
            create_dataset_command,
            analyze_drop_command
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app_handle, event| {
            if let tauri::RunEvent::ExitRequested { .. } = event {
                // Cleanup all child processes on exit
                let python_state = app_handle.state::<Arc<PythonProcessState>>().clone();

                // Block current thread to await cleanup
                tauri::async_runtime::block_on(async {
                    // Kill primary llama server
                    if let Some(mut child) = python_state.llama_server_child.lock().await.take() {
                        let _ = child.kill().await;
                    }
                    // Kill secondary llama server
                    if let Some(mut child) = python_state.llama_secondary_child.lock().await.take()
                    {
                        let _ = child.kill().await;
                    }
                    // Kill transformers server
                    if let Some(mut child) = python_state.transformers_child.lock().await.take() {
                        let _ = child.kill().await;
                    }
                    // Kill data collector
                    if let Some(mut child) = python_state.data_collector_child.lock().await.take() {
                        let _ = child.kill().await;
                    }
                    // Kill TensorBoard
                    if let Some(mut child) = python_state.tensorboard_child.lock().await.take() {
                        let _ = child.kill().await;
                    }
                });

                log::info!("All child processes cleaned up on exit.");
            }
        });
}
