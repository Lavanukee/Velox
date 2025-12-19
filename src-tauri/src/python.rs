use crate::constants::{PYTHON_DIR, PYTHON_EXE, SCRIPTS_DIR};
use log::{debug, error};
use std::path::PathBuf;
use tauri::Manager;
use tokio::process::Child;
use tokio::sync::Mutex;

pub struct PythonProcessState {
    pub data_collector_child: Mutex<Option<Child>>,
    pub llama_server_child: Mutex<Option<Child>>,
    pub transformers_child: Mutex<Option<Child>>,
    pub tensorboard_child: Mutex<Option<Child>>,
    pub tensorboard_port: Mutex<Option<u16>>,
}

impl Default for PythonProcessState {
    fn default() -> Self {
        PythonProcessState {
            data_collector_child: Mutex::new(None),
            llama_server_child: Mutex::new(None),
            transformers_child: Mutex::new(None),
            tensorboard_child: Mutex::new(None),
            tensorboard_port: Mutex::new(None),
        }
    }
}

// Struct to hold the chat context (history) for the Llama server
#[derive(Default)]
pub struct LlamaChatContext {
    pub chat_history: Mutex<Vec<serde_json::Value>>,
}

// --- Helper: Resolve Python Path and Working Directory ---
pub fn get_python_command(app_handle: &tauri::AppHandle) -> Result<(String, PathBuf), String> {
    // Use data dir (AppData in prod, Project Root in debug)
    let base_dir = crate::utils::get_data_dir(app_handle);

    // 1. Try to find the bundled/downloaded python
    let python_path = base_dir.join(PYTHON_DIR).join(PYTHON_EXE);

    if python_path.exists() {
        let mut python_exe = python_path
            .canonicalize()
            .unwrap_or(python_path.clone())
            .to_string_lossy()
            .to_string();

        // Windows specific path fix
        if cfg!(windows) && python_exe.starts_with(r"\\?\") {
            python_exe = python_exe[4..].to_string();
        }

        debug!("Using Python executable: {}", python_exe);
        return Ok((python_exe, base_dir));
    }

    // Error: No python found
    error!("Python not found at {:?}", python_path);
    Err(format!(
        "Python not found. Expected at: {}",
        python_path.display()
    ))
}

// --- Helper: Resolve Script Path ---
pub fn get_script_path(app_handle: &tauri::AppHandle, script_name: &str) -> String {
    #[cfg(debug_assertions)]
    {
        let _ = app_handle;
        // Dev: use source files directly
        let script_path = crate::utils::get_project_root()
            .join(SCRIPTS_DIR)
            .join(script_name);
        return script_path.to_string_lossy().to_string();
    }
    #[cfg(not(debug_assertions))]
    {
        // Prod: use bundled resources
        let resource_dir = app_handle
            .path()
            .resource_dir()
            .expect("Failed to get resource dir");
        let script_path = resource_dir.join("scripts").join(script_name);
        return script_path.to_string_lossy().to_string();
    }
}
