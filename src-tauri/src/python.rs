use crate::constants::{PYTHON_DIR, PYTHON_EXE, SCRIPTS_DIR};
use crate::utils::get_project_root;
use log::{debug, error};
use std::collections::HashMap;
use std::path::PathBuf;
use tokio::process::Child;
use tokio::sync::Mutex;

pub struct PythonProcessState {
    pub data_collector_child: Mutex<Option<Child>>,
    pub llama_server_child: Mutex<Option<Child>>,
    pub tensorboard_child: Mutex<Option<Child>>,
    pub tensorboard_port: Mutex<Option<u16>>,
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

// --- Helper: Resolve Python Path and Working Directory ---
pub fn get_python_command() -> Result<(String, PathBuf), String> {
    let project_root_dir = get_project_root();

    // 1. Try to find the bundled portable python
    let portable_python_path = project_root_dir.join(PYTHON_DIR).join(PYTHON_EXE);

    if portable_python_path.exists() {
        let mut python_exe = portable_python_path
            .canonicalize()
            .unwrap_or(portable_python_path.clone())
            .to_string_lossy()
            .to_string();

        // Windows specific path fix
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
pub fn get_script_path(script_name: &str) -> String {
    let script_path_in_project_root = PathBuf::from(SCRIPTS_DIR).join(script_name);
    script_path_in_project_root.to_string_lossy().to_string()
}
