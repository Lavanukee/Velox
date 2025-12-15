use std::ffi::OsStr;
#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;
use std::path::PathBuf;
use std::process::Command as StdCommand;
use tauri::{AppHandle, Manager};
use tokio::fs;
use tokio::process::Command as TokioCommand;

// --- Command Helpers ---

pub fn create_hidden_command<S: AsRef<OsStr>>(program: S) -> TokioCommand {
    let mut cmd = TokioCommand::new(program);
    #[cfg(target_os = "windows")]
    {
        cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }
    cmd
}

pub fn create_hidden_std_command<S: AsRef<OsStr>>(program: S) -> StdCommand {
    let mut cmd = StdCommand::new(program);
    #[cfg(target_os = "windows")]
    {
        cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }
    cmd
}

// --- Path Helpers ---

pub fn get_project_root() -> PathBuf {
    let src_tauri_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    src_tauri_dir
        .parent()
        .expect("Failed to get project root directory")
        .to_path_buf()
}

#[allow(unused_variables)]
pub fn get_data_dir(app_handle: &AppHandle) -> PathBuf {
    if cfg!(debug_assertions) {
        // In dev, use the project root (where data/ folder is)
        get_project_root()
    } else {
        // In production, use the system AppData folder
        // e.g., C:\Users\User\AppData\Roaming\com.velox.dev or similar
        app_handle
            .path()
            .app_data_dir()
            .expect("Failed to get app data dir")
    }
}

// --- File System Helpers ---

pub async fn copy_dir_all(src: &PathBuf, dst: &PathBuf) -> Result<(), String> {
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

pub async fn get_file_size(path: &PathBuf) -> Result<String, String> {
    let metadata = fs::metadata(path).await.map_err(|e| e.to_string())?;
    Ok(format_size(metadata.len()))
}

pub async fn get_dir_size(path: &PathBuf) -> Result<String, String> {
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

pub fn format_size(bytes: u64) -> String {
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

// --- Domain Specific Helpers ---

pub async fn check_dir_for_gguf(dir: &PathBuf) -> bool {
    if let Ok(mut entries) = fs::read_dir(dir).await {
        while let Ok(Some(entry)) = entries.next_entry().await {
            if entry.path().extension().map_or(false, |ext| ext == "gguf") {
                return true;
            }
        }
    }
    false
}

pub fn detect_quantization(filename: &str) -> String {
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

pub fn detect_dataset_format(path: &PathBuf) -> Option<String> {
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
