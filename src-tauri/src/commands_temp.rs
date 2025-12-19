#[tauri::command]
pub async fn list_directory_command(path: String) -> Result<Vec<String>, String> {
    let mut files = Vec::new();
    let entries = fs::read_dir(path).await.map_err(|e| e.to_string())?;

    let mut entries_stream = tokio_stream::wrappers::ReadDirStream::new(entries);
    while let Some(entry) = entries_stream.next().await {
        let entry = entry.map_err(|e| e.to_string())?;
        if let Ok(name) = entry.file_name().into_string() {
            files.push(name);
        }
    }
    Ok(files)
}
