use crate::constants::MODELS_CONFIG_FILE;
use log::{debug, error};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use tokio::fs;
use tokio::sync::Mutex;

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

pub struct ModelsState {
    pub configs: Mutex<HashMap<String, ModelConfig>>,
    pub active_downloads: Mutex<HashMap<String, u32>>,
}

impl ModelsState {
    pub async fn load_from_disk() -> Result<HashMap<String, ModelConfig>, String> {
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

    pub async fn save_to_disk(configs: &HashMap<String, ModelConfig>) -> Result<(), String> {
        let config_path = PathBuf::from(MODELS_CONFIG_FILE);
        let serialized = serde_json::to_string_pretty(configs).map_err(|e| e.to_string())?;
        fs::write(&config_path, serialized)
            .await
            .map_err(|e| e.to_string())?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectLoraInfo {
    pub project_name: String,
    pub checkpoints: Vec<CheckpointInfo>,
    pub base_model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointInfo {
    pub name: String, // e.g., "checkpoint-100", "final_model"
    pub path: String, // relative path from project root
    pub is_final: bool,
    pub step_number: Option<i32>,
    pub base_model_name: Option<String>,
    pub gguf_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PersistentDownloadTask {
    pub id: String,
    pub name: String,
    pub type_: String, // "model" | "dataset" | etc.
    pub repo_id: String,
    pub files: Option<Vec<String>>,
}
