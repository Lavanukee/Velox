export enum AppView {
  Dashboard,
  Utilities,
  DataCollection,
  FineTuning,
  Inference,
  Logs,
  Settings
}

export interface ModelConfig {
  id: string;
  name: string;
  baseModelId: string;
  downloadOutputFolder: string;
  llamaModelPath: string;
  mmprojPath: string;
  loraPath: string;
  serverHost: string;
  serverPort: number;
  nGpuLayers: number;
  ctxSize: number;
  batchSizeInference: number;
  ubatchSize: number;
  temperature: number;
  noMmap: boolean;
  flashAttn: boolean;
}

export interface DownloadTask {
  id: string;
  type: 'model' | 'dataset' | 'gguf' | 'lora' | 'binary' | 'other';
  name: string;
  progress: number; // 0-100
  status: 'pending' | 'downloading' | 'completed' | 'error' | 'cancelled';
  onCancel?: () => void; // Optional callback to cancel the download
  downloaded_bytes?: number;
  total_bytes?: number;
  speed_bps?: number;
  eta_seconds?: number;
}

export interface Resource {
  name: string;
  size: string;
  path: string;
  type: 'model' | 'gguf' | 'lora' | 'dataset';
  quantization?: string;
  is_mmproj?: boolean;
  has_vision?: boolean;
  is_processed?: boolean;
  dataset_format?: string;
  format_error?: string; // Error message if format detection failed
  count?: number; // Number of examples
  modalities?: string[]; // E.g. ["Vision", "Text"]
  base_model?: string; // Optional path to base model for LoRAs
}

export interface Notification {
  id: string;
  message: string;
  type: 'success' | 'error' | 'info';
}

export interface GpuInfo {
  name: string;
  vram_total: number;
}

export interface HardwareInfo {
  cpu: string;
  ram_total: number;
  gpus: GpuInfo[];
}
