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
}
