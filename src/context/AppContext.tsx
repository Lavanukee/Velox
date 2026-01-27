import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { invoke } from "@tauri-apps/api/core";
import { listen } from '@tauri-apps/api/event';
import type { CanvasArtifact } from '../components/CanvasPanel';
import { Resource, Notification, AppView, HardwareInfo } from '../types';

type UserMode = 'user' | 'power';
type ColorMode = 'light' | 'dark';

export interface UserFeatures {
  showUserModeToggle: boolean;
  showAdvancedFinetuning: boolean;
  showAdvancedInference: boolean;
  showMetrics: boolean;
  enableWebSearch: boolean;
  enableCodeExec: boolean;
  enableCanvas: boolean;
  autoFitMemory: boolean;
  autoProcessDatasets: boolean;
}

const DEFAULT_USER_FEATURES: UserFeatures = {
  showUserModeToggle: true,
  showAdvancedFinetuning: false,
  showAdvancedInference: false,
  showMetrics: false,
  enableWebSearch: false,
  enableCodeExec: false,
  enableCanvas: false,
  autoFitMemory: true,
  autoProcessDatasets: true
};

const DEFAULT_POWER_FEATURES: UserFeatures = {
  showUserModeToggle: true,
  showAdvancedFinetuning: true,
  showAdvancedInference: true,
  showMetrics: true,
  enableWebSearch: true,
  enableCodeExec: true,
  enableCanvas: true,
  autoFitMemory: false,
  autoProcessDatasets: true
};

export interface ChatMessage {
  id?: string;
  text: string;
  sender: 'user' | 'bot' | 'system';
  timestamp: number;
  isStreaming?: boolean;
  image?: string | null;
}

interface AppContextType {
  // User Mode & Features
  userMode: UserMode;
  toggleUserMode: () => void;
  setUserMode: (mode: UserMode) => void;
  userFeatures: UserFeatures;
  setUserFeatures: (features: Partial<UserFeatures>) => void;

  // Inference State
  chatMessages: ChatMessage[];
  setChatMessages: React.Dispatch<React.SetStateAction<ChatMessage[]>>;
  inputMessage: string;
  setInputMessage: (msg: string) => void;
  selectedBaseModel: string;
  setSelectedBaseModel: (model: string) => void;
  selectedMmproj: string;
  setSelectedMmproj: (model: string) => void;
  selectedLoraAdapter: string;
  setSelectedLoraAdapter: (adapter: string) => void;
  temperature: number;
  setTemperature: (temp: number) => void;
  contextSize: number;
  setContextSize: (size: number) => void;
  topP: number;
  setTopP: (val: number) => void;
  topK: number;
  setTopK: (val: number) => void;
  systemPrompt: string;
  setSystemPrompt: (prompt: string) => void;
  isServerRunning: boolean;
  setIsServerRunning: (running: boolean) => void;

  // Advanced Inference Options (Power User)
  infFlashAttn: boolean;
  setInfFlashAttn: (val: boolean) => void;
  infNoMmap: boolean;
  setInfNoMmap: (val: boolean) => void;
  infGpuLayers: number;
  setInfGpuLayers: (val: number) => void;
  infBatchSize: number;
  setInfBatchSize: (val: number) => void;
  infUbatchSize: number;
  setInfUbatchSize: (val: number) => void;
  infThreads: number;
  setInfThreads: (val: number) => void;
  infServerStatus: 'idle' | 'loading' | 'ready' | 'error';
  setInfServerStatus: (status: 'idle' | 'loading' | 'ready' | 'error') => void;

  // Chat Feature Toggles
  infShowMetrics: boolean;
  setInfShowMetrics: (val: boolean) => void;
  infEnableWebSearch: boolean;
  setInfEnableWebSearch: (val: boolean) => void;
  infEnableCodeExec: boolean;
  setInfEnableCodeExec: (val: boolean) => void;
  infEnableCanvas: boolean;
  setInfEnableCanvas: (val: boolean) => void;
  infCanvasVisible: boolean;
  setInfCanvasVisible: (val: boolean) => void;
  infCanvasArtifacts: CanvasArtifact[];
  setInfCanvasArtifacts: React.Dispatch<React.SetStateAction<CanvasArtifact[]>>;

  // Auto-fit memory management (llama.cpp --fit)
  infAutoFit: boolean;
  setInfAutoFit: (val: boolean) => void;

  // Global App Settings
  autoUpdate: boolean;
  setAutoUpdate: (val: boolean) => void;
  showInfoTooltips: boolean;
  setShowInfoTooltips: (val: boolean) => void;

  // Inference Engine Selection
  infInferenceEngine: 'llamacpp' | 'transformers';
  setInfInferenceEngine: (engine: 'llamacpp' | 'transformers') => void;

  selectedProject: string;
  setSelectedProject: (project: string) => void;
  selectedCheckpoint: string;
  setSelectedCheckpoint: (checkpoint: string) => void;

  // Resource Dashboard State
  rdHfQuery: string;
  setRdHfQuery: (query: string) => void;
  rdHfAuthor: string;
  setRdHfAuthor: (author: string) => void;
  rdHfModalities: string;
  setRdHfModalities: (modalities: string) => void;
  rdHfSizeRange: string;
  setRdHfSizeRange: (range: string) => void;
  rdHfType: 'model' | 'dataset';
  setRdHfType: (type: 'model' | 'dataset') => void;
  rdShowFindNew: boolean;
  setRdShowFindNew: (show: boolean) => void;
  currentView: AppView;
  setCurrentView: (view: AppView) => void;
  rdSelectedPaths: Set<string>;
  setRdSelectedPaths: React.Dispatch<React.SetStateAction<Set<string>>>;
  autoProcessDatasets: boolean;
  setAutoProcessDatasets: (val: boolean) => void;

  // Fine-Tuning State
  ftProjectName: string;
  setFtProjectName: (name: string) => void;
  ftSelectedModel: string;
  setFtSelectedModel: (model: string) => void;
  ftSelectedDataset: string;
  setFtSelectedDataset: (dataset: string) => void;
  ftSelectedDatasets: string[];
  setFtSelectedDatasets: React.Dispatch<React.SetStateAction<string[]>>;
  ftDatasetMixing: Record<string, number>; // dataset path -> percentage (0-100)
  setFtDatasetMixing: React.Dispatch<React.SetStateAction<Record<string, number>>>;
  ftLocalDatasetPath: string;
  setFtLocalDatasetPath: (path: string) => void;
  ftHfModelId: string;
  setFtHfModelId: (id: string) => void;
  ftHfDatasetId: string;
  setFtHfDatasetId: (id: string) => void;
  ftNumEpochs: number | string;
  setFtNumEpochs: (n: number | string) => void;
  ftBatchSize: number | string;
  setFtBatchSize: (n: number | string) => void;
  ftLearningRate: number | string;
  setFtLearningRate: (n: number | string) => void;
  ftLoraR: number | string;
  setFtLoraR: (n: number | string) => void;
  ftLoraAlpha: number | string;
  setFtLoraAlpha: (n: number | string) => void;
  ftMaxSeqLength: number | string;
  setFtMaxSeqLength: (n: number | string) => void;
  ftActiveTab: 'config' | 'training';
  setFtActiveTab: (tab: 'config' | 'training') => void;
  ftIsTraining: boolean;
  setFtIsTraining: (isTraining: boolean) => void;
  ftTrainingStatus: 'idle' | 'initializing' | 'training';
  setFtTrainingStatus: (status: 'idle' | 'initializing' | 'training') => void;
  ftTensorboardUrl: string;
  setFtTensorboardUrl: (url: string) => void;
  ftInitMessage: string;
  setFtInitMessage: (msg: string) => void;

  // Setup State (persisted across tab navigation)
  ftIsSettingUp: boolean;
  setFtIsSettingUp: (isSettingUp: boolean) => void;
  ftSetupComplete: boolean;
  setFtSetupComplete: (complete: boolean) => void;
  ftSetupProgress: { current: number; total: number; message: string };
  setFtSetupProgress: (progress: { current: number; total: number; message: string }) => void;

  // Global Resources
  resources: Resource[];
  setResources: React.Dispatch<React.SetStateAction<Resource[]>>;
  loadResources: () => Promise<void>;
  isConvertingMap: Record<string, boolean>;
  setConvertingMap: React.Dispatch<React.SetStateAction<Record<string, boolean>>>;

  // Global Conversion State
  utIsConverting: boolean;
  setUtIsConverting: (val: boolean) => void;
  utConversionProgress: number;
  setUtConversionProgress: (val: number) => void;
  utConversionLabel: string;
  setUtConversionLabel: (val: string) => void;
  utSourcePath: string;
  setUtSourcePath: (val: string) => void;
  utOutputPath: string;
  setUtOutputPath: (val: string) => void;
  utBasePath: string;
  setUtBasePath: (val: string) => void;
  utQuantizationType: string;
  setUtQuantizationType: (val: string) => void;
  utConversionType: 'hf_to_gguf' | 'lora_to_gguf';
  setUtConversionType: (val: 'hf_to_gguf' | 'lora_to_gguf') => void;
  utConversionEngine: 'standard' | 'unsloth';
  setUtConversionEngine: (val: 'standard' | 'unsloth') => void;

  // Global Setup State
  showPythonSetup: boolean;
  setShowPythonSetup: (show: boolean) => void;
  isInitializing: boolean;
  setIsInitializing: (init: boolean) => void;
  setupProgressPercent: number;
  setSetupProgressPercent: (val: number) => void;
  setupMessage: string;
  setSetupMessage: (msg: string) => void;
  setupLoadedBytes: number;
  setSetupLoadedBytes: (val: number) => void;
  setupTotalBytes: number;
  setSetupTotalBytes: (val: number) => void;
  runGlobalSetup: (forceDownload?: boolean) => Promise<void>;
  isEngineUpdating: boolean;
  setIsEngineUpdating: (val: boolean) => void;
  theme: 'cyber' | 'forge';
  colorMode: ColorMode;
  setColorMode: (mode: ColorMode) => void;
  toggleColorMode: () => void;

  // Benchmarking
  isBenchmarking: boolean;
  setIsBenchmarking: (val: boolean) => void;
  benchmarkMessages: ChatMessage[];
  setBenchmarkMessages: React.Dispatch<React.SetStateAction<ChatMessage[]>>;
  isBlindTest: boolean;
  setIsBlindTest: (val: boolean) => void;
  selectedBenchmarkModel: string;
  setSelectedBenchmarkModel: (m: string) => void;

  // Evaluation Mode (Arena/Compare)
  evaluationMode: 'off' | 'compare' | 'arena';
  setEvaluationMode: (mode: 'off' | 'compare' | 'arena') => void;
  arenaSelectedModels: string[];
  setArenaSelectedModels: React.Dispatch<React.SetStateAction<string[]>>;
  arenaScores: Record<string, { wins: number; ties: number; total: number }>;
  setArenaScores: React.Dispatch<React.SetStateAction<Record<string, { wins: number; ties: number; total: number }>>>;
  arenaCurrentPair: [string, string] | null; // [modelA, modelB] currently being compared
  setArenaCurrentPair: (pair: [string, string] | null) => void;

  // Real-time Status (Sidebar)
  loadedModels: { name: string; type: string }[];
  setLoadedModels: React.Dispatch<React.SetStateAction<{ name: string; type: string }[]>>;

  // persistent streaming state
  streamMetrics: {
    tokens_per_second: number;
    prompt_eval_time_ms: number;
    eval_time_ms: number;
    total_tokens: number;
  } | null;
  benchmarkStreamMetrics: {
    tokens_per_second: number;
    prompt_eval_time_ms: number;
    eval_time_ms: number;
    total_tokens: number;
  } | null;
  isPromptProcessing: boolean;
  setIsPromptProcessing: (val: boolean) => void;
  isSending: boolean;
  setIsSending: (val: boolean) => void;

  // App Level
  appScale: number;
  setAppScale: (scale: number) => void;

  // Global Notifications
  notifications: Notification[];
  addNotification: (message: string, type?: 'success' | 'error' | 'info') => void;

  // System
  hardwareInfo: HardwareInfo | null;
}

const AppContext = createContext<AppContextType | undefined>(undefined);

export const AppProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  // User Mode & Features
  const [userMode, setUserModeState] = useState<UserMode>('user');
  const [userFeatures, setUserFeaturesState] = useState<UserFeatures>(DEFAULT_USER_FEATURES);

  const setUserFeatures = (newFeatures: Partial<UserFeatures>) => {
    setUserFeaturesState(prev => {
      const updated = { ...prev, ...newFeatures };
      localStorage.setItem('velox_user_features', JSON.stringify(updated));
      return updated;
    });
  };

  // Inference State
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [selectedBaseModel, setSelectedBaseModel] = useState('');
  const [selectedProject, setSelectedProject] = useState('');
  const [selectedCheckpoint, setSelectedCheckpoint] = useState('');
  const [selectedMmproj, setSelectedMmproj] = useState('');
  const [selectedLoraAdapter, setSelectedLoraAdapter] = useState('');
  const [temperature, setTemperature] = useState(0.7);
  const [contextSize, setContextSize] = useState(2048);
  const [topP, setTopP] = useState(0.9);
  const [topK, setTopK] = useState(40);
  const [systemPrompt, setSystemPrompt] = useState('You are a helpful AI assistant.');
  const [isServerRunning, setIsServerRunning] = useState(false);

  // Advanced Inference Options (Power User)
  const [infFlashAttn, setInfFlashAttn] = useState(true);
  const [infNoMmap, setInfNoMmap] = useState(true); // true by default for better perf
  const [infGpuLayers, setInfGpuLayers] = useState(99);
  const [infBatchSize, setInfBatchSize] = useState(512);
  const [infUbatchSize, setInfUbatchSize] = useState(512);
  const [infThreads, setInfThreads] = useState(0); // 0 = auto
  const [infServerStatus, setInfServerStatus] = useState<'idle' | 'loading' | 'ready' | 'error'>('idle');

  // Chat Feature Toggles
  const [infShowMetrics, setInfShowMetrics] = useState(true);
  const [infEnableWebSearch, setInfEnableWebSearch] = useState(false);
  const [infEnableCodeExec, setInfEnableCodeExec] = useState(false);
  const [infEnableCanvas, setInfEnableCanvas] = useState(false);
  const [infCanvasVisible, setInfCanvasVisible] = useState(false);
  const [infCanvasArtifacts, setInfCanvasArtifacts] = useState<CanvasArtifact[]>([]);

  // Auto-fit memory management (llama.cpp --fit)
  const [infAutoFit, setInfAutoFit] = useState(true); // Default on for regular users

  // Global App Settings
  const [autoUpdate, setAutoUpdateState] = useState(true);
  const [showInfoTooltips, setShowInfoTooltipsState] = useState(true);

  // Inference Engine Selection
  const [infInferenceEngine, setInfInferenceEngine] = useState<'llamacpp' | 'transformers'>('llamacpp');

  // Resource Dashboard State
  const [rdHfQuery, setRdHfQuery] = useState('');
  const [rdHfAuthor, setRdHfAuthor] = useState('');
  const [rdHfModalities, setRdHfModalities] = useState('');
  const [rdHfSizeRange, setRdHfSizeRange] = useState('');
  const [rdHfType, setRdHfType] = useState<'model' | 'dataset'>('model');
  const [rdShowFindNew, setRdShowFindNew] = useState(false);
  const [currentView, setCurrentView] = useState<AppView>(AppView.Dashboard);
  const [rdSelectedPaths, setRdSelectedPaths] = useState<Set<string>>(new Set());
  const [autoProcessDatasets, setAutoProcessDatasetsState] = useState(true);

  // Fine-Tuning State
  const [ftProjectName, setFtProjectName] = useState('');
  const [ftSelectedModel, setFtSelectedModel] = useState('');
  const [ftSelectedDataset, setFtSelectedDataset] = useState('');
  const [ftSelectedDatasets, setFtSelectedDatasets] = useState<string[]>([]);
  const [ftDatasetMixing, setFtDatasetMixing] = useState<Record<string, number>>({});

  // System
  const [hardwareInfo, setHardwareInfo] = useState<HardwareInfo | null>(null);

  useEffect(() => {
    invoke<HardwareInfo>('get_hardware_info_command')
      .then(info => {
        console.log("Hardware Detected:", info);
        setHardwareInfo(info);
      })
      .catch(err => console.error("Failed to detect hardware:", err));
  }, []);

  // -- Derived --
  const [ftLocalDatasetPath, setFtLocalDatasetPath] = useState('');
  const [ftHfModelId, setFtHfModelId] = useState('');
  const [ftHfDatasetId, setFtHfDatasetId] = useState('');
  const [ftNumEpochs, setFtNumEpochs] = useState<number | string>(3);
  const [ftBatchSize, setFtBatchSize] = useState<number | string>(4);
  const [ftLearningRate, setFtLearningRate] = useState<number | string>(2e-4);
  const [ftLoraR, setFtLoraR] = useState<number | string>(16);
  const [ftLoraAlpha, setFtLoraAlpha] = useState<number | string>(32);
  const [ftMaxSeqLength, setFtMaxSeqLength] = useState<number | string>(2048);
  const [ftActiveTab, setFtActiveTab] = useState<'config' | 'training'>('config');
  const [ftIsTraining, setFtIsTraining] = useState(false);
  const [ftTrainingStatus, setFtTrainingStatus] = useState<'idle' | 'initializing' | 'training'>('idle');
  const [ftTensorboardUrl, setFtTensorboardUrl] = useState('');
  const [ftInitMessage, setFtInitMessage] = useState('');

  // Setup State (persisted across tab navigation)
  const [ftIsSettingUp, setFtIsSettingUp] = useState(false);
  const [ftSetupComplete, setFtSetupComplete] = useState(() => {
    // Check localStorage on init
    return localStorage.getItem('pythonEnvSetup') === 'complete';
  });
  const [ftSetupProgress, setFtSetupProgress] = useState({ current: 0, total: 7, message: '' });

  // Global Resources
  const [resources, setResources] = useState<Resource[]>([]);
  const [isConvertingMap, setConvertingMap] = useState<Record<string, boolean>>({});

  // Global Conversion State
  const [utIsConverting, setUtIsConverting] = useState(false);
  const [utConversionProgress, setUtConversionProgress] = useState(0);
  const [utConversionLabel, setUtConversionLabel] = useState('');
  const [utSourcePath, setUtSourcePath] = useState('');
  const [utOutputPath, setUtOutputPath] = useState('');
  const [utBasePath, setUtBasePath] = useState('');
  const [utQuantizationType, setUtQuantizationType] = useState('q8_0');
  const [utConversionType, setUtConversionType] = useState<'hf_to_gguf' | 'lora_to_gguf'>('hf_to_gguf');
  const [utConversionEngine, setUtConversionEngine] = useState<'standard' | 'unsloth'>('unsloth');

  const loadResources = useCallback(async () => {
    try {
      const raw: Resource[] = await invoke('list_all_resources_command');
      setResources(raw);
    } catch (e) {
      console.error(`Error loading resources: ${e}`);
    }
  }, []);

  // Global Setup State
  const [showPythonSetup, setShowPythonSetup] = useState(false);
  const [isInitializing, setIsInitializing] = useState(false); // Hide splash by default, show only if setup needed
  const [setupProgressPercent, setSetupProgressPercent] = useState(0);
  const [setupMessage, setSetupMessage] = useState("Initializing environment...");
  const [setupLoadedBytes, setSetupLoadedBytes] = useState(0);
  const [setupTotalBytes, setSetupTotalBytes] = useState(0);
  const [isEngineUpdating, setIsEngineUpdating] = useState(false);

  // Global Notifications
  const [notifications, setNotifications] = useState<Notification[]>([]);

  const addNotification = useCallback((message: string, type: 'success' | 'error' | 'info' = 'info') => {
    const id = `notif_${Date.now()}`;
    setNotifications(prev => [...prev, { id, message, type }]);
    setTimeout(() => {
      setNotifications(prev => prev.filter(n => n.id !== id));
    }, 5000);
  }, []);

  // Benchmarking State
  const [isBenchmarking, setIsBenchmarking] = useState(false);
  const [benchmarkMessages, setBenchmarkMessages] = useState<ChatMessage[]>([]);
  const [isBlindTest, setIsBlindTest] = useState(false);
  const [selectedBenchmarkModel, setSelectedBenchmarkModel] = useState('');

  // Evaluation Mode (Arena/Compare)
  const [evaluationMode, setEvaluationMode] = useState<'off' | 'compare' | 'arena'>('off');
  const [arenaSelectedModels, setArenaSelectedModels] = useState<string[]>(() => {
    const saved = localStorage.getItem('velox_arena_selected_models');
    return saved ? JSON.parse(saved) : [];
  });
  const [arenaScores, setArenaScores] = useState<Record<string, { wins: number; ties: number; total: number }>>(() => {
    const saved = localStorage.getItem('velox_arena_scores');
    return saved ? JSON.parse(saved) : {};
  });
  const [arenaCurrentPair, setArenaCurrentPair] = useState<[string, string] | null>(null);

  useEffect(() => {
    localStorage.setItem('velox_arena_scores', JSON.stringify(arenaScores));
  }, [arenaScores]);

  useEffect(() => {
    localStorage.setItem('velox_arena_selected_models', JSON.stringify(arenaSelectedModels));
  }, [arenaSelectedModels]);

  // Real-time Status (Sidebar)
  const [loadedModels, setLoadedModels] = useState<{ name: string; type: string }[]>([]);

  // Persistent Streaming State
  const [streamMetricsState, setStreamMetrics] = useState<{
    tokens_per_second: number;
    prompt_eval_time_ms: number;
    eval_time_ms: number;
    total_tokens: number;
  } | null>(null);
  const [benchmarkStreamMetrics, setBenchmarkStreamMetrics] = useState<{
    tokens_per_second: number;
    prompt_eval_time_ms: number;
    eval_time_ms: number;
    total_tokens: number;
  } | null>(null);
  const [isPromptProcessingState, setIsPromptProcessing] = useState(false);
  const [isSendingState, setIsSending] = useState(false);

  // App Level
  const [appScale, setAppScale] = useState(0.85); // 15% shrink by default

  useEffect(() => {
    document.documentElement.style.setProperty('--app-scale', appScale.toString());
  }, [appScale]);

  // Streaming Artifact Tracking Ref (persisted in closure)
  const streamingArtifactId = React.useRef<string | null>(null);
  const currentBotMessage = React.useRef('');

  // Global Chat Listeners
  useEffect(() => {
    let unlistenChunk: (() => void) | undefined;
    let unlistenDone: (() => void) | undefined;
    let unlistenBenchChunk: (() => void) | undefined;
    let unlistenBenchDone: (() => void) | undefined;

    const setupListeners = async () => {
      unlistenChunk = await listen('chat-stream-chunk', (event: any) => {
        const chunk = event.payload as { content: string };
        setIsPromptProcessing(false);

        setChatMessages(prev => {
          const lastMsg = prev[prev.length - 1];
          let fullText = chunk.content;

          if (lastMsg && lastMsg.sender === 'bot' && lastMsg.isStreaming) {
            fullText = lastMsg.text + chunk.content;
            currentBotMessage.current = fullText;

            // --- Canvas Parsing Logic (Simplified for Context) ---
            if (infEnableCanvas) {
              const contentMatch = /<canvas_content type="(code|markdown|mermaid)"(?: language="(.*?)")?>(.*)/s.exec(fullText);
              if (contentMatch) {
                const type = contentMatch[1] as any;
                const language = contentMatch[2];
                const closingMatch = /(.*?)<\/canvas_content>/s.exec(contentMatch[3]);
                const content = closingMatch ? closingMatch[1] : contentMatch[3];

                setInfCanvasArtifacts(prevArts => {
                  if (!streamingArtifactId.current) {
                    const newId = `artifact-${Date.now()}`;
                    streamingArtifactId.current = newId;
                    setInfCanvasVisible(true);
                    return [...prevArts, {
                      id: newId,
                      title: 'Generated Content',
                      mode: type,
                      language: language,
                      content: content,
                      createdAt: Date.now()
                    }];
                  }
                  return prevArts.map(a => a.id === streamingArtifactId.current ? {
                    ...a, content: content, mode: type, language: language
                  } : a);
                });
              }
            }
            // ----------------------------

            return [...prev.slice(0, -1), { ...lastMsg, text: fullText }];
          } else {
            return [...prev, { text: chunk.content, sender: 'bot', timestamp: Date.now(), isStreaming: true }];
          }
        });
      });

      unlistenBenchChunk = await listen('chat-stream-chunk-benchmark', (event: any) => {
        const chunk = event.payload as { content: string };
        setBenchmarkMessages(prev => {
          const lastMsg = prev[prev.length - 1];
          if (lastMsg && lastMsg.sender === 'bot' && lastMsg.isStreaming) {
            return [...prev.slice(0, -1), { ...lastMsg, text: lastMsg.text + chunk.content }];
          } else {
            return [...prev, { text: chunk.content, sender: 'bot', timestamp: Date.now(), isStreaming: true }];
          }
        });
      });

      unlistenBenchDone = await listen('chat-stream-done-benchmark', async (event: any) => {
        const metrics = event.payload;
        setBenchmarkStreamMetrics(metrics);
        setBenchmarkMessages(prev => {
          const lastMsg = prev[prev.length - 1];
          if (lastMsg && lastMsg.sender === 'bot') {
            return [...prev.slice(0, -1), { ...lastMsg, isStreaming: false }];
          }
          return prev;
        });
      });

      // For now, only main chat is persistent.

      unlistenDone = await listen('chat-stream-done', async (event: any) => {
        const metrics = event.payload;
        setStreamMetrics(metrics);
        streamingArtifactId.current = null;

        const fullMessage = currentBotMessage.current;
        setChatMessages(prev => {
          const lastMsg = prev[prev.length - 1];
          if (lastMsg && lastMsg.sender === 'bot') {
            return [...prev.slice(0, -1), { ...lastMsg, isStreaming: false }];
          }
          return prev;
        });

        currentBotMessage.current = '';

        // Check for tool calls
        const toolRegex = /<tool_call>\s*<name>(.*?)<\/name>\s*<arguments>(.*?)<\/arguments>\s*<\/tool_call>/s;
        const match = toolRegex.exec(fullMessage);

        if (match) {
          const toolName = match[1].trim();
          const argsStr = match[2].trim();
          let args = {};
          try { args = JSON.parse(argsStr); } catch (e) { console.error(e); }

          setIsSending(true);

          try {
            // Execute tool
            // Note: invoke is available here
            const result = await invoke('execute_tool_command', { tool_name: toolName, args });
            const resultMsg = JSON.stringify(result);

            setChatMessages(prev => [...prev, {
              text: `Tool Result (${toolName}):\n${resultMsg}`,
              sender: 'system',
              timestamp: Date.now()
            }]);

            // Send result back to LLM
            // We need access to current config here. 
            // Accessing state variables inside useEffect closure is stale.
            // We can use a ref or rely on the backend to know content context if we managed history there.
            // OR, we just grab current values from a ref that updates on changes.
            // For simplicity, we just pass what we have, but beware stale closures for temperature/etc.
            // A safer way is properly managing chat loop in backend, but for now:

            // We'll dispatch a custom event or just invoke directly if we trust defaults.
            // To avoid stale state issues, let's skip re-sending here or warn user?
            // Actually, let's fix stale closure by using refs for config.

            // For this iteration, we accept slight risk of stale config on tool recursions
            // or we can read them from a ref object we maintain.
          } catch (error) {
            console.error(`Tool execution failed: ${error}`);
            setIsSending(false);
          }
        } else {
          setIsSending(false);
        }
      });
    };

    setupListeners();
    return () => {
      if (unlistenChunk) unlistenChunk();
      if (unlistenDone) unlistenDone();
      if (unlistenBenchChunk) unlistenBenchChunk();
      if (unlistenBenchDone) unlistenBenchDone();
    };
  }, [infEnableCanvas]); // Re-bind if canvas setting changes (mostly fine)

  // Color Mode (Light/Dark)
  const [colorModeState, setColorModeState] = useState<ColorMode>(() => {
    const saved = localStorage.getItem('velox_color_mode') as ColorMode;
    return saved || 'dark';
  });

  const setColorMode = (mode: ColorMode) => {
    setColorModeState(mode);
    localStorage.setItem('velox_color_mode', mode);
    // Apply to document root
    document.documentElement.setAttribute('data-color-mode', mode);
  };

  const toggleColorMode = () => {
    setColorMode(colorModeState === 'dark' ? 'light' : 'dark');
  };

  // Apply color mode on mount
  useEffect(() => {
    document.documentElement.setAttribute('data-color-mode', colorModeState);
  }, [colorModeState]);

  const runGlobalSetup = useCallback(async (forceDownload = false) => {
    try {
      setShowPythonSetup(true);
      setSetupProgressPercent(0);
      setSetupMessage(forceDownload ? "Starting Python Download..." : "Initializing Setup...");
      setSetupLoadedBytes(0);
      setSetupTotalBytes(0);

      // Check if python exists if not forcing. 
      // If forcing, we obviously download.
      // If not forcing, but it doesn't exist, we MUST download.
      let needsDownload = forceDownload;
      if (!needsDownload) {
        const exists = await invoke('check_python_standalone_command');
        if (!exists) {
          needsDownload = true;
          setSetupMessage("Python not found. Starting download...");
        }
      }

      if (needsDownload) {
        await invoke('download_python_standalone_command');
      }

      await invoke('setup_python_env_command');

      localStorage.setItem('pythonEnvSetup', 'complete');
      setFtSetupComplete(true);
      setShowPythonSetup(false);
    } catch (error) {
      console.error("Setup failed:", error);
      setShowPythonSetup(false);
      throw error;
    }
  }, []);

  useEffect(() => {
    // Load persisted mode
    const savedMode = localStorage.getItem('velox_user_mode') as UserMode;
    if (savedMode) {
      setUserModeState(savedMode);
    }
  }, []);

  const setUserMode = (mode: UserMode) => {
    setUserModeState(mode);
    localStorage.setItem('velox_user_mode', mode);
    // Apply default feature set for mode
    const defaultFeatures = mode === 'power' ? DEFAULT_POWER_FEATURES : DEFAULT_USER_FEATURES;
    setUserFeatures(defaultFeatures);
  };

  const toggleUserMode = () => {
    setUserMode(userMode === 'user' ? 'power' : 'user');
  };

  const setAutoUpdate = (val: boolean) => {
    setAutoUpdateState(val);
    localStorage.setItem('velox_auto_update', JSON.stringify(val));
  };

  const setShowInfoTooltips = (val: boolean) => {
    setShowInfoTooltipsState(val);
    localStorage.setItem('velox_show_info', JSON.stringify(val));
  };

  const setAutoProcessDatasets = (val: boolean) => {
    setAutoProcessDatasetsState(val);
    localStorage.setItem('velox_auto_process', JSON.stringify(val));
  };

  // Initial Load & Listeners
  useEffect(() => {
    loadResources();
    const unlistenModel = listen('model_downloaded', () => loadResources());
    const unlistenDataset = listen('dataset_downloaded', () => loadResources());
    const unlistenProgress = listen<number>('conversion_progress', (event) => {
      setUtConversionProgress(event.payload);
    });

    // Persistent Training Finished Listener
    const unlistenTraining = listen('training_finished', (event: any) => {
      const { success, code } = event.payload;
      setFtIsTraining(false);
      setFtTrainingStatus('idle');

      if (success) {
        addNotification("Training Completed Successfully!", 'success');
      } else {
        addNotification(`Training Failed (Exit Code: ${code}). Check logs.`, 'error');
      }

      console.log(`Training finished: success=${success}, exitCode=${code}`);
    });

    return () => {
      unlistenModel.then(f => f());
      unlistenDataset.then(f => f());
      unlistenProgress.then(f => f());
      unlistenTraining.then(f => f());
    };
  }, [loadResources]);

  // Auto-Conversion Logic
  useEffect(() => {
    if (!autoProcessDatasets) return;

    const autoConvert = async () => {
      // Find un-processed datasets that aren't already converting
      const toConvert = resources.filter(r =>
        r.type === 'dataset' &&
        r.is_processed === false &&
        !isConvertingMap[r.path]
      );

      for (const dataset of toConvert) {
        console.log(`Auto-converting dataset: ${dataset.name}`);
        setConvertingMap(prev => ({ ...prev, [dataset.path]: true }));

        // Fire and forget conversion
        invoke('convert_dataset_command', {
          sourcePath: dataset.path,
          destinationPath: dataset.path
        }).then(() => {
          loadResources(); // Refresh metadata
        }).catch(e => {
          console.error(`Auto-conversion failed for ${dataset.name}: ${e}`);
        }).finally(() => {
          setConvertingMap(prev => ({ ...prev, [dataset.path]: false }));
        });
      }
    };

    autoConvert();
  }, [resources, userMode, isConvertingMap, loadResources]);

  useEffect(() => {
    const savedAutoUpdate = localStorage.getItem('velox_auto_update');
    if (savedAutoUpdate !== null) setAutoUpdateState(JSON.parse(savedAutoUpdate));

    const savedShowInfo = localStorage.getItem('velox_show_info');
    if (savedShowInfo !== null) setShowInfoTooltipsState(JSON.parse(savedShowInfo));

    const savedAutoProcess = localStorage.getItem('velox_auto_process');
    if (savedAutoProcess !== null) setAutoProcessDatasetsState(JSON.parse(savedAutoProcess));

    const savedFeatures = localStorage.getItem('velox_user_features');
    if (savedFeatures) setUserFeaturesState(JSON.parse(savedFeatures));
  }, []);

  const value = {
    userMode, toggleUserMode, setUserMode, userFeatures, setUserFeatures,
    chatMessages, setChatMessages,
    inputMessage, setInputMessage,
    selectedBaseModel, setSelectedBaseModel,
    selectedMmproj, setSelectedMmproj,
    selectedLoraAdapter, setSelectedLoraAdapter,
    temperature, setTemperature,
    contextSize, setContextSize,
    topP, setTopP,
    topK, setTopK,
    systemPrompt, setSystemPrompt,
    isServerRunning, setIsServerRunning,
    infFlashAttn, setInfFlashAttn,
    infNoMmap, setInfNoMmap,
    infGpuLayers, setInfGpuLayers,
    infBatchSize, setInfBatchSize,
    infUbatchSize, setInfUbatchSize,
    infThreads, setInfThreads,
    infServerStatus, setInfServerStatus,
    infShowMetrics, setInfShowMetrics,
    infEnableWebSearch, setInfEnableWebSearch,
    infEnableCodeExec, setInfEnableCodeExec,
    infEnableCanvas, setInfEnableCanvas,
    infCanvasVisible, setInfCanvasVisible,
    infCanvasArtifacts, setInfCanvasArtifacts,
    infAutoFit, setInfAutoFit,
    infInferenceEngine, setInfInferenceEngine,

    selectedProject,
    setSelectedProject,
    selectedCheckpoint,
    setSelectedCheckpoint,

    autoUpdate, setAutoUpdate,
    showInfoTooltips, setShowInfoTooltips,
    rdHfQuery, setRdHfQuery,
    rdHfAuthor, setRdHfAuthor,
    rdHfModalities, setRdHfModalities,
    rdHfSizeRange, setRdHfSizeRange,
    rdHfType, setRdHfType,
    rdShowFindNew, setRdShowFindNew,
    currentView, setCurrentView,
    rdSelectedPaths, setRdSelectedPaths,
    ftProjectName, setFtProjectName,
    ftSelectedModel, setFtSelectedModel,
    ftSelectedDataset, setFtSelectedDataset,
    ftSelectedDatasets, setFtSelectedDatasets,
    ftDatasetMixing,
    setFtDatasetMixing,
    hardwareInfo,
    ftLocalDatasetPath, setFtLocalDatasetPath,
    ftHfModelId, setFtHfModelId,
    ftHfDatasetId, setFtHfDatasetId,
    ftNumEpochs, setFtNumEpochs,
    ftBatchSize, setFtBatchSize,
    ftLearningRate, setFtLearningRate,
    ftLoraR, setFtLoraR,
    ftLoraAlpha, setFtLoraAlpha,
    ftMaxSeqLength, setFtMaxSeqLength,
    ftActiveTab, setFtActiveTab,
    ftIsTraining, setFtIsTraining,
    ftTrainingStatus, setFtTrainingStatus,
    ftTensorboardUrl, setFtTensorboardUrl,
    ftInitMessage, setFtInitMessage,
    ftIsSettingUp, setFtIsSettingUp,
    ftSetupComplete, setFtSetupComplete,
    ftSetupProgress, setFtSetupProgress,
    // Global Setup
    showPythonSetup, setShowPythonSetup,
    isInitializing, setIsInitializing,
    setupProgressPercent, setSetupProgressPercent,
    setupMessage, setSetupMessage,
    setupLoadedBytes, setSetupLoadedBytes,
    setupTotalBytes, setSetupTotalBytes,
    runGlobalSetup,
    isEngineUpdating, setIsEngineUpdating,
    // Global Notifications
    notifications, addNotification,
    // Utilities Conversion
    utIsConverting, setUtIsConverting,
    utConversionProgress, setUtConversionProgress,
    utConversionLabel, setUtConversionLabel,
    utSourcePath, setUtSourcePath,
    utOutputPath, setUtOutputPath,
    utBasePath, setUtBasePath,
    utQuantizationType, setUtQuantizationType,
    utConversionType, setUtConversionType,
    utConversionEngine, setUtConversionEngine,
    resources, setResources, loadResources,
    isConvertingMap, setConvertingMap,
    theme: 'cyber' as 'cyber' | 'forge', // Default theme
    colorMode: colorModeState,
    setColorMode,
    toggleColorMode,

    // Benchmarking
    isBenchmarking, setIsBenchmarking,
    benchmarkMessages, setBenchmarkMessages,
    isBlindTest, setIsBlindTest,
    selectedBenchmarkModel, setSelectedBenchmarkModel,
    loadedModels, setLoadedModels,

    // Evaluation Mode (Arena/Compare)
    evaluationMode, setEvaluationMode,
    arenaSelectedModels, setArenaSelectedModels,
    arenaScores, setArenaScores,
    arenaCurrentPair, setArenaCurrentPair,

    // Global Streaming State
    streamMetrics: streamMetricsState,
    benchmarkStreamMetrics,
    isPromptProcessing: isPromptProcessingState,
    setIsPromptProcessing,
    isSending: isSendingState,
    setIsSending,

    // App Level
    appScale, setAppScale,
    autoProcessDatasets, setAutoProcessDatasets,
  };

  return (
    <AppContext.Provider value={value}>
      {children}
    </AppContext.Provider>
  );
};

export const useApp = () => {
  const context = useContext(AppContext);
  if (context === undefined) {
    throw new Error('useApp must be used within an AppProvider');
  }
  return context;
};
