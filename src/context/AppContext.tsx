import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { invoke } from "@tauri-apps/api/core";

type UserMode = 'user' | 'power';

export interface ChatMessage {
  text: string;
  sender: 'user' | 'bot' | 'system';
  timestamp: number;
  isStreaming?: boolean;
}

interface AppContextType {
  // User Mode
  userMode: UserMode;
  toggleUserMode: () => void;
  setUserMode: (mode: UserMode) => void;

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

  // Resource Dashboard State
  rdHfQuery: string;
  setRdHfQuery: (query: string) => void;
  rdHfType: 'model' | 'dataset';
  setRdHfType: (type: 'model' | 'dataset') => void;
  rdSelectedPaths: Set<string>;
  setRdSelectedPaths: React.Dispatch<React.SetStateAction<Set<string>>>;

  // Fine-Tuning State
  ftProjectName: string;
  setFtProjectName: (name: string) => void;
  ftSelectedModel: string;
  setFtSelectedModel: (model: string) => void;
  ftSelectedDataset: string;
  setFtSelectedDataset: (dataset: string) => void;
  ftLocalDatasetPath: string;
  setFtLocalDatasetPath: (path: string) => void;
  ftHfModelId: string;
  setFtHfModelId: (id: string) => void;
  ftHfDatasetId: string;
  setFtHfDatasetId: (id: string) => void;
  ftNumEpochs: number;
  setFtNumEpochs: (n: number) => void;
  ftBatchSize: number;
  setFtBatchSize: (n: number) => void;
  ftLearningRate: number;
  setFtLearningRate: (n: number) => void;
  ftLoraR: number;
  setFtLoraR: (n: number) => void;
  ftLoraAlpha: number;
  setFtLoraAlpha: (n: number) => void;
  ftMaxSeqLength: number;
  setFtMaxSeqLength: (n: number) => void;
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
}

const AppContext = createContext<AppContextType | undefined>(undefined);

export const AppProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  // User Mode
  const [userMode, setUserModeState] = useState<UserMode>('user');

  // Inference State
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [selectedBaseModel, setSelectedBaseModel] = useState('');
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
  const [infEnableCanvas, setInfEnableCanvas] = useState(true); // Canvas enabled by default

  // Resource Dashboard State
  const [rdHfQuery, setRdHfQuery] = useState('');
  const [rdHfType, setRdHfType] = useState<'model' | 'dataset'>('model');
  const [rdSelectedPaths, setRdSelectedPaths] = useState<Set<string>>(new Set());

  // Fine-Tuning State
  const [ftProjectName, setFtProjectName] = useState('');
  const [ftSelectedModel, setFtSelectedModel] = useState('');
  const [ftSelectedDataset, setFtSelectedDataset] = useState('');
  const [ftLocalDatasetPath, setFtLocalDatasetPath] = useState('');
  const [ftHfModelId, setFtHfModelId] = useState('');
  const [ftHfDatasetId, setFtHfDatasetId] = useState('');
  const [ftNumEpochs, setFtNumEpochs] = useState(3);
  const [ftBatchSize, setFtBatchSize] = useState(4);
  const [ftLearningRate, setFtLearningRate] = useState(2e-4);
  const [ftLoraR, setFtLoraR] = useState(16);
  const [ftLoraAlpha, setFtLoraAlpha] = useState(32);
  const [ftMaxSeqLength, setFtMaxSeqLength] = useState(2048);
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

  // Global Setup State
  const [showPythonSetup, setShowPythonSetup] = useState(false);
  const [isInitializing, setIsInitializing] = useState(false); // Default false in context, App.tsx sets it true on mount? Or manage here?
  const [setupProgressPercent, setSetupProgressPercent] = useState(0);
  const [setupMessage, setSetupMessage] = useState("Initializing environment...");
  const [setupLoadedBytes, setSetupLoadedBytes] = useState(0);
  const [setupTotalBytes, setSetupTotalBytes] = useState(0);

  const runGlobalSetup = useCallback(async (forceDownload = false) => {
    try {
      setShowPythonSetup(true);
      setSetupProgressPercent(0);
      setSetupMessage(forceDownload ? "Starting Python Download..." : "Initializing Setup...");
      setSetupLoadedBytes(0);
      setSetupTotalBytes(0);

      if (forceDownload) {
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
  };

  const toggleUserMode = () => {
    setUserMode(userMode === 'user' ? 'power' : 'user');
  };

  const value = {
    userMode, toggleUserMode, setUserMode,
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
    rdHfQuery, setRdHfQuery,
    rdHfType, setRdHfType,
    rdSelectedPaths, setRdSelectedPaths,
    ftProjectName, setFtProjectName,
    ftSelectedModel, setFtSelectedModel,
    ftSelectedDataset, setFtSelectedDataset,
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
