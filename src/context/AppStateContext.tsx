import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

interface DownloadProgress {
    id: string;
    filename: string;
    totalBytes: number;
    downloadedBytes: number;
    percentage: number;
    speed: string;
    status: 'downloading' | 'completed' | 'cancelled' | 'failed';
}

interface LoadedModel {
    path: string;
    name: string;
    serverId?: string;
    serverPort?: number;
    hasMMProj?: boolean;
    mmProjPath?: string;
}

interface FineTuningState {
    selectedModel: string;
    selectedDataset: string;
    localDatasetPath: string;
    numEpochs: number;
    batchSize: number;
    learningRate: number;
    loraR: number;
    loraAlpha: number;
    maxSeqLength: number;
    isTraining: boolean;
    trainingStatus: 'idle' | 'initializing' | 'training';
}

interface ResourceDashboardState {
    activeTab: 'models' | 'datasets';
    searchQuery: string;
    selectedItems: Set<string>;
}

interface InferenceState {
    loadedModels: LoadedModel[];
    selectedModelPath: string | null;
    systemPrompt: string;
    temperature: number;
    maxTokens: number;
}

interface AppState {
    fineTuning: FineTuningState;
    resourceDashboard: ResourceDashboardState;
    inference: InferenceState;
    downloads: Map<string, DownloadProgress>;
}

interface AppStateContextType {
    state: AppState;
    updateFineTuning: (updates: Partial<FineTuningState>) => void;
    updateResourceDashboard: (updates: Partial<ResourceDashboardState>) => void;
    updateInference: (updates: Partial<InferenceState>) => void;
    addDownload: (download: DownloadProgress) => void;
    updateDownload: (id: string, updates: Partial<DownloadProgress>) => void;
    removeDownload: (id: string) => void;
    loadModel: (model: LoadedModel) => void;
    unloadModel: (path: string) => void;
    selectModel: (path: string) => void;
}

// ============================================================================
// DEFAULT STATE
// ============================================================================

const defaultState: AppState = {
    fineTuning: {
        selectedModel: '',
        selectedDataset: '',
        localDatasetPath: '',
        numEpochs: 3,
        batchSize: 4,
        learningRate: 2e-4,
        loraR: 16,
        loraAlpha: 32,
        maxSeqLength: 2048,
        isTraining: false,
        trainingStatus: 'idle',
    },
    resourceDashboard: {
        activeTab: 'models',
        searchQuery: '',
        selectedItems: new Set(),
    },
    inference: {
        loadedModels: [],
        selectedModelPath: null,
        systemPrompt: 'You are a helpful assistant.',
        temperature: 0.7,
        maxTokens: 2048,
    },
    downloads: new Map(),
};

// ============================================================================
// CONTEXT
// ============================================================================

const AppStateContext = createContext<AppStateContextType | undefined>(undefined);

// ============================================================================
// PROVIDER
// ============================================================================

interface AppStateProviderProps {
    children: ReactNode;
}

export const AppStateProvider: React.FC<AppStateProviderProps> = ({ children }) => {
    const [state, setState] = useState<AppState>(() => {
        // Load from localStorage on mount
        const saved = localStorage.getItem('appState');
        if (saved) {
            try {
                const parsed = JSON.parse(saved);
                // Convert selectedItems back to Set
                if (parsed.resourceDashboard?.selectedItems) {
                    parsed.resourceDashboard.selectedItems = new Set(
                        parsed.resourceDashboard.selectedItems
                    );
                }
                // Convert downloads back to Map
                if (parsed.downloads) {
                    parsed.downloads = new Map(Object.entries(parsed.downloads));
                }
                return { ...defaultState, ...parsed };
            } catch (e) {
                console.error('Failed to parse saved state:', e);
            }
        }
        return defaultState;
    });

    // Save to localStorage whenever state changes (debounced)
    useEffect(() => {
        const timeoutId = setTimeout(() => {
            const toSave = {
                ...state,
                // Convert Set to Array for JSON
                resourceDashboard: {
                    ...state.resourceDashboard,
                    selectedItems: Array.from(state.resourceDashboard.selectedItems),
                },
                // Convert Map to Object for JSON
                downloads: Object.fromEntries(state.downloads),
            };
            localStorage.setItem('appState', JSON.stringify(toSave));
        }, 500); // Debounce 500ms

        return () => clearTimeout(timeoutId);
    }, [state]);

    // ========================================================================
    // UPDATE FUNCTIONS
    // ========================================================================

    const updateFineTuning = (updates: Partial<FineTuningState>) => {
        setState(prev => ({
            ...prev,
            fineTuning: { ...prev.fineTuning, ...updates },
        }));
    };

    const updateResourceDashboard = (updates: Partial<ResourceDashboardState>) => {
        setState(prev => ({
            ...prev,
            resourceDashboard: { ...prev.resourceDashboard, ...updates },
        }));
    };

    const updateInference = (updates: Partial<InferenceState>) => {
        setState(prev => ({
            ...prev,
            inference: { ...prev.inference, ...updates },
        }));
    };

    const addDownload = (download: DownloadProgress) => {
        setState(prev => {
            const newDownloads = new Map(prev.downloads);
            newDownloads.set(download.id, download);
            return { ...prev, downloads: newDownloads };
        });
    };

    const updateDownload = (id: string, updates: Partial<DownloadProgress>) => {
        setState(prev => {
            const newDownloads = new Map(prev.downloads);
            const existing = newDownloads.get(id);
            if (existing) {
                newDownloads.set(id, { ...existing, ...updates });
            }
            return { ...prev, downloads: newDownloads };
        });
    };

    const removeDownload = (id: string) => {
        setState(prev => {
            const newDownloads = new Map(prev.downloads);
            newDownloads.delete(id);
            return { ...prev, downloads: newDownloads };
        });
    };

    const loadModel = (model: LoadedModel) => {
        setState(prev => ({
            ...prev,
            inference: {
                ...prev.inference,
                loadedModels: [...prev.inference.loadedModels, model],
                // Auto-select if it's the first model
                selectedModelPath: prev.inference.selectedModelPath || model.path,
            },
        }));
    };

    const unloadModel = (path: string) => {
        setState(prev => ({
            ...prev,
            inference: {
                ...prev.inference,
                loadedModels: prev.inference.loadedModels.filter(m => m.path !== path),
                // Clear selection if unloading the selected model
                selectedModelPath: prev.inference.selectedModelPath === path
                    ? prev.inference.loadedModels.find(m => m.path !== path)?.path || null
                    : prev.inference.selectedModelPath,
            },
        }));
    };

    const selectModel = (path: string) => {
        setState(prev => ({
            ...prev,
            inference: {
                ...prev.inference,
                selectedModelPath: path,
            },
        }));
    };

    const value: AppStateContextType = {
        state,
        updateFineTuning,
        updateResourceDashboard,
        updateInference,
        addDownload,
        updateDownload,
        removeDownload,
        loadModel,
        unloadModel,
        selectModel,
    };

    return (
        <AppStateContext.Provider value={value}>
            {children}
        </AppStateContext.Provider>
    );
};

// ============================================================================
// HOOK
// ============================================================================

export const useAppState = () => {
    const context = useContext(AppStateContext);
    if (!context) {
        throw new Error('useAppState must be used within AppStateProvider');
    }
    return context;
};

// Export types for use in other files
export type { AppState, FineTuningState, ResourceDashboardState, InferenceState, DownloadProgress, LoadedModel };
