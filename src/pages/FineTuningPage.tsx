import React, { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { open } from '@tauri-apps/plugin-dialog';
import { appDataDir } from '@tauri-apps/api/path';
import { listen } from '@tauri-apps/api/event';
import { useApp } from '../context/AppContext';
import { Card } from '../components/Card';
import { Button } from '../components/Button';
import { Input, Select } from '../components/Input';

import { BrainCircuit, Database, Play, Square, Download, FolderOpen, Activity, Terminal } from 'lucide-react';

interface FineTuningPageProps {
    addLogMessage: (message: string) => void;
    addNotification: (message: string, type?: 'success' | 'error' | 'info') => void;
}

const FineTuningPage: React.FC<FineTuningPageProps> = ({ addLogMessage, addNotification }) => {
    const {
        userMode,
        ftProjectName, setFtProjectName,
        ftSelectedModel: selectedModel, setFtSelectedModel: setSelectedModel,
        ftSelectedDataset: selectedDataset, setFtSelectedDataset: setSelectedDataset,
        ftLocalDatasetPath: localDatasetPath, setFtLocalDatasetPath: setLocalDatasetPath,
        ftHfModelId: hfModelId, setFtHfModelId: setHfModelId,
        ftHfDatasetId: hfDatasetId, setFtHfDatasetId: setHfDatasetId,
        ftNumEpochs: numEpochs, setFtNumEpochs: setNumEpochs,
        ftBatchSize: batchSize, setFtBatchSize: setBatchSize,
        ftLearningRate: learningRate, setFtLearningRate: setLearningRate,
        ftLoraR: loraR, setFtLoraR: setLoraR,
        ftLoraAlpha: loraAlpha, setFtLoraAlpha: setLoraAlpha,
        ftMaxSeqLength: maxSeqLength, setFtMaxSeqLength: setMaxSeqLength,
        ftActiveTab: activeTab, setFtActiveTab: setActiveTab,
        ftIsTraining: isTraining, setFtIsTraining: setIsTraining,
        ftTrainingStatus: trainingStatus, setFtTrainingStatus: setTrainingStatus,
        ftTensorboardUrl: tensorboardUrl, setFtTensorboardUrl: setTensorboardUrl,
        ftInitMessage: initMessage, setFtInitMessage: setInitMessage
    } = useApp();

    // Resources
    const [availableModels, setAvailableModels] = useState<string[]>([]);
    const [availableDatasets, setAvailableDatasets] = useState<string[]>([]);
    const [existingProjects, setExistingProjects] = useState<string[]>([]);

    // HF Downloads
    const [isDownloadingModel, setIsDownloadingModel] = useState(false);
    const [isDownloadingDataset, setIsDownloadingDataset] = useState(false);

    // Setup State
    const [isSettingUp, setIsSettingUp] = useState(false);
    const [setupComplete, setSetupComplete] = useState(false);
    const [setupProgress, setSetupProgress] = useState({ current: 0, total: 0, message: '' });

    useEffect(() => {
        loadResources();
        checkSetupStatus();

        const unlistenPromise = listen('log', (event) => {
            const msg = event.payload as string;
            const progressMatch = msg.match(/📦 \[(\d+)\/(\d+)\]/);
            if (progressMatch) {
                setSetupProgress({
                    current: parseInt(progressMatch[1]),
                    total: parseInt(progressMatch[2]),
                    message: msg.split(']')[1]?.trim() || ''
                });
            }
            if (msg.includes('✅ Python environment setup complete!')) {
                setSetupComplete(true);
                localStorage.setItem('pythonEnvSetup', 'complete');
            }
            if (msg.includes('STATUS:')) {
                const statusText = msg.split('STATUS:')[1].trim();
                setInitMessage(statusText);
                // If the STATUS contains a URL for TensorBoard, set iframe URL
                const urlMatch = statusText.match(/https?:\/\/[^\s]+/);
                if (urlMatch) {
                    setTensorboardUrl(urlMatch[0]);
                }
            }
            if (msg.includes('🚀 Starting Training')) {
                setTrainingStatus('training');
            }
        });

        return () => {
            unlistenPromise.then(f => f());
        };
    }, []);

    const checkSetupStatus = () => {
        if (localStorage.getItem('pythonEnvSetup') === 'complete') {
            setSetupComplete(true);
        }
    };

    const loadResources = async () => {
        try {
            const models: string[] = await invoke('list_finetuning_models_command');
            const datasets: string[] = await invoke('list_dataset_folders_command');
            const projects: string[] = await invoke('list_training_projects_command');
            setAvailableModels(models);
            setAvailableDatasets(datasets);
            setExistingProjects(projects);
        } catch (error) {
            addLogMessage(`ERROR loading resources: ${error}`);
        }
    };

    const handleDownloadModel = async () => {
        if (!hfModelId.trim()) return;
        setIsDownloadingModel(true);
        try {
            await invoke('download_hf_model_command', { modelId: hfModelId, files: null, token: null });
            addNotification('Model downloaded successfully', 'success');
            setHfModelId('');
            loadResources();
        } catch (error) {
            addNotification('Model download failed', 'error');
            addLogMessage(`Download error: ${error}`);
        } finally {
            setIsDownloadingModel(false);
        }
    };

    const handleDownloadDataset = async () => {
        if (!hfDatasetId.trim()) return;
        setIsDownloadingDataset(true);
        try {
            await invoke('download_hf_dataset_command', { datasetId: hfDatasetId, token: null });
            addNotification('Dataset downloaded successfully', 'success');
            setHfDatasetId('');
            loadResources();
        } catch (error) {
            addNotification('Dataset download failed', 'error');
            addLogMessage(`Download error: ${error}`);
        } finally {
            setIsDownloadingDataset(false);
        }
    };

    const handleBrowseDataset = async () => {
        try {
            const selected = await open({
                multiple: false,
                directory: true,
                defaultPath: await appDataDir(),
            });
            if (typeof selected === 'string') {
                setLocalDatasetPath(selected);
            }
        } catch (error) {
            console.error(error);
        }
    };

    const handleStartTraining = async () => {
        if (!ftProjectName.trim()) {
            addNotification('Please enter a project name', 'error');
            return;
        }
        if (!selectedModel || (!selectedDataset && !localDatasetPath)) {
            addNotification('Please select a model and dataset', 'error');
            return;
        }

        setIsTraining(true);
        setTrainingStatus('initializing');
        setInitMessage('Starting process...');
        setActiveTab('training');
        addLogMessage('Starting training...');

        try {
            const result = await invoke('start_training_command', {
                projectName: ftProjectName.trim(),
                modelPath: selectedModel,
                datasetPath: localDatasetPath || selectedDataset,
                numEpochs,
                batchSize,
                learningRate,
                loraR,
                loraAlpha,
                maxSeqLength,
            });

            const port = (result as any).tensorboard_port || 6006;
            // Fallback: if backend STATUS is missed, set iframe after a longer delay
            setTimeout(() => {
                setTensorboardUrl(`http://localhost:${port}`);
            }, 12000);
            addNotification('Training started successfully', 'success');
        } catch (error) {
            addLogMessage(`ERROR starting training: ${error}`);
            addNotification('Failed to start training', 'error');
            addNotification('Failed to start training', 'error');
            setIsTraining(false);
            setTrainingStatus('idle');
            setActiveTab('config');
        }
    };

    const handleStopTraining = async () => {
        try {
            await invoke('stop_training_command');
            setIsTraining(false);
            setTrainingStatus('idle');
            setTensorboardUrl('');
            addNotification('Training stopped', 'info');
        } catch (error) {
            addLogMessage(`ERROR stopping: ${error}`);
        }
    };

    const handleSetupEnv = async () => {
        setIsSettingUp(true);
        setSetupProgress({ current: 0, total: 7, message: 'Initializing...' });
        try {
            await invoke('setup_python_env_command');
            setSetupComplete(true);
            localStorage.setItem('pythonEnvSetup', 'complete');
            addNotification('Environment setup complete', 'success');
        } catch (error) {
            addNotification('Setup failed', 'error');
        } finally {
            setIsSettingUp(false);
        }
    };

    if (!setupComplete) {
        return (
            <div className="flex flex-col items-center justify-center h-[60vh] text-center gap-6">
                <div className="p-6 rounded-full bg-white/5 border border-white/10">
                    <BrainCircuit size={48} className="text-accent-primary" />
                </div>
                <div>
                    <h2 className="text-2xl font-bold mb-2">Setup Required</h2>
                    <p className="text-gray-400 max-w-md">
                        To start fine-tuning, we need to set up the Python environment and dependencies. This only needs to happen once.
                    </p>
                </div>

                {isSettingUp ? (
                    <div className="w-full max-w-md space-y-4">
                        <div className="flex justify-between text-sm">
                            <span>Installing dependencies...</span>
                            <span>{setupProgress.current}/{setupProgress.total}</span>
                        </div>
                        <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                            <div
                                className="h-full bg-accent-gradient transition-all duration-300"
                                style={{ width: `${(setupProgress.current / setupProgress.total) * 100}%` }}
                            />
                        </div>
                        <p className="text-xs text-gray-500">{setupProgress.message}</p>
                    </div>
                ) : (
                    <Button size="lg" onClick={handleSetupEnv}>
                        Install Dependencies
                    </Button>
                )}
            </div>
        );
    }

    return (
        <div className="space-y-6">
            {/* Tabs */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 0, marginBottom: '10px' }}>
                <button
                    onClick={() => setActiveTab('config')}
                    style={{
                        background: activeTab === 'config' ? 'var(--accent-gradient)' : 'rgba(255,255,255,0.04)',
                        color: activeTab === 'config' ? '#000' : '#9ca3af',
                        padding: '12px 16px',
                        fontSize: '14px',
                        fontWeight: 600,
                        border: '1px solid rgba(255,255,255,0.06)',
                        cursor: 'pointer',
                        transition: 'all 0.24s ease',
                        borderTopLeftRadius: 8,
                        borderBottomLeftRadius: 0,
                        borderTopRightRadius: 0,
                        borderBottomRightRadius: 5,
                        marginRight: 1
                    }}
                    onMouseEnter={(e) => {
                        if (activeTab !== 'config') {
                            (e.currentTarget as HTMLButtonElement).style.background = 'rgba(255,255,255,0.06)';
                            (e.currentTarget as HTMLButtonElement).style.color = '#fff';
                        }
                    }}
                    onMouseLeave={(e) => {
                        if (activeTab !== 'config') {
                            (e.currentTarget as HTMLButtonElement).style.background = 'rgba(255,255,255,0.04)';
                            (e.currentTarget as HTMLButtonElement).style.color = '#9ca3af';
                        }
                    }}
                >
                    Configuration
                </button>

                <button
                    onClick={() => setActiveTab('training')}
                    style={{
                        background: activeTab === 'training' ? 'var(--accent-gradient)' : 'rgba(255,255,255,0.04)',
                        color: activeTab === 'training' ? '#000' : '#9ca3af',
                        padding: '12px 16px',
                        fontSize: '14px',
                        fontWeight: 600,
                        border: '1px solid rgba(255,255,255,0.06)',
                        cursor: 'pointer',
                        transition: 'all 0.24s ease',
                        borderTopLeftRadius: 0,
                        borderBottomLeftRadius: 0,
                        borderTopRightRadius: 8,
                        borderBottomRightRadius: 5,
                        marginLeft: 1
                    }}
                    onMouseEnter={(e) => {
                        if (activeTab !== 'training') {
                            (e.currentTarget as HTMLButtonElement).style.background = 'rgba(255,255,255,0.06)';
                            (e.currentTarget as HTMLButtonElement).style.color = '#fff';
                        }
                    }}
                    onMouseLeave={(e) => {
                        if (activeTab !== 'training') {
                            (e.currentTarget as HTMLButtonElement).style.background = 'rgba(255,255,255,0.04)';
                            (e.currentTarget as HTMLButtonElement).style.color = '#9ca3af';
                        }
                    }}
                >
                    Training Dashboard
                </button>
            </div>

            {activeTab === 'config' && (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '24px' }}>
                    {/* Left Column: Selection */}
                    <div className="space-y-6" style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
                        <Card>
                            <div className="flex items-center gap-3 mb-4">
                                <BrainCircuit className="text-accent-primary" size={24} />
                                <h3 className="text-lg font-semibold">Model Selection</h3>
                            </div>

                            {/* Model Tabs */}
                            <div style={{ display: 'flex', gap: '4px', marginBottom: '16px', background: 'rgba(255,255,255,0.03)', padding: '4px', borderRadius: '8px' }}>
                                <button
                                    onClick={() => {
                                        setHfModelId('');
                                        // Focus is now on local selection
                                    }}
                                    style={{
                                        flex: 1,
                                        padding: '8px 12px',
                                        fontSize: '13px',
                                        fontWeight: 500,
                                        border: 'none',
                                        background: !hfModelId ? 'rgba(167, 139, 250, 0.15)' : 'transparent',
                                        color: !hfModelId ? '#a78bfa' : '#9ca3af',
                                        borderRadius: '6px',
                                        cursor: 'pointer',
                                        transition: 'all 0.2s'
                                    }}
                                >
                                    📁 Local
                                </button>
                                <button
                                    onClick={() => {
                                        setSelectedModel('');
                                        if (!hfModelId) setHfModelId(' '); // Set to non-empty to trigger HF view
                                    }}
                                    style={{
                                        flex: 1,
                                        padding: '8px 12px',
                                        fontSize: '13px',
                                        fontWeight: 500,
                                        border: 'none',
                                        background: hfModelId ? 'rgba(167, 139, 250, 0.15)' : 'transparent',
                                        color: hfModelId ? '#a78bfa' : '#9ca3af',
                                        borderRadius: '6px',
                                        cursor: 'pointer',
                                        transition: 'all 0.2s'
                                    }}
                                >
                                    🤗 HuggingFace
                                </button>
                            </div>

                            <div className="space-y-4">
                                {!hfModelId ? (
                                    <Select
                                        label="Choose from downloaded models"
                                        options={[
                                            { value: '', label: 'Select a model...' },
                                            ...availableModels.map(m => ({ value: m, label: m }))
                                        ]}
                                        value={selectedModel}
                                        onChange={(e) => setSelectedModel(e.target.value)}
                                    />
                                ) : (
                                    <div className="flex gap-2">
                                        <Input
                                            label="HuggingFace Model ID"
                                            placeholder="e.g. meta-llama/Llama-2-7b"
                                            value={hfModelId.trim()}
                                            onChange={(e) => setHfModelId(e.target.value)}
                                            className="flex-1"
                                        />
                                        <Button
                                            variant="secondary"
                                            onClick={handleDownloadModel}
                                            isLoading={isDownloadingModel}
                                            leftIcon={<Download size={16} />}
                                            style={{ marginTop: '26px' }}
                                        >
                                            Download
                                        </Button>
                                    </div>
                                )}
                            </div>
                        </Card>

                        <Card>
                            <div className="flex items-center gap-3 mb-4">
                                <Database className="text-accent-secondary" size={24} />
                                <h3 className="text-lg font-semibold">Dataset Selection</h3>
                            </div>

                            {/* Dataset Tabs */}
                            <div style={{ display: 'flex', gap: '4px', marginBottom: '16px', background: 'rgba(255,255,255,0.03)', padding: '4px', borderRadius: '8px' }}>
                                <button
                                    onClick={() => {
                                        setHfDatasetId('');
                                        setLocalDatasetPath('');
                                    }}
                                    style={{
                                        flex: 1,
                                        padding: '8px 12px',
                                        fontSize: '13px',
                                        fontWeight: 500,
                                        border: 'none',
                                        background: (!hfDatasetId && !localDatasetPath) ? 'rgba(125, 211, 252, 0.15)' : 'transparent',
                                        color: (!hfDatasetId && !localDatasetPath) ? '#7dd3fc' : '#9ca3af',
                                        borderRadius: '6px',
                                        cursor: 'pointer',
                                        transition: 'all 0.2s'
                                    }}
                                >
                                    📁 Local
                                </button>
                                <button
                                    onClick={() => {
                                        setSelectedDataset('');
                                        setLocalDatasetPath('');
                                        if (!hfDatasetId) setHfDatasetId(' '); // Set to trigger HF view
                                    }}
                                    style={{
                                        flex: 1,
                                        padding: '8px 12px',
                                        fontSize: '13px',
                                        fontWeight: 500,
                                        border: 'none',
                                        background: hfDatasetId ? 'rgba(125, 211, 252, 0.15)' : 'transparent',
                                        color: hfDatasetId ? '#7dd3fc' : '#9ca3af',
                                        borderRadius: '6px',
                                        cursor: 'pointer',
                                        transition: 'all 0.2s'
                                    }}
                                >
                                    🤗 HuggingFace
                                </button>
                                <button
                                    onClick={() => {
                                        setSelectedDataset('');
                                        setHfDatasetId('');
                                        if (!localDatasetPath) setLocalDatasetPath(' '); // Set to trigger Browse view
                                    }}
                                    style={{
                                        flex: 1,
                                        padding: '8px 12px',
                                        fontSize: '13px',
                                        fontWeight: 500,
                                        border: 'none',
                                        background: localDatasetPath ? 'rgba(125, 211, 252, 0.15)' : 'transparent',
                                        color: localDatasetPath ? '#7dd3fc' : '#9ca3af',
                                        borderRadius: '6px',
                                        cursor: 'pointer',
                                        transition: 'all 0.2s'
                                    }}
                                >
                                    📂 Browse
                                </button>
                            </div>

                            <div className="space-y-4">
                                {!hfDatasetId && !localDatasetPath ? (
                                    <Select
                                        label="Choose from downloaded datasets"
                                        options={[
                                            { value: '', label: 'Select a dataset...' },
                                            ...availableDatasets.map(d => ({ value: d, label: d }))
                                        ]}
                                        value={selectedDataset}
                                        onChange={(e) => setSelectedDataset(e.target.value)}
                                    />
                                ) : hfDatasetId ? (
                                    <div className="flex gap-2">
                                        <Input
                                            label="HuggingFace Dataset ID"
                                            placeholder="e.g. tatsu-lab/alpaca"
                                            value={hfDatasetId.trim()}
                                            onChange={(e) => setHfDatasetId(e.target.value)}
                                            className="flex-1"
                                        />
                                        <Button
                                            variant="secondary"
                                            onClick={handleDownloadDataset}
                                            isLoading={isDownloadingDataset}
                                            leftIcon={<Download size={16} />}
                                            style={{ marginTop: '26px' }}
                                        >
                                            Download
                                        </Button>
                                    </div>
                                ) : (
                                    <div>
                                        <label className="text-sm font-medium mb-2 block">Custom Dataset Folder</label>
                                        <div className="flex flex-col gap-2">
                                            <Button
                                                variant="outline"
                                                onClick={handleBrowseDataset}
                                                leftIcon={<FolderOpen size={16} />}
                                            >
                                                Choose Folder
                                            </Button>
                                            {localDatasetPath && localDatasetPath.trim() && (
                                                <div className="text-xs text-gray-400 p-2 bg-white/5 rounded border border-white/10 truncate" title={localDatasetPath}>
                                                    📂 {localDatasetPath.trim()}
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                )}
                            </div>
                        </Card>
                    </div>

                    {/* Right Column: Parameters */}
                    <div className="space-y-6" style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
                        <Card>
                            <div className="flex items-center gap-3 mb-4">
                                <BrainCircuit className="text-blue-500" size={24} />
                                <h3 className="text-lg font-semibold">Project Name</h3>
                            </div>

                            {existingProjects.length > 0 && (
                                <>
                                    <Select
                                        label="Select Existing Project"
                                        options={[
                                            { value: '', label: 'Create new project...' },
                                            ...existingProjects.map(p => ({ value: p, label: p }))
                                        ]}
                                        value={existingProjects.includes(ftProjectName) ? ftProjectName : ''}
                                        onChange={(e) => setFtProjectName(e.target.value)}
                                    />
                                    <div className="relative my-3">
                                        <div className="absolute inset-0 flex items-center">
                                            <div className="w-full border-t border-white/10"></div>
                                        </div>
                                        <div className="relative flex justify-center text-xs uppercase">
                                            <span className="bg-panel px-2 text-gray-500" style={{ background: '#141419' }}>Or create new</span>
                                        </div>
                                    </div>
                                </>
                            )}

                            <Input
                                placeholder="e.g., my-custom-model"
                                value={ftProjectName}
                                onChange={(e) => setFtProjectName(e.target.value)}
                            />
                            <p className="text-xs text-gray-500 mt-2">
                                All checkpoints and outputs will be saved under this project name
                            </p>
                        </Card>

                        <Card>
                            <div className="flex items-center gap-3 mb-4">
                                <Activity className="text-green-500" size={24} />
                                <h3 className="text-lg font-semibold">Training Parameters</h3>
                            </div>

                            <div className="grid grid-cols-2 gap-4" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                                <Input
                                    label="Epochs"
                                    type="number"
                                    value={numEpochs}
                                    onChange={(e) => setNumEpochs(Number(e.target.value))}
                                />
                                <Input
                                    label="Batch Size"
                                    type="number"
                                    value={batchSize}
                                    onChange={(e) => setBatchSize(Number(e.target.value))}
                                />

                                {userMode === 'power' && (
                                    <>
                                        <Input
                                            label="Learning Rate"
                                            type="number"
                                            step="0.00001"
                                            value={learningRate}
                                            onChange={(e) => setLearningRate(Number(e.target.value))}
                                        />
                                        <Input
                                            label="Max Seq Length"
                                            type="number"
                                            value={maxSeqLength}
                                            onChange={(e) => setMaxSeqLength(Number(e.target.value))}
                                        />
                                        <Input
                                            label="LoRA R"
                                            type="number"
                                            value={loraR}
                                            onChange={(e) => setLoraR(Number(e.target.value))}
                                        />
                                        <Input
                                            label="LoRA Alpha"
                                            type="number"
                                            value={loraAlpha}
                                            onChange={(e) => setLoraAlpha(Number(e.target.value))}
                                        />
                                    </>
                                )}
                            </div>

                            {userMode === 'user' && (
                                <p className="text-xs text-gray-500 mt-4 text-center">
                                    Switch to Power User mode for advanced hyperparameters (LR, LoRA, etc.)
                                </p>
                            )}
                        </Card>

                        <div className="pt-4">
                            <Button
                                variant="primary"
                                size="lg"
                                className="w-full"
                                onClick={handleStartTraining}
                                disabled={isTraining || !selectedModel || (!selectedDataset && !localDatasetPath)}
                                leftIcon={<Play size={20} />}
                                style={{ width: '100%' }}
                            >
                                Start Fine-Tuning
                            </Button>
                        </div>
                    </div>
                </div>
            )}

            {activeTab === 'training' && (
                <div className="h-[70vh] flex flex-col gap-4" style={{ height: '70vh', display: 'flex', flexDirection: 'column', gap: '16px' }}>
                    <div className="flex justify-between items-center p-4 rounded-lg border" style={{
                        background: 'linear-gradient(135deg, rgba(28, 28, 36, 0.6) 0%, rgba(30, 28, 38, 0.6) 100%)',
                        padding: '16px',
                        borderRadius: '12px',
                        border: '1px solid rgba(167, 139, 250, 0.2)'
                    }}>
                        <div className="flex items-center gap-3">
                            <div
                                className={`w-3 h-3 rounded-full ${isTraining ? 'animate-pulse' : ''}`}
                                style={{
                                    width: '12px',
                                    height: '12px',
                                    borderRadius: '50%',
                                    background: isTraining ? '#10b981' : '#71717a',
                                    boxShadow: isTraining ? '0 0 10px #10b981' : 'none'
                                }}
                            />
                            <span className="font-medium" style={{ color: isTraining ? '#10b981' : '#a1a1aa' }}>
                                {trainingStatus === 'initializing' ? 'Initializing...' : (isTraining ? 'Training in progress...' : 'Training stopped')}
                            </span>
                        </div>
                        {isTraining && (
                            <Button variant="danger" size="sm" onClick={handleStopTraining} leftIcon={<Square size={14} />}>
                                Stop Training
                            </Button>
                        )}
                    </div>

                    {tensorboardUrl && trainingStatus === 'training' ? (
                        <iframe
                            src={tensorboardUrl}
                            className="flex-1 w-full rounded-lg border"
                            style={{
                                flex: 1,
                                width: '100%',
                                borderRadius: '12px',
                                border: '1px solid rgba(167, 139, 250, 0.3)',
                                background: '#1a1a1f',
                                minHeight: '500px'
                            }}
                        />
                    ) : (
                        <div
                            className="flex-1 flex items-center justify-center border border-dashed rounded-lg"
                            style={{
                                borderColor: 'rgba(167, 139, 250, 0.2)',
                                borderRadius: '12px',
                                background: 'linear-gradient(135deg, rgba(28, 28, 36, 0.3) 0%, rgba(30, 28, 38, 0.3) 100%)'
                            }}
                        >
                            <div className="text-center">
                                {trainingStatus === 'initializing' ? (
                                    <>
                                        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-accent-primary mx-auto mb-4"></div>
                                        <p className="text-lg font-medium text-white mb-2">{initMessage || 'Preparing...'}</p>
                                        <p className="text-sm text-gray-400">Please wait while we set everything up</p>
                                    </>
                                ) : (
                                    <>
                                        <Terminal size={48} className="mx-auto mb-4 opacity-30" style={{ opacity: 0.3, color: '#a1a1aa' }} />
                                        <p style={{ color: '#71717a', fontSize: '0.95rem' }}>
                                            {isTraining ? 'Waiting for training metrics...' : 'Start training to view metrics'}
                                        </p>
                                        <p style={{ color: '#52525b', fontSize: '0.85rem', marginTop: '8px' }}>
                                            TensorBoard will appear here once training begins
                                        </p>
                                    </>
                                )}
                            </div>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

export default FineTuningPage;
