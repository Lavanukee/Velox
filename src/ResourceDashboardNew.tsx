import React, { useState, useEffect, useMemo } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { open } from '@tauri-apps/plugin-dialog';
import { useApp } from './context/AppContext';
import { DownloadTask } from './types';
import {
    Download,
    Trash2,
    Search,
    Package,
    Database,
    Brain,
    Layers,
    ChevronDown,
    ChevronRight,
    FolderInput,
    FolderOutput,
    X,
    Loader2,
    FileCode,
    Check,
    Square,
    RefreshCw

} from 'lucide-react';
import './styles/ResourceDashboard.css';

// --- Types ---

interface Resource {
    name: string;
    size: string;
    path: string;
    type: 'model' | 'gguf' | 'lora' | 'dataset';
    quantization?: string;
    is_mmproj?: boolean;
    has_vision?: boolean;
    is_processed?: boolean; // New: Checks if processed_data folder exists
    dataset_format?: string; // New: Detected format (arrow, csv, json, parquet)
}

interface CheckpointInfo {
    name: string;
    path: string;
    is_final: boolean;
    step_number: number | null;
}

interface ProjectLoraInfo {
    project_name: string;
    checkpoints: CheckpointInfo[];
}

interface HFSearchResult {
    id: string;
    name: string;
    downloads: number;
    likes: number;
    tags?: string[];
}

interface HFFile {
    path: string;
    size: number | null;
    lfs: any;
    // Updated file_type to include dataset-specific types
    file_type: 'gguf' | 'mmproj' | 'weight' | 'config' | 'dataset_file' | 'info' | 'other';
    quantization?: string;
    is_mmproj: boolean;
}

interface Props {
    addLogMessage: (message: string) => void;
    addNotification: (message: string, type?: 'success' | 'error' | 'info') => void;
    setDownloadTasks: React.Dispatch<React.SetStateAction<DownloadTask[]>>;
}

// --- Helpers ---

const formatBytes = (bytes: number | null): string => {
    if (bytes === null) return 'N/A';
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const dm = 2;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
};

// --- Component ---

const ResourceDashboard: React.FC<Props> = ({ addLogMessage, addNotification, setDownloadTasks }) => {
    const {
        rdHfQuery: hfQuery, setRdHfQuery: setHfQuery,
        rdHfType: hfType, setRdHfType: setHfType,
        rdSelectedPaths: selectedPaths, setRdSelectedPaths: setSelectedPaths
    } = useApp();

    // --- State: Resources & Dashboard ---
    const [resources, setResources] = useState<Resource[]>([]);
    const [projectLoras, setProjectLoras] = useState<ProjectLoraInfo[]>([]);
    const [expandedProjects, setExpandedProjects] = useState<Set<string>>(new Set());
    const [convertingMap, setConvertingMap] = useState<Record<string, boolean>>({}); // Track conversion loading state

    const [sections, setSections] = useState({
        models: true,
        ggufs: true,
        hfWeights: true,
        loras: false,
        datasets: true,
    });

    // --- State: Find New / HF Search ---
    const [showFindNew, setShowFindNew] = useState(false);
    const [lastSearchedQuery, setLastSearchedQuery] = useState(''); // For debounce check
    const [hfResults, setHfResults] = useState<HFSearchResult[]>([]);
    const [isSearching, setIsSearching] = useState(false);
    const [hfToken, setHfToken] = useState('');

    // --- State: File Browser & Selection ---
    const [selectedRepo, setSelectedRepo] = useState<string | null>(null);
    const [repoFiles, setRepoFiles] = useState<HFFile[]>([]);
    const [repoLoading, setRepoLoading] = useState(false);

    // Granular selection states
    const [selectedGGUFs, setSelectedGGUFs] = useState<Set<string>>(new Set());
    const [selectedMMProjs, setSelectedMMProjs] = useState<Set<string>>(new Set());
    const [selectedWeights, setSelectedWeights] = useState<Set<string>>(new Set());
    const [selectedDatasetFiles, setSelectedDatasetFiles] = useState<Set<string>>(new Set());

    // --- Initialization ---
    useEffect(() => {
        loadResources();
        loadToken();

        const unlistenModel = listen('model_downloaded', () => loadResources());
        const unlistenDataset = listen('dataset_downloaded', () => loadResources());

        return () => {
            unlistenModel.then(f => f());
            unlistenDataset.then(f => f());
        };
    }, []);

    // --- Debounced Search Effect ---
    useEffect(() => {
        if (!showFindNew || hfQuery.trim() === '' || hfQuery === lastSearchedQuery) return;

        const timer = setTimeout(() => {
            performSearch();
        }, 100); // 0.1s debounce

        return () => clearTimeout(timer);
    }, [hfQuery, showFindNew]);

    const loadResources = async () => {
        try {
            const raw: Resource[] = await invoke('list_all_resources_command');
            setResources(raw);

            // Load project-based LoRAs
            const projects: ProjectLoraInfo[] = await invoke('list_loras_by_project_command');
            setProjectLoras(projects);
        } catch (e) {
            addLogMessage(`Error loading resources: ${e}`);
        }
    };

    const loadToken = async () => {
        try {
            const t: string = await invoke('get_hf_token_command');
            if (t) setHfToken(t);
        } catch { }
    };

    // --- Memoized File Lists ---
    const { ggufFiles, mmprojFiles, weightFiles, datasetFiles } = useMemo(() => {
        const gguf: HFFile[] = [];
        const mmproj: HFFile[] = [];
        const weight: HFFile[] = [];
        const data: HFFile[] = [];

        repoFiles.forEach(f => {
            if (f.file_type === 'gguf' && !f.is_mmproj) gguf.push(f);
            else if ((f.file_type === 'gguf' && f.is_mmproj) || f.file_type === 'mmproj') mmproj.push(f);
            else if (f.file_type === 'weight' || f.file_type === 'config') weight.push(f);
            else if (f.file_type === 'dataset_file') data.push(f);
        });

        return { ggufFiles: gguf, mmprojFiles: mmproj, weightFiles: weight, datasetFiles: data };
    }, [repoFiles]);

    // --- Validation Logic (canDownload) ---
    const canDownload = useMemo(() => {
        if (!selectedRepo) return false;

        if (hfType === 'dataset') {
            return selectedDatasetFiles.size > 0;
        }

        const totalSelected = selectedGGUFs.size + selectedMMProjs.size + selectedWeights.size;
        if (totalSelected === 0) return false;

        return true;
    }, [hfType, selectedRepo, selectedGGUFs, selectedMMProjs, selectedWeights, selectedDatasetFiles]);

    // --- Smart Default Selection Logic ---
    const applySmartDefaults = (files: HFFile[]) => {
        if (hfType === 'dataset') {
            applySmartDefaultsDataset(files);
        } else {
            applySmartDefaultsModel(files);
        }
    };

    const applySmartDefaultsDataset = (files: HFFile[]) => {
        const datasetFiles = files.filter(f => f.file_type === 'dataset_file');
        const newSelectedDatasetFiles = new Set<string>();

        if (datasetFiles.length === 0) return;

        // Truncate if over 100 files to prevent crashes
        const filesToProcess = datasetFiles.length > 100 ? datasetFiles.slice(0, 100) : datasetFiles;

        // Priority 1: Select ALL Parquet files (gold standard for HF)
        const parquetFiles = filesToProcess.filter(f => f.path.toLowerCase().endsWith('.parquet'));
        if (parquetFiles.length > 0) {
            parquetFiles.forEach(f => newSelectedDatasetFiles.add(f.path));
            setSelectedDatasetFiles(newSelectedDatasetFiles);
            return;
        }

        // Priority 2: If no Parquet, select Arrow/JSONL/CSV files, but skip loose image files
        const dataFormatFiles = filesToProcess.filter(f => {
            const lower = f.path.toLowerCase();
            // Include data formats
            if (lower.endsWith('.arrow') || lower.endsWith('.jsonl') || lower.endsWith('.csv')) {
                return true;
            }
            // Also include archives that might contain datasets
            if (lower.endsWith('.zip') || lower.endsWith('.tar') || lower.endsWith('.tar.gz')) {
                return true;
            }
            // Exclude loose image files
            if (lower.endsWith('.png') || lower.endsWith('.jpg') || lower.endsWith('.jpeg') ||
                lower.endsWith('.gif') || lower.endsWith('.webp') || lower.endsWith('.bmp')) {
                return false;
            }
            return false;
        });

        if (dataFormatFiles.length > 0) {
            dataFormatFiles.forEach(f => newSelectedDatasetFiles.add(f.path));
        }

        setSelectedDatasetFiles(newSelectedDatasetFiles);
    };

    const applySmartDefaultsModel = (files: HFFile[]) => {
        if (hfType === 'dataset') return; // Do not auto-select for datasets (usually too large)

        const newSelectedGGUF = new Set<string>();
        const newSelectedMMProj = new Set<string>();
        const newSelectedWeights = new Set<string>();

        // 1. GGUF Selection Logic
        const ggufs = files.filter(f => f.file_type === 'gguf' && !f.is_mmproj);
        if (ggufs.length > 0) {
            // Ranking Weights
            const rankQuant = (q?: string) => {
                if (!q) return 0;
                const upper = q.toUpperCase();
                if (upper.includes("Q8")) return 100;
                if (upper.includes("Q6")) return 90;
                if (upper.includes("Q5")) return 80;
                if (upper.includes("Q4")) return 70;
                if (upper.includes("FP16") || upper.includes("F16")) return 20; // Fallback if no Qs
                if (upper.includes("BF16")) return 20;
                return 10;
            };

            // Find best
            const sortedGGUFs = [...ggufs].sort((a, b) => rankQuant(b.quantization) - rankQuant(a.quantization));

            // Logic: Pick Q8 or highest below Q8. 
            // If only FP16/BF16 exist, pick them.
            // Our sort puts best at top.
            if (sortedGGUFs.length > 0) {
                newSelectedGGUF.add(sortedGGUFs[0].path);
            }
        }

        // 2. Vision (MMProj) Selection Logic
        const mmprojs = files.filter(f => (f.file_type === 'gguf' && f.is_mmproj) || f.file_type === 'mmproj');
        if (mmprojs.length > 0) {
            const rankVision = (f: HFFile) => {
                const name = f.path.toUpperCase();
                if (name.includes("FP16")) return 100; // Favor FP16
                if (name.includes("FP32")) return 90;
                if (name.includes("Q8")) return 80;
                return 50;
            };
            const sortedVision = [...mmprojs].sort((a, b) => rankVision(b) - rankVision(a));
            if (sortedVision.length > 0) {
                newSelectedMMProj.add(sortedVision[0].path);
            }
        }

        // 3. HF Weights Selection Logic (Safetensors + Configs)
        const weights = files.filter(f => f.file_type === 'weight' || f.file_type === 'config');
        if (weights.length > 0) {
            // Select all relevant files for a full model download
            // Typically: .safetensors, config.json, tokenizer.json, vocab.json/txt
            weights.forEach(w => {
                const name = w.path.toLowerCase();
                if (name.endsWith(".safetensors") ||
                    name.includes("config") ||
                    name.includes("tokenizer") ||
                    name.includes("vocab")) {
                    newSelectedWeights.add(w.path);
                }
            });
        }

        // Apply selections
        setSelectedGGUFs(newSelectedGGUF);
        setSelectedMMProjs(newSelectedMMProj);
        setSelectedWeights(newSelectedWeights);
    };

    // --- Dashboard Actions ---
    const toggleSelection = (path: string) => {
        const next = new Set(selectedPaths);
        if (next.has(path)) next.delete(path);
        else next.add(path);
        setSelectedPaths(next);
    };

    const toggleSection = (key: keyof typeof sections) => {
        setSections(prev => ({ ...prev, [key]: !prev[key] }));
    };

    const handleImport = async (type: 'model' | 'lora' | 'dataset') => {
        try {
            const selected = await open({
                multiple: false,
                directory: type === 'model' || type === 'dataset',
            });
            if (selected) {
                await invoke('import_resource_command', {
                    resourceType: type === 'model' ? 'gguf' : type,
                    sourcePath: selected
                });
                addNotification(`Imported ${type} successfully`, 'success');
                loadResources();
            }
        } catch (e) {
            addNotification(`Import failed: ${e}`, 'error');
        }
    };

    const handleExport = async () => {
        if (selectedPaths.size === 0) return;
        try {
            const targetDir = await open({ directory: true, multiple: false });
            if (targetDir) {
                await invoke('export_resources_command', {
                    resourcePaths: Array.from(selectedPaths),
                    destination: targetDir
                });
                addNotification(`Exported ${selectedPaths.size} resources`, 'success');
                setSelectedPaths(new Set());
            }
        } catch (e) {
            addNotification(`Export failed: ${e}`, 'error');
        }
    };

    const handleDelete = async (r: Resource) => {
        try {
            await invoke('delete_resource_command', { resourceType: r.type, resourcePath: r.path });
            loadResources();
            addNotification('Deleted', 'success');
        } catch (e) {
            addNotification(`Delete failed: ${e}`, 'error');
        }
    };

    const handleDeleteCheckpoint = async (checkpointPath: string) => {
        try {
            await invoke('delete_resource_command', {
                resourceType: 'lora',
                resourcePath: checkpointPath
            });
            loadResources();
            addNotification('Checkpoint deleted', 'success');
        } catch (e) {
            addNotification(`Delete failed: ${e}`, 'error');
        }
    };

    // --- Dataset Conversion ---
    const handleConvertDataset = async (r: Resource) => {
        if (convertingMap[r.path]) return;

        setConvertingMap(prev => ({ ...prev, [r.path]: true }));
        addNotification('Starting dataset conversion...', 'info');

        try {
            await invoke('convert_dataset_command', {
                sourcePath: r.path,
                destinationPath: r.path // Output to same folder (in /processed_data subfolder)
            });

            addNotification('Dataset converted successfully!', 'success');
            loadResources(); // Refresh to potentially show new status
        } catch (e) {
            addNotification(`Conversion failed: ${e}`, 'error');
            addLogMessage(`Conversion Error: ${e}`);
        } finally {
            setConvertingMap(prev => ({ ...prev, [r.path]: false }));
        }
    };

    // --- Search & HF Actions ---
    const performSearch = async () => {
        if (!hfQuery) return;
        setIsSearching(true);
        setLastSearchedQuery(hfQuery);
        try {
            const res: HFSearchResult[] = await invoke('search_huggingface_command', { query: hfQuery, resourceType: hfType });
            setHfResults(res);
        } catch {
            addNotification('Search failed', 'error');
        } finally {
            setIsSearching(false);
        }
    };

    const selectRepo = async (id: string) => {
        setSelectedRepo(id);
        setRepoLoading(true);
        // Reset selections
        setSelectedGGUFs(new Set());
        setSelectedMMProjs(new Set());
        setSelectedWeights(new Set());
        setSelectedDatasetFiles(new Set());
        setRepoFiles([]);

        try {
            const files: HFFile[] = await invoke('list_hf_repo_files_command', {
                repoId: id,
                token: hfToken || null,
                resourceType: hfType
            });
            setRepoFiles(files);

            // Apply Smart Defaults
            applySmartDefaults(files);
        } catch (e) {
            addNotification(`Failed to list files: ${e}`, 'error');
            console.error("Error listing HF repo files:", e);
        } finally {
            setRepoLoading(false);
        }
    };

    const toggleFileSelection = (file: HFFile) => {
        const path = file.path;
        let setFunc: React.Dispatch<React.SetStateAction<Set<string>>> | undefined;
        let currentSet: Set<string> | undefined;

        if (hfType === 'dataset') {
            setFunc = setSelectedDatasetFiles;
            currentSet = selectedDatasetFiles;
        } else {
            if (file.file_type === 'gguf' && !file.is_mmproj) {
                setFunc = setSelectedGGUFs;
                currentSet = selectedGGUFs;
            } else if (file.is_mmproj || file.file_type === 'mmproj') {
                setFunc = setSelectedMMProjs;
                currentSet = selectedMMProjs;
            } else if (file.file_type === 'weight' || file.file_type === 'config') {
                setFunc = setSelectedWeights;
                currentSet = selectedWeights;
            }
        }

        if (setFunc && currentSet) {
            const next = new Set(currentSet);
            if (next.has(path)) next.delete(path);
            else next.add(path);
            setFunc(next);
        }
    };

    const startDownload = async () => {
        if (!selectedRepo) return;

        if (!canDownload) {
            addNotification('Invalid selection for download.', 'error');
            return;
        }

        const taskId = `dl_${Date.now()}`;
        let filesToDownload: string[] = [];

        if (hfType === 'model') {
            filesToDownload = [
                ...Array.from(selectedGGUFs),
                ...Array.from(selectedMMProjs),
                ...Array.from(selectedWeights)
            ];
        } else {
            filesToDownload = Array.from(selectedDatasetFiles);
        }

        if (filesToDownload.length === 0) {
            addNotification('No files selected.', 'error');
            return;
        }

        const handleCancel = async () => {
            try {
                await invoke('cancel_download_command', { taskId: taskId });
                addLogMessage(`Download cancelled: ${taskId}`);
            } catch (e) {
                addLogMessage(`Failed to cancel download: ${e}`);
            }
        };

        setDownloadTasks(prev => [...prev, {
            id: taskId,
            name: selectedRepo,
            progress: 0,
            status: 'downloading',
            type: hfType,
            onCancel: handleCancel
        }]);
        setShowFindNew(false);

        try {
            if (hfType === 'model') {
                await invoke('download_hf_model_command', {
                    modelId: selectedRepo,
                    files: filesToDownload,
                    token: hfToken || null,
                    taskId: taskId
                });
            } else {
                await invoke('download_hf_dataset_command', {
                    datasetId: selectedRepo,
                    files: filesToDownload,
                    token: hfToken || null,
                    taskId: taskId
                });
            }

            addNotification('Download started', 'success');
            loadResources();
        } catch (e) {
            addLogMessage(`Download error: ${e}`);
            addNotification(`Download failed: ${e}`, 'error');
        } finally {
            // Reset
            setSelectedGGUFs(new Set());
            setSelectedMMProjs(new Set());
            setSelectedWeights(new Set());
            setSelectedDatasetFiles(new Set());
            setSelectedRepo(null);
            setRepoFiles([]);
        }
    };

    // --- Memoized Lists for Dashboard ---
    const { localModels, datasets } = useMemo(() => ({
        localModels: resources.filter(r => r.type === 'model' || r.type === 'gguf'),
        datasets: resources.filter(r => r.type === 'dataset'),
    }), [resources]);

    // --- Render Helpers ---
    const renderResourceRow = (r: Resource) => (
        <div key={r.path} className="rd-item">
            <div
                className={`rd-item-select ${selectedPaths.has(r.path) ? 'selected' : ''}`}
                onClick={() => toggleSelection(r.path)}
            >
                {selectedPaths.has(r.path) ? <Check size={20} /> : <Square size={20} />}
            </div>

            <div className="rd-item-icon">
                {r.type === 'gguf' ? <Package size={22} /> :
                    r.type === 'lora' ? <Layers size={22} /> :
                        r.type === 'dataset' ? <Database size={22} /> : <Brain size={22} />}
            </div>

            <div className="rd-item-details">
                <div className="rd-item-name" title={r.name}>{r.name}</div>
                <div className="rd-item-meta">
                    <span>{r.size}</span>
                    {r.quantization && <span className="rd-badge blue">{r.quantization}</span>}
                    {r.is_mmproj && <span className="rd-badge purple">Vision</span>}

                    {/* Dataset Specific Badges & Actions */}
                    {r.type === 'dataset' && (
                        <>
                            {r.dataset_format && <span className="rd-badge blue">{r.dataset_format.toUpperCase()}</span>}
                            {r.is_processed ? (
                                <span className="rd-badge green flex items-center gap-1">
                                    <Check size={10} /> Ready for Use
                                </span>
                            ) : (
                                <span className="rd-badge gray">Raw</span>
                            )}
                        </>
                    )}
                </div>
            </div>

            {/* Dataset Convert Action */}
            {r.type === 'dataset' && !r.is_processed && (
                <button
                    className="rd-action-btn"
                    title="Convert to Arrow"
                    onClick={() => handleConvertDataset(r)}
                    disabled={convertingMap[r.path]}
                >
                    {convertingMap[r.path] ? <Loader2 size={18} className="animate-spin" /> : <RefreshCw size={18} />}
                </button>
            )}

            <button onClick={() => handleDelete(r)} className="rd-delete-btn" title="Delete">
                <Trash2 size={18} />
            </button>
        </div>
    );

    return (
        <div className="rd-container">
            {/* Top Bar */}
            <div className="rd-header">
                <div className="rd-title">
                    <h1>Library</h1>
                    <p>{resources.length} local resources</p>
                </div>

                <div className="rd-actions">
                    {selectedPaths.size > 0 && (
                        <button className="btn btn-secondary" onClick={handleExport}>
                            <FolderOutput size={16} style={{ marginRight: '8px' }} />
                            Export ({selectedPaths.size})
                        </button>
                    )}

                    <div className="rd-dropdown-container">
                        <button className="btn btn-secondary">
                            <FolderInput size={16} style={{ marginRight: '8px' }} />
                            Import
                        </button>
                        <div className="rd-dropdown-menu">
                            <div className="rd-dropdown-item" onClick={() => handleImport('model')}>Import Model</div>
                            <div className="rd-dropdown-item" onClick={() => handleImport('lora')}>Import LoRA</div>
                            <div className="rd-dropdown-item" onClick={() => handleImport('dataset')}>Import Dataset</div>
                        </div>
                    </div>

                    <button className="btn btn-primary" onClick={() => setShowFindNew(true)}>
                        <Search size={16} style={{ marginRight: '8px' }} />
                        Find New
                    </button>
                </div>
            </div>

            {/* Main Content */}
            <div className="rd-content">
                {/* Models Section */}
                <div className="rd-section">
                    <div className="rd-section-header" onClick={() => toggleSection('models')}>
                        {sections.models ? <ChevronDown size={18} /> : <ChevronRight size={18} />}
                        <span className="rd-section-title">Models</span>
                        <span className="rd-section-count">{localModels.length}</span>
                    </div>

                    {sections.models && (
                        <div className="rd-section-body">
                            <div className="rd-grid">
                                {localModels.map(renderResourceRow)}
                                {localModels.length === 0 && <span className="text-muted italic">No models found.</span>}
                            </div>
                        </div>
                    )}
                </div>

                {/* LoRAs Section - Project Based */}
                <div className="rd-section">
                    <div className="rd-section-header" onClick={() => toggleSection('loras')}>
                        {sections.loras ? <ChevronDown size={18} /> : <ChevronRight size={18} />}
                        <span className="rd-section-title">LoRAs / Adapters</span>
                        <span className="rd-section-count">{projectLoras.length} projects</span>
                    </div>
                    {sections.loras && (
                        <div className="rd-section-body">
                            {projectLoras.length === 0 && <span className="text-muted italic">No training projects found.</span>}

                            {projectLoras.map(project => (
                                <div key={project.project_name} className="rd-project-group" style={{ marginBottom: '16px' }}>
                                    {/* Project Header */}
                                    <div
                                        className="rd-project-header"
                                        onClick={() => {
                                            const next = new Set(expandedProjects);
                                            if (next.has(project.project_name)) {
                                                next.delete(project.project_name);
                                            } else {
                                                next.add(project.project_name);
                                            }
                                            setExpandedProjects(next);
                                        }}
                                        style={{
                                            padding: '12px 16px',
                                            background: 'rgba(255,255,255,0.03)',
                                            borderRadius: '8px',
                                            cursor: 'pointer',
                                            display: 'flex',
                                            alignItems: 'center',
                                            gap: '12px',
                                            marginBottom: '8px'
                                        }}
                                    >
                                        {expandedProjects.has(project.project_name) ?
                                            <ChevronDown size={16} /> :
                                            <ChevronRight size={16} />
                                        }
                                        <Layers size={18} className="text-purple-400" />
                                        <span style={{ fontWeight: '600' }}>{project.project_name}</span>
                                        <span style={{
                                            marginLeft: 'auto',
                                            fontSize: '0.85rem',
                                            color: 'rgba(255,255,255,0.5)'
                                        }}>
                                            {project.checkpoints.length} checkpoint{project.checkpoints.length !== 1 ? 's' : ''}
                                        </span>
                                    </div>

                                    {/* Checkpoints */}
                                    {expandedProjects.has(project.project_name) && (
                                        <div className="rd-grid" style={{ marginLeft: '32px' }}>
                                            {project.checkpoints.map(checkpoint => (
                                                <div key={checkpoint.path} className="rd-item">
                                                    <div
                                                        className={`rd-item-select ${selectedPaths.has(checkpoint.path) ? 'selected' : ''}`}
                                                        onClick={() => toggleSelection(checkpoint.path)}
                                                    >
                                                        {selectedPaths.has(checkpoint.path) ? <Check size={20} /> : <Square size={20} />}
                                                    </div>

                                                    <div className="rd-item-icon">
                                                        <Layers size={22} />
                                                    </div>

                                                    <div className="rd-item-details">
                                                        <div className="rd-item-name" title={checkpoint.name}>
                                                            {checkpoint.name}
                                                        </div>
                                                        <div className="rd-item-meta">
                                                            {checkpoint.is_final && (
                                                                <span className="rd-badge green">Final Model</span>
                                                            )}
                                                            {checkpoint.step_number !== null && (
                                                                <span className="rd-badge blue">Step {checkpoint.step_number}</span>
                                                            )}
                                                        </div>
                                                    </div>

                                                    <button
                                                        onClick={() => handleDeleteCheckpoint(checkpoint.path)}
                                                        className="rd-delete-btn"
                                                        title="Delete"
                                                    >
                                                        <Trash2 size={18} />
                                                    </button>
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            ))}
                        </div>
                    )}
                </div>

                {/* Datasets Section */}
                <div className="rd-section">
                    <div className="rd-section-header" onClick={() => toggleSection('datasets')}>
                        {sections.datasets ? <ChevronDown size={18} /> : <ChevronRight size={18} />}
                        <span className="rd-section-title">Datasets</span>
                        <span className="rd-section-count">{datasets.length}</span>
                    </div>
                    {sections.datasets && (
                        <div className="rd-section-body">
                            <div className="rd-grid">
                                {datasets.map(renderResourceRow)}
                                {datasets.length === 0 && <span className="text-muted italic">No datasets found.</span>}
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {/* --- Find New Overlay --- */}
            {showFindNew && (
                <div className="fn-overlay">
                    {/* Header */}
                    <div className="fn-header">
                        <button className="fn-close-btn" onClick={() => setShowFindNew(false)}>
                            <X size={24} />
                        </button>
                        <div className="fn-search-bar">
                            <div className="fn-toggle">
                                <div
                                    className={`fn-toggle-opt ${hfType === 'model' ? 'active' : ''}`}
                                    onClick={() => { setHfType('model'); setHfResults([]); setHfQuery(''); }}
                                >
                                    Models
                                </div>
                                <div
                                    className={`fn-toggle-opt ${hfType === 'dataset' ? 'active' : ''}`}
                                    onClick={() => { setHfType('dataset'); setHfResults([]); setHfQuery(''); }}
                                >
                                    Datasets
                                </div>
                            </div>
                            <input
                                className="fn-input"
                                placeholder={`Search HuggingFace ${hfType}s...`}
                                value={hfQuery}
                                onChange={e => setHfQuery(e.target.value)}
                                // Removed onKeyDown Enter since we have debounce
                                autoFocus
                            />
                            <button className="btn btn-icon">
                                {isSearching ? <Loader2 size={18} className="animate-spin" /> : <Search size={18} />}
                            </button>
                        </div>
                    </div>

                    <div className="fn-main">
                        {/* Search Results */}
                        <div className="fn-results-list">
                            {hfResults.length === 0 && !isSearching && (
                                <div className="text-center text-muted" style={{ marginTop: '4rem' }}>
                                    <Search size={48} style={{ opacity: 0.2, margin: '0 auto 1rem' }} />
                                    <p>Start typing to search...</p>
                                </div>
                            )}
                            {hfResults.map(res => (
                                <div
                                    key={res.id}
                                    className={`fn-result-item ${selectedRepo === res.id ? 'selected' : ''}`}
                                    onClick={() => selectRepo(res.id)}
                                >
                                    <div className="fn-result-title">{res.name}</div>
                                    <div className="fn-result-stats">
                                        <span>⬇ {res.downloads.toLocaleString()}</span>
                                        <span>♥ {res.likes.toLocaleString()}</span>
                                    </div>
                                    <div className="fn-tags">
                                        {res.tags?.slice(0, 5).map(tag => (
                                            <span key={tag} className="rd-badge gray">{tag}</span>
                                        ))}
                                    </div>
                                </div>
                            ))}
                        </div>

                        {/* File Browser Pane */}
                        {selectedRepo && (
                            <div className="fn-detail-pane">
                                <div className="fn-detail-header">
                                    <h3>{selectedRepo}</h3>
                                    <button className="btn btn-ghost" onClick={() => setSelectedRepo(null)}><X size={18} /></button>
                                </div>

                                <div className="fn-file-scroller">
                                    {repoLoading ? (
                                        <div style={{ display: 'flex', justifyContent: 'center', padding: '2rem', color: 'var(--accent-primary)' }}>
                                            <Loader2 size={32} className="animate-spin" />
                                        </div>
                                    ) : (
                                        <>
                                            {/* Dataset File Loop */}
                                            {hfType === 'dataset' && datasetFiles.length > 0 && (
                                                <>
                                                    {/* Selected Files at Top */}
                                                    {selectedDatasetFiles.size > 0 && (
                                                        <>
                                                            <div className="fn-selected-header" style={{
                                                                background: 'linear-gradient(135deg, rgba(167, 139, 250, 0.2) 0%, rgba(125, 211, 252, 0.1) 100%)',
                                                                border: '1px solid rgba(167, 139, 250, 0.4)',
                                                                padding: '12px 16px',
                                                                margin: '0 0 12px 0',
                                                                borderRadius: '8px',
                                                                fontWeight: '600',
                                                                color: '#c4b5fd',
                                                                display: 'flex',
                                                                alignItems: 'center',
                                                                gap: '8px'
                                                            }}>
                                                                <Check size={16} style={{ color: '#4caf50' }} />
                                                                Selected Files ({selectedDatasetFiles.size})
                                                            </div>
                                                            {datasetFiles.filter(f => selectedDatasetFiles.has(f.path)).map(file => {
                                                                const fileFormat = file.path.split('.').pop()?.toUpperCase() || 'DATA';
                                                                return (
                                                                    <div key={file.path} className="fn-file selected" onClick={() => toggleFileSelection(file)}>
                                                                        <div className="fn-file-checkbox">
                                                                            <Check size={12} />
                                                                        </div>
                                                                        <Database size={16} className="text-muted" />
                                                                        <div className="fn-file-info">
                                                                            <div className="fn-file-path">{file.path}</div>
                                                                            <div className="fn-file-size">
                                                                                {formatBytes(file.size)}
                                                                                <span className="rd-badge blue ml-2">{fileFormat}</span>
                                                                            </div>
                                                                        </div>
                                                                    </div>
                                                                );
                                                            })}
                                                        </>
                                                    )}

                                                    {/* Unselected Files */}
                                                    {datasetFiles.filter(f => !selectedDatasetFiles.has(f.path)).length > 0 && (
                                                        <>
                                                            <div className="fn-other-files-header">Other Files</div>
                                                            {datasetFiles.filter(f => !selectedDatasetFiles.has(f.path)).map(file => {
                                                                const fileFormat = file.path.split('.').pop()?.toUpperCase() || 'DATA';
                                                                return (
                                                                    <div key={file.path} className="fn-file" onClick={() => toggleFileSelection(file)}>
                                                                        <div className="fn-file-checkbox" />
                                                                        <Database size={16} className="text-muted" />
                                                                        <div className="fn-file-info">
                                                                            <div className="fn-file-path">{file.path}</div>
                                                                            <div className="fn-file-size">
                                                                                {formatBytes(file.size)}
                                                                                <span className="rd-badge blue ml-2">{fileFormat}</span>
                                                                            </div>
                                                                        </div>
                                                                    </div>
                                                                );
                                                            })}
                                                        </>
                                                    )}
                                                </>
                                            )}

                                            {/* Model File Loops */}
                                            {hfType === 'model' && (
                                                <>
                                                    {/* Selected Models at Top */}
                                                    {(selectedGGUFs.size > 0 || selectedMMProjs.size > 0 || selectedWeights.size > 0) && (
                                                        <>
                                                            <div className="fn-selected-header" style={{
                                                                background: 'linear-gradient(135deg, rgba(167, 139, 250, 0.2) 0%, rgba(125, 211, 252, 0.1) 100%)',
                                                                border: '1px solid rgba(167, 139, 250, 0.4)',
                                                                padding: '12px 16px',
                                                                margin: '0 0 12px 0',
                                                                borderRadius: '8px',
                                                                fontWeight: '600',
                                                                color: '#c4b5fd',
                                                                display: 'flex',
                                                                alignItems: 'center',
                                                                gap: '8px'
                                                            }}>
                                                                <Check size={16} style={{ color: '#4caf50' }} />
                                                                Selected Files ({selectedGGUFs.size + selectedMMProjs.size + selectedWeights.size})
                                                            </div>

                                                            {/* Selected GGUFs */}
                                                            {Array.from(selectedGGUFs).map(path => {
                                                                const file = ggufFiles.find(f => f.path === path);
                                                                if (!file) return null;
                                                                return (
                                                                    <div key={file.path} className="fn-file selected" onClick={() => toggleFileSelection(file)}>
                                                                        <div className="fn-file-checkbox"><Check size={12} /></div>
                                                                        <Package size={16} className="text-muted" />
                                                                        <div className="fn-file-info">
                                                                            <div className="fn-file-path">{file.path}</div>
                                                                            <div className="fn-file-size">
                                                                                {formatBytes(file.size)}
                                                                                {file.quantization && <span className="rd-badge blue ml-2">{file.quantization}</span>}
                                                                            </div>
                                                                        </div>
                                                                    </div>
                                                                );
                                                            })}

                                                            {/* Selected MMProjs */}
                                                            {Array.from(selectedMMProjs).map(path => {
                                                                const file = mmprojFiles.find(f => f.path === path);
                                                                if (!file) return null;
                                                                return (
                                                                    <div key={file.path} className="fn-file selected" onClick={() => toggleFileSelection(file)}>
                                                                        <div className="fn-file-checkbox"><Check size={12} /></div>
                                                                        <Brain size={16} className="text-muted" />
                                                                        <div className="fn-file-info">
                                                                            <div className="fn-file-path">{file.path}</div>
                                                                            <div className="fn-file-size">{formatBytes(file.size)}</div>
                                                                        </div>
                                                                    </div>
                                                                );
                                                            })}

                                                            {/* Selected Weights */}
                                                            {Array.from(selectedWeights).map(path => {
                                                                const file = weightFiles.find(f => f.path === path);
                                                                if (!file) return null;
                                                                return (
                                                                    <div key={file.path} className="fn-file selected" onClick={() => toggleFileSelection(file)}>
                                                                        <div className="fn-file-checkbox"><Check size={12} /></div>
                                                                        <FileCode size={16} className="text-muted" />
                                                                        <div className="fn-file-info">
                                                                            <div className="fn-file-path">{file.path}</div>
                                                                            <div className="fn-file-size">{formatBytes(file.size)}</div>
                                                                        </div>
                                                                    </div>
                                                                );
                                                            })}
                                                        </>
                                                    )}

                                                    {/* Unselected GGUFs */}
                                                    {ggufFiles.filter(f => !selectedGGUFs.has(f.path)).length > 0 && (
                                                        <>
                                                            <div className="fn-other-files-header">GGUF Quantizations</div>
                                                            {ggufFiles.filter(f => !selectedGGUFs.has(f.path)).map(file => {
                                                                return (
                                                                    <div key={file.path} className="fn-file" onClick={() => toggleFileSelection(file)}>
                                                                        <div className="fn-file-checkbox" />
                                                                        <Package size={16} className="text-muted" />
                                                                        <div className="fn-file-info">
                                                                            <div className="fn-file-path">{file.path}</div>
                                                                            <div className="fn-file-size">
                                                                                {formatBytes(file.size)}
                                                                                {file.quantization && <span className="rd-badge blue ml-2">{file.quantization}</span>}
                                                                            </div>
                                                                        </div>
                                                                    </div>
                                                                );
                                                            })}
                                                        </>
                                                    )}

                                                    {/* Unselected MMProjs */}
                                                    {mmprojFiles.filter(f => !selectedMMProjs.has(f.path)).length > 0 && (
                                                        <>
                                                            <div className="fn-other-files-header">Vision / Projectors</div>
                                                            {mmprojFiles.filter(f => !selectedMMProjs.has(f.path)).map(file => {
                                                                return (
                                                                    <div key={file.path} className="fn-file" onClick={() => toggleFileSelection(file)}>
                                                                        <div className="fn-file-checkbox" />
                                                                        <Brain size={16} className="text-muted" />
                                                                        <div className="fn-file-info">
                                                                            <div className="fn-file-path">{file.path}</div>
                                                                            <div className="fn-file-size">{formatBytes(file.size)}</div>
                                                                        </div>
                                                                    </div>
                                                                );
                                                            })}
                                                        </>
                                                    )}

                                                    {/* Unselected Weights/Configs */}
                                                    {weightFiles.filter(f => !selectedWeights.has(f.path)).length > 0 && (
                                                        <>
                                                            <div className="fn-other-files-header">Weights & Configs</div>
                                                            {weightFiles.filter(f => !selectedWeights.has(f.path)).map(file => {
                                                                return (
                                                                    <div key={file.path} className="fn-file" onClick={() => toggleFileSelection(file)}>
                                                                        <div className="fn-file-checkbox" />
                                                                        <FileCode size={16} className="text-muted" />
                                                                        <div className="fn-file-info">
                                                                            <div className="fn-file-path">{file.path}</div>
                                                                            <div className="fn-file-size">{formatBytes(file.size)}</div>
                                                                        </div>
                                                                    </div>
                                                                );
                                                            })}
                                                        </>
                                                    )}
                                                </>
                                            )}

                                            {/* Empty State */}
                                            {repoFiles.length === 0 && (
                                                <div className="text-muted text-center py-4">No files found.</div>
                                            )}
                                        </>
                                    )}
                                </div>

                                <div className="fn-actions">
                                    <button
                                        className="btn btn-primary"
                                        disabled={!canDownload}
                                        onClick={startDownload}
                                        style={{ width: '100%', marginTop: '1rem' }}
                                    >
                                        <Download size={16} style={{ marginRight: '8px' }} />
                                        {hfType === 'model'
                                            ? `Download Selected (${selectedGGUFs.size + selectedMMProjs.size + selectedWeights.size})`
                                            : `Download Selected (${selectedDatasetFiles.size})`
                                        }
                                    </button>

                                    {!hfToken && (
                                        <div className="fn-footer-note">Note: HF Token required for gated repos</div>
                                    )}
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
};

export default ResourceDashboard;