import React, { useState, useEffect, useMemo } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { open } from '@tauri-apps/plugin-dialog';
import { startDrag } from '@crabnebula/tauri-plugin-drag';
import { useApp } from './context/AppContext';
import { DownloadTask, Resource } from './types';
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
    Plus,
    Wrench,
    Sparkles,
    AlertTriangle,
    ArrowUpRight,
    SlidersHorizontal,
    Pencil
} from 'lucide-react';
import { Modal } from './components/Modal';
import { Button } from './components/Button';
import { Input } from './components/Input';
import './styles/ResourceDashboard.css';

// --- Types ---

interface CheckpointInfo {
    name: string;
    path: string;
    is_final: boolean;
    step_number: number | null;
    gguf_path?: string | null;
}

interface ProjectLoraInfo {
    project_name: string;
    checkpoints: CheckpointInfo[];
    base_model?: string;
}

interface HFSearchResult {
    id: string;
    name: string;
    author?: string;
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
    downloadTasks: DownloadTask[];
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

const formatCount = (num: number | null | undefined): string => {
    if (num === undefined || num === null) return '';
    if (num < 1000) return num.toString();
    if (num < 1000000) return (num / 1000).toFixed(1) + 'k';
    return (num / 1000000).toFixed(1) + 'M';
};

// --- Component ---

interface ResourceRowProps {
    resource: Resource;
    isSelected: boolean;
    toggleSelection: (path: string) => void;
    handleExpandResource: (r: Resource) => void;
    handleConvertDataset: (r: Resource) => void;
    handleFixDataset: (r: Resource) => void;
    handleDelete: (r: Resource) => void;
    handleConvertModel: (r: Resource) => void;
    handleConvertLora: (r: Resource) => void;
    handleRename: (r: Resource) => void;
    converting: boolean;
    fixing: boolean;
}

const ResourceRow: React.FC<ResourceRowProps> = ({
    resource: r,
    isSelected,
    toggleSelection,
    handleExpandResource,
    handleConvertDataset,
    handleFixDataset,
    handleDelete,
    handleConvertModel,
    handleConvertLora,
    handleRename,
    converting,
    fixing
}) => {
    const [absPath, setAbsPath] = useState<string>('');

    // Pre-resolve path on hover to ensure drag starts instantly
    const handleMouseEnter = () => {
        if (!absPath) {
            invoke<string>('resolve_path_command', { path: r.path })
                .then(p => setAbsPath(p))
                .catch(console.error);
        }
    };

    const handleDragStart = async (e: React.DragEvent) => {
        // Prevent default to allow Tauri to handle
        e.preventDefault();

        try {
            let targetPath = absPath;
            if (!targetPath) {
                targetPath = await invoke('resolve_path_command', { path: r.path });
            }

            if (!targetPath) {
                console.error("Failed to resolve path for drag:", r.path);
                return;
            }

            console.log("Starting drag for:", targetPath);
            await startDrag({
                item: [targetPath],
                // Pass valid string or allow plugin default if handling specific
                icon: '' // Empty string is standard for "default" in many plugins
            });
        } catch (err) {
            console.error('Drag failed:', err);
            // Verify if user notification is needed
        }
    };

    return (
        <div
            className="rd-item"
            onClick={() => handleExpandResource(r)}
            draggable={true}
            onDragStart={handleDragStart}
            onMouseEnter={handleMouseEnter}
            style={{ cursor: 'grab' }}
        >
            <div
                className={`rd-item-select ${isSelected ? 'selected' : ''}`}
                onClick={(e) => { e.stopPropagation(); toggleSelection(r.path); }}
            >
                {isSelected ? <Check size={18} /> : <Square size={18} />}
            </div>

            <div className="rd-item-icon">
                {r.type === 'gguf' ? <Package size={18} /> :
                    r.type === 'lora' ? <Layers size={18} /> :
                        r.type === 'dataset' ? <Database size={18} /> : <Brain size={18} />}
            </div>

            <div className="rd-item-details">
                <div className="rd-item-name" title={r.name}>{r.name}</div>
                <div className="rd-item-meta">
                    <span>{r.size}</span>
                    {r.quantization && <span className="rd-badge blue">{r.quantization}</span>}
                    {r.is_mmproj && <span className="rd-badge blue">Vision</span>}
                    {r.type === 'dataset' && (
                        <>
                            {r.dataset_format && <span className="rd-badge blue">{r.dataset_format.toUpperCase()}</span>}
                            {r.count !== undefined && <span className="rd-badge gray" title={`${r.count} examples`}>{formatCount(r.count)} examples</span>}
                            {r.modalities?.map(m => (
                                <span key={m} className={`rd-badge ${m === 'Vision' ? 'blue' : 'gray'}`}>{m}</span>
                            ))}
                            {r.is_processed && (
                                <span className="rd-badge green">
                                    <Check size={10} /> Ready
                                </span>
                            )}
                            {r.format_error && (
                                <span className="rd-badge red" title={r.format_error}>
                                    <AlertTriangle size={10} /> Error
                                </span>
                            )}
                        </>
                    )}
                </div>
            </div>

            {/* Actions */}
            <div className="flex items-center gap-2" onClick={e => e.stopPropagation()}>
                {r.type === 'dataset' && !r.is_processed && (
                    <>
                        <button
                            className="rd-action-btn warning"
                            title="Fix Dataset (Repair JSONL)"
                            onClick={() => handleFixDataset(r)}
                            disabled={fixing}
                        >
                            {fixing ? <Loader2 size={14} className="animate-spin" /> : <Wrench size={14} />}
                            <span>Repair</span>
                        </button>
                        <button
                            className="rd-action-btn primary"
                            title="Process/Convert Dataset"
                            onClick={() => handleConvertDataset(r)}
                            disabled={converting}
                        >
                            {converting ? <Loader2 size={14} className="animate-spin" /> : <Sparkles size={14} />}
                            <span>Process</span>
                        </button>
                    </>
                )}
                {(r.type === 'model' || r.type === 'lora') && (
                    <button
                        className="rd-action-btn primary"
                        title="Convert to GGUF (Unsloth -> fallback)"
                        onClick={(e) => {
                            e.stopPropagation();
                            if (r.type === 'lora') handleConvertLora(r);
                            else handleConvertModel(r);
                        }}
                        disabled={converting}
                    >
                        {converting ? <Loader2 size={14} className="animate-spin" /> : <Layers size={14} />}
                        <span>Convert</span>
                    </button>
                )}
                <button
                    className="rd-action-btn secondary"
                    title="Rename"
                    onClick={(e) => { e.stopPropagation(); handleRename(r); }}
                >
                    <Pencil size={14} />
                    <span>Rename</span>
                </button>
                <button
                    onClick={() => handleDelete(r)}
                    className="rd-delete-btn"
                    title="Delete"
                    style={{ background: 'none', border: 'none', color: 'var(--text-dim)', cursor: 'pointer', padding: '6px' }}
                >
                    <Trash2 size={16} />
                </button>
            </div>
        </div>
    );
};

// --- Component ---

const ResourceDashboard: React.FC<Props> = ({ addLogMessage, addNotification, setDownloadTasks, downloadTasks }) => {
    const {
        rdHfQuery: hfQuery, setRdHfQuery: setHfQuery,
        rdHfAuthor: hfAuthor, setRdHfAuthor: setAuthor,
        rdHfModalities: hfModalities, setRdHfModalities: setModalities,
        rdHfSizeRange: hfSizeRange, setRdHfSizeRange: setSizeRange,
        rdHfType: hfType, setRdHfType: setHfType,
        rdShowFindNew: showFindNew, setRdShowFindNew: setShowFindNew,
        rdSelectedPaths: selectedPaths, setRdSelectedPaths: setSelectedPaths,
        resources, loadResources, isConvertingMap, setConvertingMap
    } = useApp();

    // Local processing states
    const [fixingMap, setFixingMap] = useState<Record<string, boolean>>({});

    // Rename State
    const [renamingResource, setRenamingResource] = useState<Resource | null>(null);
    const [renameValue, setRenameValue] = useState('');

    // --- State: Dashboard & UI ---
    const [projectLoras, setProjectLoras] = useState<ProjectLoraInfo[]>([]);
    const [expandedProjects, setExpandedProjects] = useState<Set<string>>(new Set());

    const [sections, setSections] = useState({
        models: true,
        ggufs: true,
        hfWeights: true,
        loras: false,
        datasets: true,
    });

    // --- State: Find New / HF Search ---
    const [showFilters, setShowFilters] = useState(false);
    const [lastSearchedQuery, setLastSearchedQuery] = useState("");
    // For debounce check
    const [lastSearchedType, setLastSearchedType] = useState<'model' | 'dataset'>('model');
    const [hfResults, setHfResults] = useState<HFSearchResult[]>([]);
    const [isSearching, setIsSearching] = useState(false);
    const [isDownloading, setIsDownloading] = useState(false);
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

    const [isImportOpen, setIsImportOpen] = useState(false);
    const importRef = React.useRef<HTMLDivElement>(null);

    const [expandedResourceId, setExpandedResourceId] = useState<string | null>(null);
    const [expandedFiles, setExpandedFiles] = useState<Record<string, { local: Set<string>, remote: HFFile[] }>>({});
    const [loadingExpansion, setLoadingExpansion] = useState(false);

    const activeResource = useMemo(() => {
        if (!expandedResourceId) return null;
        return resources.find(r => r.path === expandedResourceId) || null;
    }, [expandedResourceId, resources]);

    const activeFiles = useMemo(() => {
        if (!expandedResourceId || !expandedFiles[expandedResourceId]) return [];
        const details = expandedFiles[expandedResourceId];
        const local = details.local;
        const remote = details.remote;

        const merged = remote.map(f => ({
            name: f.path,
            size: f.size,
            isLocal: local.has(f.path)
        }));

        local.forEach(f => {
            if (!remote.find(rf => rf.path === f)) {
                merged.push({ name: f, size: null, isLocal: true });
            }
        });

        return merged.sort((a, b) => a.name.localeCompare(b.name));
    }, [expandedResourceId, expandedFiles]);

    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (importRef.current && !importRef.current.contains(event.target as Node)) {
                setIsImportOpen(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    // --- Global Drag Prevention (Browser Default) ---
    useEffect(() => {
        const preventDefault = (e: DragEvent) => e.preventDefault();
        document.addEventListener('dragover', preventDefault);
        document.addEventListener('dragenter', preventDefault);
        document.addEventListener('dragleave', preventDefault);
        document.addEventListener('drop', preventDefault);
        return () => {
            document.removeEventListener('dragover', preventDefault);
            document.removeEventListener('dragenter', preventDefault);
            document.removeEventListener('dragleave', preventDefault);
            document.removeEventListener('drop', preventDefault);
        };
    }, []);

    // --- Drag and Drop Handler (Now handled by GlobalDropZone in App.tsx) ---
    // Global drop handling provides better UX with full-screen overlay
    // and intelligent file type detection via analyze_drop_command.

    // --- Initialization ---
    useEffect(() => {
        loadToken();
        const fetchProjectLoras = async () => {
            try {
                const projects: ProjectLoraInfo[] = await invoke('list_loras_by_project_command');
                setProjectLoras(projects);
            } catch (e) {
                console.error("Error loading project loras", e);
            }
        };
        fetchProjectLoras();
    }, []);

    // --- Debounced Search Effect ---
    useEffect(() => {
        if (!showFindNew || hfQuery.trim() === '') return;

        // If query and type are both the same as last search, skip unless it's a forced refresh
        if (hfQuery === lastSearchedQuery && hfType === lastSearchedType) return;

        const timer = setTimeout(() => {
            performSearch();
        }, 100); // 0.1s debounce

        return () => clearTimeout(timer);
    }, [hfQuery, hfType, showFindNew]);

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
        // Broaden filter to ensure we catch index files regardless of strict classification
        const weights = files.filter(f => f.file_type !== 'gguf' && f.file_type !== 'mmproj' && f.file_type !== 'dataset_file');

        if (weights.length > 0) {
            // Select all relevant files for a full model download
            // Typically: .safetensors, config.json, tokenizer.json, vocab.json/txt
            weights.forEach(w => {
                const name = w.path.toLowerCase();
                const isConfig = name.includes("config.json") ||
                    name.includes("tokenizer") ||
                    name.includes("special_tokens_map.json") ||
                    name.includes("chat_template.json") ||
                    name.includes("vocab.json") ||
                    name.includes("merges.txt") ||
                    name.includes("generation_config.json") ||
                    name.includes("preprocessor_config.json") ||
                    name.endsWith(".index.json") ||
                    name.endsWith(".json"); // Fallback for other json configs

                const isSafetensor = name.endsWith(".safetensors") || name.endsWith(".bin") || name.endsWith(".pt");

                if (isConfig || isSafetensor) {
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
                directory: type === 'model',
                filters: type === 'dataset' ? [
                    { name: 'Dataset Files', extensions: ['jsonl', 'parquet', 'arrow', 'csv', 'json'] },
                    { name: 'All Files', extensions: ['*'] }
                ] : type === 'lora' ? [
                    { name: 'LoRA Files', extensions: ['safetensors', 'bin', 'pt'] },
                    { name: 'All Files', extensions: ['*'] }
                ] : undefined,
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

    const handleDelete = async (r: Resource) => {
        if (!await confirm(`Are you sure you want to delete ${r.name}?`)) return;
        try {
            await invoke('delete_resource_command', {
                resourceType: r.type === 'gguf' ? 'model' : r.type,
                resourcePath: r.path.split(/[\\/]/).pop() // Just filename
            });
            addNotification("Deleted successfully", "success");
            loadResources();
        } catch (e) {
            addNotification(`Delete failed: ${e}`, 'error');
        }
    };

    const handleRename = (r: Resource) => {
        setRenamingResource(r);
        setRenameValue(r.name);
    };

    const handleRenameSubmit = async () => {
        if (!renamingResource || !renameValue.trim()) return;

        try {
            await invoke('rename_resource_command', {
                resourceType: renamingResource.type === 'gguf' ? 'model' : renamingResource.type,
                currentName: renamingResource.name,
                newName: renameValue.trim()
            });
            addNotification(`Renamed to ${renameValue}`, 'success');
            setRenamingResource(null);
            loadResources();
        } catch (e) {
            console.error(e);
            addNotification(`Rename failed: ${e}`, 'error');
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



    const handleExpandResource = async (r: Resource) => {
        if (expandedResourceId === r.path) {
            setExpandedResourceId(null);
            return;
        }

        setExpandedResourceId(r.path);

        // If already cached, don't refetch
        if (expandedFiles[r.path]) return;

        setLoadingExpansion(true);
        try {
            // 1. Get Local Files
            // If the path is likely a file (e.g. ends with .gguf), use its parent directory for listing
            let listPath = r.path;
            if (r.type === 'gguf' && (r.path.endsWith('.gguf') || r.path.endsWith('.bin'))) {
                // Try to get parent directory
                const lastSlash = Math.max(r.path.lastIndexOf('/'), r.path.lastIndexOf('\\'));
                if (lastSlash !== -1) {
                    listPath = r.path.substring(0, lastSlash);
                }
            }

            const localFilesList: string[] = await invoke('list_directory_command', { path: listPath });
            const localSet = new Set(localFilesList);

            // 2. Get Remote Files (Try to guess Repo ID from folder name)
            // Convention: author--repo
            let remoteFiles: HFFile[] = [];
            const folderName = r.path.split(/[\\/]/).pop() || "";
            if (folderName.includes("--")) {
                const repoId = folderName.replace("--", "/");
                try {
                    remoteFiles = await invoke('list_hf_repo_files_command', {
                        repoId: repoId,
                        token: hfToken || null,
                        resourceType: r.type === 'dataset' ? 'dataset' : 'model'
                    });
                } catch (e) {
                    console.warn("Could not fetch remote files for " + repoId, e);
                }
            }

            setExpandedFiles(prev => ({
                ...prev,
                [r.path]: { local: localSet, remote: remoteFiles }
            }));

        } catch (e) {
            addNotification(`Failed to load file details: ${e}`, 'error');
        } finally {
            setLoadingExpansion(false);
        }
    };

    const handleDownloadSingleFile = async (r: Resource, filename: string) => {
        const folderName = r.path.split(/[\\/]/).pop() || "";
        const repoId = folderName.replace("--", "/");
        const taskId = `dl_file_${Date.now()}`;

        // Add to global download tasks
        setDownloadTasks(prev => [
            ...prev,
            {
                id: taskId,
                type: r.type,
                name: filename,
                progress: 0,
                status: 'pending',
                onCancel: () => invoke('cancel_download_command', { taskId }).catch(console.error)
            }
        ]);

        addNotification(`Downloading ${filename}...`, 'info');

        try {
            if (r.type === 'dataset') {
                await invoke('download_hf_dataset_command', {
                    datasetId: repoId,
                    files: [filename],
                    token: hfToken || null,
                    taskId
                });
            } else {
                await invoke('download_hf_model_command', {
                    modelId: repoId,
                    files: [filename],
                    token: hfToken || null,
                    taskId
                });
            }
            // Removed optimistic update - wait for background tasks to finish and refresh
        } catch (e) {
            addNotification(`Download failed: ${e}`, 'error');
            setDownloadTasks(prev => prev.map(t => t.id === taskId ? { ...t, status: 'error' } : t));
        }
    };

    const handleDeleteCheckpoint = async (checkpointPath: string) => {
        try {
            await invoke('delete_resource_command', {
                resourceType: 'lora',
                resourcePath: checkpointPath
            });
            loadResources();
            setTimeout(() => loadResources(), 500); // Double check refresh
            addNotification('Checkpoint deleted', 'success');
        } catch (e) {
            addNotification(`Delete failed: ${e}`, 'error');
        }
    };

    // --- Dataset Conversion ---
    const handleConvertDataset = async (r: Resource) => {
        if (isConvertingMap[r.path]) return;

        const taskId = `conv_${Date.now()}`;

        setDownloadTasks(prev => [...prev, {
            id: taskId,
            name: `Converting ${r.name}`,
            progress: 0,
            status: 'downloading',
            type: 'dataset',
            onCancel: async () => {
                try {
                    await invoke('cancel_download_command', { taskId: taskId });
                    addLogMessage(`Conversion cancelled: ${taskId}`);
                } catch (e) {
                    addLogMessage(`Failed to cancel conversion: ${e}`);
                }
            }
        }]);

        setConvertingMap(prev => ({ ...prev, [r.path]: true }));
        addNotification('Starting dataset conversion...', 'info');

        try {
            await invoke('convert_dataset_command', {
                sourcePath: r.path,
                destinationPath: r.path, // Output to same folder (in /processed_data subfolder)
                taskId: taskId
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

    const handleConvertModel = async (r: Resource) => {
        if (isConvertingMap[r.path]) return;

        const taskId = `conv_mod_${Date.now()}`;
        setDownloadTasks(prev => [...prev, {
            id: taskId,
            name: `Converting ${r.name}`,
            progress: 0,
            status: 'downloading',
            type: 'model',
            onCancel: async () => { }
        }]);

        setConvertingMap(prev => ({ ...prev, [r.path]: true }));
        addNotification(`Starting GGUF conversion for ${r.name}...`, 'info');

        // Default to q4_k_m for general usage
        const quant = "q8_0";
        const outputPath = r.path + `.${quant}.gguf`;

        try {
            // 1. Try Unsloth
            addLogMessage(`Attempting Unsloth conversion for ${r.name}...`);
            await invoke('convert_unsloth_gguf_command', {
                sourcePath: r.path,
                outputPath: outputPath,
                quantizationType: quant,
                loraPath: null
            });
            addNotification('Unsloth conversion successful!', 'success');
            loadResources();
        } catch (unslothErr) {
            console.warn("Unsloth conversion failed, trying fallback", unslothErr);
            addLogMessage(`Unsloth failed: ${unslothErr}. Falling back to llama.cpp...`);

            // 2. Fallback to Standard
            try {
                await invoke('convert_hf_to_gguf_command', {
                    sourcePath: r.path,
                    outputPath: outputPath,
                    quantizationType: quant
                });
                addNotification('Fallback conversion successful!', 'success');
                loadResources();
            } catch (fbErr) {
                console.error("Fallback failed", fbErr);
                addNotification(`All conversion methods failed: ${fbErr}`, 'error');
            }
        } finally {
            setDownloadTasks(prev => prev.filter(t => t.id !== taskId));
            setConvertingMap(prev => ({ ...prev, [r.path]: false }));
        }
    };

    const handleConvertLora = async (r: Resource) => {
        if (isConvertingMap[r.path]) return;
        let taskId: string | null = null;

        // 1. Prompt for Base Model if not provided
        try {
            let base_model_path: string | undefined | null = r.base_model;

            if (!base_model_path) {
                addNotification("Please select the Base Model for this LoRA", 'info');
                base_model_path = await open({
                    multiple: false,
                    directory: true,
                    title: "Select Base Model for LoRA Conversion"
                }) as string | null;
            }

            if (!base_model_path) return; // Cancelled

            taskId = `conv_lora_${Date.now()}`;
            setDownloadTasks(prev => [...prev, {
                id: taskId!,
                name: `Converting LoRA ${r.name}`,
                progress: 0,
                status: 'downloading',
                type: 'model',
                onCancel: async () => { }
            }]);

            setConvertingMap(prev => ({ ...prev, [r.path]: true }));
            addNotification(`Converting LoRA ${r.name}...`, 'info');

            const quant = "q8_0";
            const outputPath = r.path + `.${quant}.gguf`; // Merged GGUF usually?

            // 1. Try Unsloth (Merge & Export)
            // Note: Unsloth conversion with lora_path implies merging into base
            try {
                addLogMessage(`Attempting Unsloth Merge+Convert...`);
                await invoke('convert_unsloth_gguf_command', {
                    sourcePath: base_model_path as string, // Base
                    outputPath: outputPath,
                    quantizationType: quant,
                    loraPath: r.path
                });
                addNotification('Unsloth LoRA Merge successful!', 'success');
                loadResources();
            } catch (unslothErr) {
                console.warn("Unsloth LoRA failed", unslothErr);
                addLogMessage(`Unsloth LoRA failed. Falling back to simple LoRA adapter conversion...`);

                // 2. Fallback: Convert LoRA to GGUF Adapter (Not merged)
                // This requires a different output name usually to distinguish, but user expects GGUF.
                // Ideally we'd merge with llama.cpp `export-lora` but that's complex.
                // We will run `convert_lora_to_gguf_command` which creates an adapter GGUF.
                try {
                    const adapterOutput = r.path + `.adapter.${quant}.gguf`;
                    await invoke('convert_lora_to_gguf_command', {
                        loraPath: r.path,
                        basePath: base_model_path as string,
                        outputPath: adapterOutput,
                        quantizationType: quant
                    });
                    addNotification(`LoRA converted to GGUF Adapter (not merged): ${adapterOutput}`, 'success');
                    loadResources();
                } catch (fbErr) {
                    addNotification(`LoRA conversion failed: ${fbErr}`, 'error');
                }
            }

        } catch (e) {
            console.error(e);
        } finally {
            if (taskId) setDownloadTasks(prev => prev.filter(t => t.id !== taskId));
            setConvertingMap(prev => ({ ...prev, [r.path]: false }));
        }
    };

    const handleFixDataset = async (r: Resource) => {
        if (fixingMap[r.path]) return;
        setFixingMap(prev => ({ ...prev, [r.path]: true }));
        addNotification('Starting dataset repair...', 'info');

        try {
            await invoke('fix_dataset_command', { sourcePath: r.path });
            addNotification('Dataset repaired successfully!', 'success');
            loadResources();
        } catch (e) {
            addNotification(`Repair failed: ${e}`, 'error');
            addLogMessage(`Repair Error: ${e}`);
        } finally {
            setFixingMap(prev => ({ ...prev, [r.path]: false }));
        }
    };



    // --- Search & HF Actions ---
    const performSearch = async () => {
        if (!hfQuery) return;
        setIsSearching(true);
        setLastSearchedQuery(hfQuery);
        setLastSearchedType(hfType);
        try {
            const res: HFSearchResult[] = await invoke('search_huggingface_command', {
                query: hfQuery,
                resourceType: hfType,
                author: hfAuthor || null,
                modalities: hfModalities || null,
                sizeRange: hfSizeRange || null
            });
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

        setIsDownloading(true);
        const taskId = `dl_repo_${Date.now()}`;
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
            setIsDownloading(false);
            return;
        }

        // Add to global download tasks
        setDownloadTasks(prev => [
            ...prev,
            {
                id: taskId,
                type: hfType as any,
                name: selectedRepo,
                progress: 0,
                status: 'pending',
                onCancel: () => invoke('cancel_download_command', { taskId }).catch(console.error)
            }
        ]);

        try {
            if (hfType === 'dataset') {
                await invoke('download_hf_dataset_command', {
                    datasetId: selectedRepo,
                    files: filesToDownload.length > 0 ? filesToDownload : null,
                    token: hfToken || null,
                    taskId
                });
            } else {
                await invoke('download_hf_model_command', {
                    modelId: selectedRepo,
                    files: filesToDownload.length > 0 ? filesToDownload : null,
                    token: hfToken || null,
                    taskId
                });
            }
            addNotification("Download started in background", "info");
            setShowFindNew(false);
            loadResources();
        } catch (e) {
            addNotification(`Download failed: ${e}`, 'error');
            setDownloadTasks(prev => prev.map(t => t.id === taskId ? { ...t, status: 'error' } : t));
        } finally {
            setIsDownloading(false);
            // Reset
            setSelectedGGUFs(new Set());
            setSelectedMMProjs(new Set());
            setSelectedWeights(new Set());
            setSelectedDatasetFiles(new Set());
            setSelectedRepo(null);
            setRepoFiles([]);
        }
    };

    const handleDeleteProject = async (projectName: string) => {
        if (!window.confirm(`Are you sure you want to delete project "${projectName}"? This cannot be undone.`)) return;

        try {
            const path = `data/outputs/${projectName}`;
            await invoke('delete_resource_command', {
                resourceType: 'lora_project',
                resourcePath: path
            });
            addLogMessage(`Project ${projectName} deleted.`);
            addNotification(`Project ${projectName} deleted`, 'success');
            loadResources();
        } catch (e) {
            addLogMessage(`Error deleting project: ${e}`);
            addNotification(`Error deleting project: ${e}`, 'error');
        }
    };

    // --- Memoized Lists for Dashboard ---
    const { localModels, datasets } = useMemo(() => ({
        localModels: resources.filter(r => r.type === 'model' || r.type === 'gguf'),
        datasets: resources.filter(r => r.type === 'dataset'),
    }), [resources]);



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

                    <div className="rd-dropdown-container" ref={importRef}>
                        <div
                            className={`rd-import-btn ${isImportOpen ? 'open' : ''}`}
                            onClick={() => setIsImportOpen(!isImportOpen)}
                            style={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: '8px',
                                padding: '10px 14px',
                                background: 'rgba(0,0,0,0.3)',
                                border: isImportOpen ? '1px solid rgba(139,92,246,0.5)' : '1px solid rgba(255,255,255,0.15)',
                                borderRadius: '10px',
                                color: 'white',
                                fontSize: '14px',
                                cursor: 'pointer',
                                transition: 'all 0.2s'
                            }}
                        >
                            <FolderInput size={16} />
                            <span>Import</span>
                            <ChevronDown size={14} style={{ transform: isImportOpen ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }} />
                        </div>
                        {isImportOpen && (
                            <div className="rd-dropdown-menu">
                                <div
                                    className="rd-dropdown-item"
                                    onClick={() => { handleImport('model'); setIsImportOpen(false); }}
                                >
                                    Import Model
                                </div>
                                <div
                                    className="rd-dropdown-item"
                                    onClick={() => { handleImport('lora'); setIsImportOpen(false); }}
                                >
                                    Import LoRA
                                </div>
                                <div
                                    className="rd-dropdown-item"
                                    onClick={() => { handleImport('dataset'); setIsImportOpen(false); }}
                                >
                                    Import Dataset
                                </div>
                            </div>
                        )}
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
                                {localModels.map(r => (
                                    <ResourceRow
                                        key={r.path}
                                        resource={r}
                                        isSelected={selectedPaths.has(r.path)}
                                        toggleSelection={toggleSelection}
                                        handleExpandResource={handleExpandResource}
                                        handleConvertDataset={handleConvertDataset}
                                        handleFixDataset={() => { }}
                                        handleDelete={handleDelete}
                                        handleConvertModel={handleConvertModel}
                                        handleConvertLora={handleConvertLora}
                                        handleRename={handleRename}
                                        converting={!!isConvertingMap[r.path]}
                                        fixing={false}
                                    />
                                ))}
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
                                        <Layers size={18} className="text-blue-400" />
                                        <span className="rd-project-title">{project.project_name}</span>
                                        <span className="rd-project-count">
                                            {project.checkpoints.length} checkpoint{project.checkpoints.length !== 1 ? 's' : ''}
                                        </span>
                                        <div style={{ flex: 1 }} />
                                        <button
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                handleDeleteProject(project.project_name);
                                            }}
                                            className="rd-delete-btn"
                                            title="Delete Project"
                                            style={{ opacity: 0.7, padding: '4px' }}
                                            onMouseEnter={(e) => e.currentTarget.style.opacity = '1'}
                                            onMouseLeave={(e) => e.currentTarget.style.opacity = '0.7'}
                                        >
                                            <Trash2 size={16} />
                                        </button>
                                    </div>

                                    {/* Checkpoints */}
                                    {expandedProjects.has(project.project_name) && (
                                        <div className="list-container" style={{ marginLeft: 'var(--space-8)', marginBottom: 'var(--space-4)' }}>
                                            {project.checkpoints.map(checkpoint => (
                                                <div key={checkpoint.path} className="list-item">
                                                    <div
                                                        className={`rd-item-select ${selectedPaths.has(checkpoint.path) ? 'selected' : ''}`}
                                                        onClick={() => toggleSelection(checkpoint.path)}
                                                    >
                                                        {selectedPaths.has(checkpoint.path) ? <Check size={20} /> : <Square size={20} />}
                                                    </div>

                                                    <div className="rd-item-icon" style={{ width: '32px', height: '32px' }}>
                                                        <Layers size={18} />
                                                    </div>

                                                    <div className="rd-item-details">
                                                        <div className="rd-item-name" style={{ fontSize: '0.875rem' }}>
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
                                                    <button
                                                        onClick={(e) => {
                                                            e.stopPropagation();
                                                            if (checkpoint.gguf_path) {
                                                                // If GGUF exists, treat this as an export action
                                                                toggleSelection(checkpoint.gguf_path);
                                                                handleExport();
                                                            } else {
                                                                // Construct a temporary Resource object for the LoRA
                                                                const loraResource: Resource = {
                                                                    name: checkpoint.name,
                                                                    path: checkpoint.path,
                                                                    type: 'lora',
                                                                    size: 'N/A', // or fetch
                                                                    is_mmproj: false,
                                                                    base_model: project.base_model
                                                                };
                                                                handleConvertLora(loraResource);
                                                            }
                                                        }}
                                                        className="rd-delete-btn"
                                                        style={{ color: checkpoint.gguf_path ? 'var(--accent-primary)' : 'rgba(255,255,255,0.4)' }}
                                                        title={checkpoint.gguf_path ? "Export GGUF" : "Convert to GGUF"}
                                                        disabled={!!isConvertingMap[checkpoint.path]}
                                                    >
                                                        {isConvertingMap[checkpoint.path] ? (
                                                            <Loader2 size={18} className="animate-spin" />
                                                        ) : checkpoint.gguf_path ? (
                                                            <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                                                                <FileCode size={18} />
                                                                <ArrowUpRight size={14} />
                                                            </div>
                                                        ) : (
                                                            <Download size={18} />
                                                        )}
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
                            <div className="rd-section-content">
                                {datasets.map(r => (
                                    <ResourceRow
                                        key={r.path}
                                        resource={r}
                                        isSelected={selectedPaths.has(r.path)}
                                        toggleSelection={toggleSelection}
                                        handleExpandResource={handleExpandResource}
                                        handleConvertDataset={handleConvertDataset}
                                        handleFixDataset={handleFixDataset}
                                        handleDelete={handleDelete}
                                        handleConvertModel={handleConvertModel}
                                        handleConvertLora={handleConvertLora}
                                        handleRename={handleRename}
                                        converting={!!isConvertingMap[r.path]}
                                        fixing={!!fixingMap[r.path]}
                                    />
                                ))}
                                {datasets.length === 0 && <span className="text-muted italic">No datasets found.</span>}
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {/* --- Find New Overlay --- */}

            {/* Rename Modal */}
            <Modal
                isOpen={!!renamingResource}
                onClose={() => setRenamingResource(null)}
                title={`Rename ${renamingResource?.type === 'gguf' ? 'Model' : renamingResource?.type === 'dataset' ? 'Dataset' : 'Resource'}`}
                footer={
                    <>
                        <Button variant="ghost" onClick={() => setRenamingResource(null)}>Cancel</Button>
                        <Button variant="primary" onClick={handleRenameSubmit} disabled={!renameValue.trim() || renameValue === renamingResource?.name}>
                            Rename
                        </Button>
                    </>
                }
            >
                <div className="flex flex-col gap-4">
                    <p className="text-sm text-gray-400">
                        Enter a new name for <strong>{renamingResource?.name}</strong>.
                        Note: This will rename the file/folder on disk.
                    </p>
                    <Input
                        value={renameValue}
                        onChange={(e) => setRenameValue(e.target.value)}
                        placeholder="New name..."
                        autoFocus
                        onKeyDown={(e) => {
                            if (e.key === 'Enter' && renameValue.trim()) {
                                handleRenameSubmit();
                            }
                        }}
                    />
                </div>
            </Modal>

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
                                    onClick={() => setHfType('model')}
                                >
                                    Models
                                </div>
                                <div
                                    className={`fn-toggle-opt ${hfType === 'dataset' ? 'active' : ''}`}
                                    onClick={() => setHfType('dataset')}
                                >
                                    Datasets
                                </div>
                            </div>
                            <input
                                className="fn-input"
                                placeholder={`Search HuggingFace ${hfType}s...`}
                                value={hfQuery}
                                onChange={e => setHfQuery(e.target.value)}
                                onKeyDown={(e) => {
                                    if (e.key === 'Tab') {
                                        e.preventDefault();
                                        setHfType(hfType === 'model' ? 'dataset' : 'model');
                                    }
                                }}
                                autoFocus
                            />
                            <button className="btn btn-icon" onClick={() => setShowFilters(!showFilters)} title="Toggle Filters">
                                <SlidersHorizontal size={18} style={{ color: showFilters ? 'var(--accent-primary)' : 'inherit' }} />
                            </button>
                            <button className="btn btn-icon" onClick={performSearch}>
                                {isSearching ? <Loader2 size={18} className="animate-spin" /> : <Search size={18} />}
                            </button>
                        </div>
                        {showFilters && (
                            <div className="fn-filters-bar" style={{
                                display: 'flex',
                                gap: '12px',
                                padding: '12px 24px',
                                background: 'rgba(255,255,255,0.03)',
                                borderBottom: '1px solid rgba(255,255,255,0.05)',
                                alignItems: 'center'
                            }}>
                                <div className="fn-filter-group" style={{ flex: 1 }}>
                                    <label style={{ fontSize: '11px', color: 'var(--text-dim)', textTransform: 'uppercase', marginBottom: '4px', display: 'block' }}>Author / Org</label>
                                    <input
                                        className="fn-input-small"
                                        placeholder="Username or Org..."
                                        value={hfAuthor}
                                        onChange={e => setAuthor(e.target.value)}
                                        style={{ width: '100%', background: 'rgba(0,0,0,0.2)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '4px', padding: '6px 10px', color: 'white', fontSize: '13px' }}
                                    />
                                </div>
                                <div className="fn-filter-group" style={{ flex: 1 }}>
                                    <label style={{ fontSize: '11px', color: 'var(--text-dim)', textTransform: 'uppercase', marginBottom: '4px', display: 'block' }}>Modality</label>
                                    <select
                                        className="fn-input-small"
                                        value={hfModalities}
                                        onChange={e => setModalities(e.target.value)}
                                        style={{ width: '100%', background: 'rgba(0,0,0,0.2)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '4px', padding: '6px 10px', color: 'white', fontSize: '13px' }}
                                    >
                                        <option value="">Any Modality</option>
                                        <option value="text">Text</option>
                                        <option value="image">Vision / Image</option>
                                        <option value="audio">Audio</option>
                                        <option value="video">Video</option>
                                        <option value="3d">3D</option>
                                    </select>
                                </div>
                                {hfType === 'dataset' && (
                                    <div className="fn-filter-group" style={{ flex: 1 }}>
                                        <label style={{ fontSize: '11px', color: 'var(--text-dim)', textTransform: 'uppercase', marginBottom: '4px', display: 'block' }}>Size Range</label>
                                        <select
                                            className="fn-input-small"
                                            value={hfSizeRange}
                                            onChange={e => setSizeRange(e.target.value)}
                                            style={{ width: '100%', background: 'rgba(0,0,0,0.2)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '4px', padding: '6px 10px', color: 'white', fontSize: '13px' }}
                                        >
                                            <option value="">Any Size</option>
                                            <option value="1K">&lt; 1K rows</option>
                                            <option value="10K">1K - 10K</option>
                                            <option value="100K">10K - 100K</option>
                                            <option value="1M">100K - 1M</option>
                                            <option value="10M">1M - 10M</option>
                                            <option value="100M">10M - 100M</option>
                                            <option value="1G">100M - 1B</option>
                                            <option value="10G">1B - 10B</option>
                                        </select>
                                    </div>
                                )}
                                <button
                                    className="btn btn-secondary btn-sm"
                                    onClick={() => {
                                        setAuthor('');
                                        setModalities('');
                                        setSizeRange('');
                                    }}
                                    style={{ marginTop: '16px', padding: '6px 10px' }}
                                >
                                    Reset
                                </button>
                            </div>
                        )}
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
                                    <div className="fn-result-title">
                                        {res.name}
                                        {res.author && <span className="fn-result-author"> by {res.author}</span>}
                                    </div>
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
                                                                background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(59, 130, 246, 0.1) 100%)',
                                                                border: '1px solid rgba(59, 130, 246, 0.4)',
                                                                padding: '12px 16px',
                                                                margin: '0 0 12px 0',
                                                                borderRadius: '8px',
                                                                fontWeight: '600',
                                                                color: '#93c5fd',
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
                                                                background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(59, 130, 246, 0.1) 100%)',
                                                                border: '1px solid rgba(59, 130, 246, 0.4)',
                                                                padding: '12px 16px',
                                                                margin: '0 0 12px 0',
                                                                borderRadius: '8px',
                                                                fontWeight: '600',
                                                                color: '#93c5fd',
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
                                        disabled={!canDownload || isDownloading}
                                        onClick={startDownload}
                                        style={{ width: '100%', marginTop: '1rem' }}
                                    >
                                        {isDownloading ? (
                                            <>
                                                <Loader2 size={16} className="animate-spin" style={{ marginRight: '8px' }} />
                                                Starting Download...
                                            </>
                                        ) : (
                                            <>
                                                <Download size={16} style={{ marginRight: '8px' }} />
                                                {hfType === 'model'
                                                    ? `Download Selected (${selectedGGUFs.size + selectedMMProjs.size + selectedWeights.size})`
                                                    : `Download Selected (${selectedDatasetFiles.size})`
                                                }
                                            </>
                                        )}
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

            {/* --- Resource Detail Overlay --- */}
            {activeResource && (
                <div className="rd-overlay" onClick={() => setExpandedResourceId(null)}>
                    <div className="rd-overlay-card" onClick={e => e.stopPropagation()}>
                        <div className="rd-overlay-header">
                            <div className="rd-overlay-header-info">
                                <h2>{activeResource.name}</h2>
                                <div className="rd-item-meta">
                                    <span>{activeResource.size}</span>
                                    {activeResource.quantization && <span className="rd-badge blue">{activeResource.quantization}</span>}
                                    {activeResource.is_mmproj && <span className="rd-badge blue">Vision</span>}
                                </div>
                            </div>
                            <button className="fn-close-btn" onClick={() => setExpandedResourceId(null)}>
                                <X size={20} />
                            </button>
                        </div>
                        <div className="rd-overlay-body">
                            {loadingExpansion ? (
                                <div className="flex justify-center p-8"><Loader2 size={32} className="animate-spin text-blue-400" /></div>
                            ) : (
                                <div className="rd-file-grid">
                                    {activeFiles.length === 0 ? (
                                        <div className="p-8 text-center text-muted italic">No files found in this directory.</div>
                                    ) : (
                                        activeFiles.map(file => {
                                            const activeTask = downloadTasks.find(t => t.name === file.name && (t.status === 'downloading' || t.status === 'pending'));
                                            return (
                                                <div key={file.name} className="rd-file-row">
                                                    <div className="rd-file-info">
                                                        <span className="rd-file-name" title={file.name}>{file.name}</span>
                                                        {file.size && <span className="rd-file-size">{formatBytes(file.size)}</span>}
                                                        {activeTask && (
                                                            <span style={{ fontSize: '10px', color: 'var(--accent-primary)', marginLeft: '8px' }}>
                                                                {activeTask.progress > 0 ? `${Math.round(activeTask.progress)}%` : 'Pending...'}
                                                            </span>
                                                        )}
                                                    </div>
                                                    <div className="rd-file-actions">
                                                        {activeTask ? (
                                                            <div className="rd-file-action-btn loading">
                                                                <Loader2 size={16} className="animate-spin" />
                                                            </div>
                                                        ) : file.isLocal ? (
                                                            <button className="rd-file-action-btn downloaded" disabled>
                                                                <Check size={16} />
                                                            </button>
                                                        ) : (
                                                            <button
                                                                className="rd-file-action-btn"
                                                                title="Download file"
                                                                onClick={() => handleDownloadSingleFile(activeResource, file.name)}
                                                            >
                                                                <Plus size={16} />
                                                            </button>
                                                        )}
                                                    </div>
                                                </div>
                                            );
                                        })
                                    )}
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default ResourceDashboard;