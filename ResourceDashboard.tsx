import React, { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { Card } from './components/Card';
import { Button } from './components/Button';
import { Input } from './components/Input';
import { Modal } from './components/Modal';
import {
    Download,
    Trash2,
    Search,
    Package,
    Database as DatabaseIcon,
    Brain,
    Layers,
    Eye,
    Filter,
    TrendingDown,
    Star,
    X
} from 'lucide-react';

interface Resource {
    name: string;
    size: string;
    path: string;
    type: 'model' | 'gguf' | 'lora' | 'dataset';
    quantization?: string;
    is_mmproj?: boolean;
    has_vision?: boolean;
}

interface HFSearchResult {
    id: string;
    name: string;
    downloads: number;
    likes: number;
    author?: string;
    tags?: string[];
}

interface HFFile {
    path: string;
    size: number | null;
}

interface ResourceDashboardProps {
    addLogMessage: (message: string) => void;
    addNotification: (message: string, type?: 'success' | 'error' | 'info') => void;
    setDownloadTasks: React.Dispatch<React.SetStateAction<any[]>>;
}

const ResourceDashboardNew: React.FC<ResourceDashboardProps> = ({
    addLogMessage,
    addNotification,
    setDownloadTasks,
}) => {
    // Local Resources
    const [models, setModels] = useState<Resource[]>([]);
    const [ggufModels, setGgufModels] = useState<Resource[]>([]);
    const [loras, setLoras] = useState<Resource[]>([]);
    const [datasets, setDatasets] = useState<Resource[]>([]);

    // HuggingFace Search Modal
    const [showHFModal, setShowHFModal] = useState(false);
    const [hfSearchType, setHfSearchType] = useState<'model' | 'dataset'>('model');
    const [hfSearchQuery, setHfSearchQuery] = useState('');
    const [hfSearchResults, setHfSearchResults] = useState<HFSearchResult[]>([]);
    const [isSearching, setIsSearching] = useState(false);
    const [hfToken, setHfToken] = useState('');

    // File Selection
    const [showFileSelector, setShowFileSelector] = useState(false);
    const [selectedRepo, setSelectedRepo] = useState<string | null>(null);
    const [repoFiles, setRepoFiles] = useState<HFFile[]>([]);
    const [selectedFiles, setSelectedFiles] = useState<Set<string>>(new Set());
    const [isLoadingFiles, setIsLoadingFiles] = useState(false);

    // Filters
    const [quantFilter, setQuantFilter] = useState('all');
    const [sortBy, setSortBy] = useState<'name' | 'size' | 'recent'>('name');

    useEffect(() => {
        loadLocalResources();
        loadSavedToken();
    }, []);

    const loadLocalResources = async () => {
        try {
            const allResources: Resource[] = await invoke('list_all_resources_command');

            // Associate mmproj with models
            const resourcesByFolder = new Map<string, Resource[]>();
            allResources.forEach(r => {
                const dir = r.path.substring(0, Math.max(r.path.lastIndexOf('/'), r.path.lastIndexOf('\\\\')));
                if (!resourcesByFolder.has(dir)) resourcesByFolder.set(dir, []);
                resourcesByFolder.get(dir)?.push(r);
            });

            const processedResources: Resource[] = [];
            const hiddenPaths = new Set<string>();

            resourcesByFolder.forEach((group) => {
                const mmproj = group.find(r => r.is_mmproj);
                const ggufs = group.filter(r => r.type === 'gguf' && !r.is_mmproj);

                if (mmproj && ggufs.length > 0) {
                    ggufs.forEach(g => {
                        (g as any).has_vision = true;
                        (g as any).mmproj_path = mmproj.path;
                    });
                    hiddenPaths.add(mmproj.path);
                }
                processedResources.push(...group);
            });

            const visibleResources = processedResources.filter(r => !hiddenPaths.has(r.path));

            setModels(visibleResources.filter(r => r.type === 'model'));
            setGgufModels(visibleResources.filter(r => r.type === 'gguf'));
            setLoras(visibleResources.filter(r => r.type === 'lora'));
            setDatasets(visibleResources.filter(r => r.type === 'dataset'));
        } catch (error) {
            addLogMessage(`Error loading resources: ${error}`);
        }
    };

    const loadSavedToken = async () => {
        try {
            const token: string = await invoke('get_hf_token_command');
            if (token) setHfToken(token);
        } catch (error) { /* Ignore */ }
    };

    const handleHFSearch = async () => {
        if (!hfSearchQuery.trim()) return;

        setIsSearching(true);
        try {
            const results: HFSearchResult[] = await invoke('search_huggingface_command', {
                query: hfSearchQuery,
                resourceType: hfSearchType
            });
            setHfSearchResults(results);
        } catch (error) {
            addNotification('Search failed', 'error');
            addLogMessage(`HF Search error: ${error}`);
        } finally {
            setIsSearching(false);
        }
    };

    const handleSelectRepo = async (repoId: string) => {
        setSelectedRepo(repoId);
        setShowFileSelector(true);
        setIsLoadingFiles(true);
        setSelectedFiles(new Set());

        try {
            const files: HFFile[] = await invoke('list_repo_files_command', {
                repoId,
                token: hfToken || null
            });
            setRepoFiles(files);
        } catch (error) {
            addNotification('Failed to load files', 'error');
            addLogMessage(`File list error: ${error}`);
        } finally {
            setIsLoadingFiles(false);
        }
    };

    const handleDownload = async () => {
        if (!selectedRepo || selectedFiles.size === 0) return;

        setShowFileSelector(false);
        setShowHFModal(false);

        const taskId = `dl_${Date.now()}`;
        setDownloadTasks(prev => [...prev, {
            id: taskId,
            name: selectedRepo,
            progress: 0,
            status: 'downloading',
            type: hfSearchType
        }]);

        try {
            if (hfSearchType === 'model') {
                await invoke('download_hf_model_command', {
                    modelId: selectedRepo,
                    files: Array.from(selectedFiles),
                    token: hfToken || null
                });
            } else {
                await invoke('download_hf_dataset_command', {
                    datasetId: selectedRepo,
                    token: hfToken || null
                });
            }

            addNotification('Download started', 'success');
            loadLocalResources();
        } catch (error) {
            addNotification('Download failed', 'error');
            addLogMessage(`Download error: ${error}`);
        }
    };

    const handleDelete = async (resource: Resource) => {
        try {
            await invoke('delete_resource_command', { path: resource.path });
            addNotification('Resource deleted', 'success');
            loadLocalResources();
        } catch (error) {
            addNotification('Delete failed', 'error');
        }
    };

    const getResourceIcon = (type: string) => {
        switch (type) {
            case 'model': return <Brain size={24} className="text-purple-400" />;
            case 'gguf': return <Package size={24} className="text-blue-400" />;
            case 'lora': return <Layers size={24} className="text-cyan-400" />;
            case 'dataset': return <DatabaseIcon size={24} className="text-green-400" />;
            default: return <Package size={24} />;
        }
    };

    const filteredGgufs = quantFilter === 'all'
        ? ggufModels
        : ggufModels.filter(g => g.quantization === quantFilter);

    const allResources = [...models, ...filteredGgufs, ...loras, ...datasets];
    const sortedResources = [...allResources].sort((a, b) => {
        if (sortBy === 'name') return a.name.localeCompare(b.name);
        if (sortBy === 'size') return a.size.localeCompare(b.size);
        return 0;
    });

    return (
        <div className="p-6 space-y-6">
            {/* Header */}
            <div className="flex justify-between items-center">
                <div>
                    <h1 className="text-2xl font-bold text-white mb-1">My Resources</h1>
                    <p className="text-gray-400 text-sm">
                        {allResources.length} items • {ggufModels.length} models • {datasets.length} datasets
                    </p>
                </div>

                <Button
                    variant="primary"
                    size="lg"
                    leftIcon={<Download size={20} />}
                    onClick={() => setShowHFModal(true)}
                >
                    Download from HuggingFace
                </Button>
            </div>

            {/* Filters */}
            <div className="flex gap-3 items-center">
                <div className="flex items-center gap-2 text-sm text-gray-400">
                    <Filter size={16} />
                    <span>Filters:</span>
                </div>

                {ggufModels.length > 0 && (
                    <select
                        value={quantFilter}
                        onChange={e => setQuantFilter(e.target.value)}
                        className="bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white"
                    >
                        <option value="all">All Quantizations</option>
                        {Array.from(new Set(ggufModels.map(g => g.quantization).filter(Boolean))).map(q => (
                            <option key={q} value={q}>{q}</option>
                        ))}
                    </select>
                )}

                <select
                    value={sortBy}
                    onChange={e => setSortBy(e.target.value as any)}
                    className="bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white"
                >
                    <option value="name">Sort by Name</option>
                    <option value="size">Sort by Size</option>
                    <option value="recent">Recently Added</option>
                </select>
            </div>

            {/* Resources Grid */}
            {sortedResources.length === 0 ? (
                <Card>
                    <div className="text-center py-12">
                        <Package size={48} className="mx-auto mb-4 opacity-30 text-gray-500" />
                        <p className="text-gray-400 mb-4">No resources yet</p>
                        <Button onClick={() => setShowHFModal(true)} leftIcon={<Download size={16} />}>
                            Download Your First Model
                        </Button>
                    </div>
                </Card>
            ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                    {sortedResources.map((resource, idx) => (
                        <Card
                            key={idx}
                            hoverable
                            className="group relative"
                        >
                            <div className="flex items-start gap-3">
                                <div className="flex-shrink-0 mt-1">
                                    {getResourceIcon(resource.type)}
                                </div>

                                <div className="flex-1 min-w-0">
                                    <h3 className="font-medium text-white text-sm mb-1 truncate" title={resource.name}>
                                        {resource.name}
                                    </h3>
                                    <p className="text-xs text-gray-400 mb-2">{resource.size}</p>

                                    <div className="flex flex-wrap gap-1">
                                        {resource.quantization && (
                                            <span className="inline-block px-2 py-0.5 bg-blue-500/20 text-blue-300 text-xs rounded">
                                                {resource.quantization}
                                            </span>
                                        )}
                                        {resource.has_vision && (
                                            <span className="inline-block px-2 py-0.5 bg-green-500/20 text-green-300 text-xs rounded flex items-center gap-1">
                                                <Eye size={10} /> Vision
                                            </span>
                                        )}
                                    </div>
                                </div>

                                <button
                                    onClick={() => handleDelete(resource)}
                                    className="opacity-0 group-hover:opacity-100 transition-opacity p-1.5 hover:bg-red-500/20 rounded text-red-400"
                                    title="Delete"
                                >
                                    <Trash2 size={16} />
                                </button>
                            </div>
                        </Card>
                    ))}
                </div>
            )}

            {/* HuggingFace Search Modal */}
            {showHFModal && (
                <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md animate-fade-in">
                    <div
                        className="w-full max-w-4xl max-h-[85vh] rounded-xl overflow-hidden flex flex-col"
                        style={{
                            background: 'linear-gradient(135deg, rgba(18, 18, 22, 0.98) 0%, rgba(20, 18, 24, 0.98) 100%)',
                            border: '1px solid rgba(167, 139, 250, 0.3)',
                            boxShadow: '0 20px 60px rgba(0, 0, 0, 0.5)'
                        }}
                    >
                        {/* Modal Header */}
                        <div className="p-6 border-b border-white/10">
                            <div className="flex justify-between items-start mb-4">
                                <div>
                                    <h2 className="text-2xl font-bold text-white mb-1">HuggingFace Hub</h2>
                                    <p className="text-gray-400 text-sm">Download models and datasets</p>
                                </div>
                                <button
                                    onClick={() => setShowHFModal(false)}
                                    className="p-2 hover:bg-white/10 rounded-lg transition-colors text-gray-400 hover:text-white"
                                >
                                    <X size={20} />
                                </button>
                            </div>

                            {/* Search Type Toggle */}
                            <div className="flex gap-2 mb-4">
                                <button
                                    onClick={() => setHfSearchType('model')}
                                    className={`flex-1 py-2 px-4 rounded-lg font-medium transition-all ${hfSearchType === 'model'
                                        ? 'bg-gradient-to-r from-purple-500 to-cyan-500 text-white'
                                        : 'bg-white/5 text-gray-400 hover:bg-white/10'
                                        }`}
                                >
                                    <Brain size={16} className="inline mr-2" />
                                    Models
                                </button>
                                <button
                                    onClick={() => setHfSearchType('dataset')}
                                    className={`flex-1 py-2 px-4 rounded-lg font-medium transition-all ${hfSearchType === 'dataset'
                                        ? 'bg-gradient-to-r from-purple-500 to-cyan-500 text-white'
                                        : 'bg-white/5 text-gray-400 hover:bg-white/10'
                                        }`}
                                >
                                    <DatabaseIcon size={16} className="inline mr-2" />
                                    Datasets
                                </button>
                            </div>

                            {/* Search Bar */}
                            <div className="flex gap-2">
                                <div className="flex-1 relative">
                                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" size={18} />
                                    <input
                                        type="text"
                                        value={hfSearchQuery}
                                        onChange={e => setHfSearchQuery(e.target.value)}
                                        onKeyDown={e => e.key === 'Enter' && handleHFSearch()}
                                        placeholder={`Search ${hfSearchType}s... (e.g., ${hfSearchType === 'model' ? 'meta-llama/Llama-2-7b' : 'databricks/dolly'})`}
                                        className="w-full pl-10 pr-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500/50"
                                    />
                                </div>
                                <Button
                                    onClick={handleHFSearch}
                                    isLoading={isSearching}
                                    variant="primary"
                                    className="px-6"
                                >
                                    Search
                                </Button>
                            </div>

                            {/* Token Input */}
                            {hfToken ? (
                                <div className="mt-3 text-xs text-green-400 flex items-center gap-2">
                                    <span className="w-2 h-2 bg-green-400 rounded-full"></span>
                                    HuggingFace token configured
                                </div>
                            ) : (
                                <div className="mt-3">
                                    <Input
                                        type="password"
                                        value={hfToken}
                                        onChange={e => setHfToken(e.target.value)}
                                        placeholder="HuggingFace token (optional, for gated models)"
                                        className="text-sm"
                                    />
                                </div>
                            )}
                        </div>

                        {/* Search Results */}
                        <div className="flex-1 overflow-y-auto p-6">
                            {isSearching ? (
                                <div className="flex items-center justify-center py-12">
                                    <div className="text-center">
                                        <div className="w-12 h-12 border-4 border-purple-500/30 border-t-purple-500 rounded-full animate-spin mx-auto mb-4"></div>
                                        <p className="text-gray-400">Searching HuggingFace...</p>
                                    </div>
                                </div>
                            ) : hfSearchResults.length === 0 ? (
                                <div className="text-center py-12 text-gray-500">
                                    <Search size={48} className="mx-auto mb-4 opacity-30" />
                                    <p>Enter a search query to find {hfSearchType}s</p>
                                    <p className="text-sm mt-2">Try searching for popular {hfSearchType}s like "{hfSearchType === 'model' ? 'llama' : 'alpaca'}"</p>
                                </div>
                            ) : (
                                <div className="space-y-3">
                                    {hfSearchResults.map((result, idx) => (
                                        <div
                                            key={idx}
                                            className="p-4 rounded-lg border border-white/10 hover:border-purple-500/50 transition-all cursor-pointer group"
                                            style={{
                                                background: 'linear-gradient(135deg, rgba(28, 28, 36, 0.4) 0%, rgba(30, 28, 38, 0.4) 100%)'
                                            }}
                                            onClick={() => handleSelectRepo(result.id)}
                                        >
                                            <div className="flex justify-between items-start">
                                                <div className="flex-1">
                                                    <h3 className="font-medium text-white group-hover:text-purple-300 transition-colors">{result.name || result.id}</h3>
                                                    <p className="text-sm text-gray-400 mt-1">{result.id}</p>
                                                    {result.tags && result.tags.length > 0 && (
                                                        <div className="flex flex-wrap gap-1 mt-2">
                                                            {result.tags.slice(0, 5).map((tag, i) => (
                                                                <span key={i} className="px-2 py-0.5 bg-white/5 text-gray-400 text-xs rounded">
                                                                    {tag}
                                                                </span>
                                                            ))}
                                                        </div>
                                                    )}
                                                </div>
                                                <div className="flex flex-col items-end gap-2 ml-4">
                                                    <div className="flex items-center gap-4 text-sm text-gray-400">
                                                        <span className="flex items-center gap-1">
                                                            <TrendingDown size={14} />
                                                            {result.downloads.toLocaleString()}
                                                        </span>
                                                        <span className="flex items-center gap-1">
                                                            <Star size={14} />
                                                            {result.likes}
                                                        </span>
                                                    </div>
                                                    <div className="text-xs text-purple-400 opacity-0 group-hover:opacity-100 transition-opacity">
                                                        Click to download →
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            )}

            {/* File Selector Modal */}
            {showFileSelector && selectedRepo && (
                <Modal
                    isOpen={showFileSelector}
                    onClose={() => setShowFileSelector(false)}
                    title={`Select Files: ${selectedRepo}`}
                    footer={
                        <>
                            <Button variant="ghost" onClick={() => setShowFileSelector(false)}>
                                Cancel
                            </Button>
                            <Button
                                variant="primary"
                                onClick={handleDownload}
                                disabled={selectedFiles.size === 0}
                                leftIcon={<Download size={16} />}
                            >
                                Download {selectedFiles.size} {selectedFiles.size === 1 ? 'File' : 'Files'}
                            </Button>
                        </>
                    }
                >
                    {isLoadingFiles ? (
                        <div className="flex items-center justify-center py-8">
                            <div className="w-8 h-8 border-4 border-purple-500/30 border-t-purple-500 rounded-full animate-spin"></div>
                        </div>
                    ) : (
                        <div className="space-y-2 max-h-96 overflow-y-auto">
                            <div className="flex justify-between items-center mb-3 pb-2 border-b border-white/10">
                                <span className="text-sm text-gray-400">{repoFiles.length} files available</span>
                                <button
                                    onClick={() => {
                                        if (selectedFiles.size === repoFiles.length) {
                                            setSelectedFiles(new Set());
                                        } else {
                                            setSelectedFiles(new Set(repoFiles.map(f => f.path)));
                                        }
                                    }}
                                    className="text-sm text-purple-400 hover:text-purple-300"
                                >
                                    {selectedFiles.size === repoFiles.length ? 'Deselect All' : 'Select All'}
                                </button>
                            </div>

                            {repoFiles.map((file, idx) => (
                                <label
                                    key={idx}
                                    className="flex items-center gap-3 p-3 rounded-lg hover:bg-white/5 cursor-pointer transition-colors"
                                >
                                    <input
                                        type="checkbox"
                                        checked={selectedFiles.has(file.path)}
                                        onChange={e => {
                                            const newSet = new Set(selectedFiles);
                                            if (e.target.checked) {
                                                newSet.add(file.path);
                                            } else {
                                                newSet.delete(file.path);
                                            }
                                            setSelectedFiles(newSet);
                                        }}
                                        className="w-4 h-4 accent-purple-500"
                                    />
                                    <div className="flex-1 min-w-0">
                                        <p className="text-sm text-white truncate">{file.path}</p>
                                        {file.size && (
                                            <p className="text-xs text-gray-500">
                                                {(file.size / 1024 / 1024).toFixed(2)} MB
                                            </p>
                                        )}
                                    </div>
                                </label>
                            ))}
                        </div>
                    )}
                </Modal>
            )}
        </div>
    );
};

export default ResourceDashboardNew;
