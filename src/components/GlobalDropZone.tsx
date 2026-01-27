import React, { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { Package, Database, Layers, HelpCircle, Upload } from 'lucide-react';
import '../styles/DropZone.css';

interface DropDetails {
    is_split_model: boolean;
    has_vision: boolean;
    file_count: number;
    total_size: string;
    detected_format: string | null;
    shard_info: string | null;
}

interface DropAnalysis {
    path: string;
    name: string;
    resource_type: string;
    confidence: string;
    details: DropDetails;
}

interface GlobalDropZoneProps {
    onImportComplete: () => void;
    addNotification: (message: string, type?: 'success' | 'error' | 'info') => void;
}

export const GlobalDropZone: React.FC<GlobalDropZoneProps> = ({ onImportComplete, addNotification }) => {
    const [isHovering, setIsHovering] = useState(false);
    const [analysisResults, setAnalysisResults] = useState<DropAnalysis[]>([]);
    const [isAnalyzing, setIsAnalyzing] = useState(false);

    // Listen for file drops
    useEffect(() => {
        let unlistenHover: (() => void) | undefined;
        let unlistenCancelled: (() => void) | undefined;
        let unlistenDrop: (() => void) | undefined;

        const setupListeners = async () => {
            // Show overlay on hover
            unlistenHover = await listen<{ paths: string[] }>('tauri://drag-over', async (event) => {
                if (!isHovering) {
                    setIsHovering(true);
                    setIsAnalyzing(true);

                    // Analyze files while hovering
                    try {
                        const paths = event.payload.paths;
                        if (paths && paths.length > 0) {
                            const analysis: DropAnalysis[] = await invoke('analyze_drop_command', { paths });
                            setAnalysisResults(analysis);
                        }
                    } catch (e) {
                        console.error('Drop analysis failed:', e);
                    } finally {
                        setIsAnalyzing(false);
                    }
                }
            });

            // Hide on cancel
            unlistenCancelled = await listen('tauri://drag-cancelled', () => {
                setIsHovering(false);
                setAnalysisResults([]);
            });

            // Also hide on drag leave
            await listen('tauri://drag-leave', () => {
                setIsHovering(false);
                setAnalysisResults([]);
            });

            // Process drop
            unlistenDrop = await listen<{ paths: string[] }>('tauri://drag-drop', async (event) => {
                setIsHovering(false);
                const paths = event.payload.paths;
                if (!paths || paths.length === 0) return;

                // Use cached analysis if available, otherwise analyze now
                let toImport = analysisResults;
                if (toImport.length === 0) {
                    try {
                        toImport = await invoke('analyze_drop_command', { paths });
                    } catch (e) {
                        addNotification(`Analysis failed: ${e}`, 'error');
                        return;
                    }
                }

                // Import each item
                for (const item of toImport) {
                    if (item.resource_type === 'unknown') {
                        addNotification(`Skipped unknown file: ${item.name}`, 'info');
                        continue;
                    }

                    try {
                        // Map resource_type to import type
                        let importType = item.resource_type;
                        if (importType === 'mmproj') importType = 'gguf'; // mmproj goes to models

                        addNotification(`Importing: ${item.name}...`, 'info');
                        await invoke('import_resource_command', {
                            resourceType: importType,
                            sourcePath: item.path
                        });

                        const typeLabel = item.details.has_vision ? 'Vision Model' :
                            item.details.is_split_model ? 'Split Model' :
                                item.resource_type.charAt(0).toUpperCase() + item.resource_type.slice(1);
                        addNotification(`Imported ${typeLabel}: ${item.name}`, 'success');
                    } catch (e) {
                        addNotification(`Import failed: ${e}`, 'error');
                    }
                }

                setAnalysisResults([]);
                onImportComplete();
            });
        };

        setupListeners();

        return () => {
            if (unlistenHover) unlistenHover();
            if (unlistenCancelled) unlistenCancelled();
            if (unlistenDrop) unlistenDrop();
        };
    }, [addNotification, onImportComplete, isHovering, analysisResults]);

    const getResourceIcon = (type: string, hasVision: boolean) => {
        if (type === 'dataset') return <Database size={48} />;
        if (type === 'lora') return <Layers size={48} />;
        if (type === 'gguf' || type === 'model' || type === 'mmproj') {
            return hasVision ? (
                <div className="icon-stack">
                    <Package size={48} />
                    <span className="vision-badge">👁</span>
                </div>
            ) : <Package size={48} />;
        }
        return <HelpCircle size={48} />;
    };

    const getResourceLabel = (item: DropAnalysis) => {
        if (item.details.is_split_model && item.details.shard_info) {
            return `Split Model (${item.details.shard_info})`;
        }
        if (item.details.has_vision) {
            return 'Vision Model';
        }
        switch (item.resource_type) {
            case 'gguf': return 'GGUF Model';
            case 'model': return 'Model';
            case 'dataset': return 'Dataset';
            case 'lora': return 'LoRA Adapter';
            case 'mmproj': return 'Vision Projector';
            default: return 'Unknown';
        }
    };

    if (!isHovering) return null;

    return (
        <div className="global-drop-zone">
            <div className="drop-zone-content">
                <div className="drop-zone-icon">
                    <Upload size={64} className="upload-icon" />
                </div>

                <h2 className="drop-zone-title">Drop to Import</h2>

                {isAnalyzing ? (
                    <p className="drop-zone-subtitle">Analyzing files...</p>
                ) : analysisResults.length > 0 ? (
                    <div className="drop-zone-items">
                        {analysisResults.slice(0, 5).map((item, index) => (
                            <div key={index} className={`drop-item ${item.resource_type}`}>
                                <div className="drop-item-icon">
                                    {getResourceIcon(item.resource_type, item.details.has_vision)}
                                </div>
                                <div className="drop-item-info">
                                    <span className="drop-item-name">{item.name}</span>
                                    <span className="drop-item-type">{getResourceLabel(item)}</span>
                                    <span className="drop-item-size">{item.details.total_size}</span>
                                </div>
                                <span className={`confidence-badge ${item.confidence}`}>
                                    {item.confidence === 'high' ? '✓' : item.confidence === 'medium' ? '?' : '!'}
                                </span>
                            </div>
                        ))}
                        {analysisResults.length > 5 && (
                            <p className="more-items">+{analysisResults.length - 5} more items</p>
                        )}
                    </div>
                ) : (
                    <p className="drop-zone-subtitle">Drop files or folders here</p>
                )}
            </div>
        </div>
    );
};

export default GlobalDropZone;
