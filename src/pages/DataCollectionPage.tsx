import React, { useState, useEffect, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { open } from '@tauri-apps/plugin-dialog';
import { useApp } from '../context/AppContext';
import { VirtualizedDataTable, ColumnDef, DataRow } from '../components/VirtualizedDataTable';
import {
    Database, FolderOpen, Search, Plus,
    FileText, Table, Loader2, RefreshCw,
    Sparkles, Edit3, Save, RotateCcw, Trash2
} from 'lucide-react';
import { RecipeBuilder } from '../components/generation/RecipeBuilder';

interface DataCollectionPageProps {
    addLogMessage: (message: string) => void;
}

interface DatasetInfo {
    name: string;
    path: string;
    format: string;
    rowCount?: number;
}

const DataCollectionPage: React.FC<DataCollectionPageProps> = ({ addLogMessage }) => {
    // Dataset list from resources
    const {
        resources, loadResources
    } = useApp();

    const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
    const [selectedDataset, setSelectedDataset] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [searchQuery, setSearchQuery] = useState('');

    // Dataset preview state
    const [previewData, setPreviewData] = useState<DataRow[]>([]);
    const [previewColumns, setPreviewColumns] = useState<ColumnDef[]>([]);
    const [totalRows, setTotalRows] = useState(0);
    const [previewError, setPreviewError] = useState<string | null>(null);
    const [activeTab, setActiveTab] = useState<'view' | 'edit' | 'generate'>('view');
    const [isSaving, setIsSaving] = useState(false);

    // Filter/Sort
    const [sortColumn, setSortColumn] = useState<string | null>(null);
    const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');

    const [edits, setEdits] = useState<Record<number, DataRow>>({});
    const [detailRow, setDetailRow] = useState<DataRow | null>(null);

    // Load datasets from resources
    useEffect(() => {
        const datasetResources = resources.filter(r => r.type === 'dataset');
        const datasetInfos: DatasetInfo[] = datasetResources.map(r => ({
            name: r.name,
            path: r.path,
            format: (r as any).format || 'unknown'
        }));
        setDatasets(datasetInfos);
    }, [resources]);

    // Load dataset preview
    const loadDatasetPreview = useCallback(async (path: string) => {
        setIsLoading(true);
        setPreviewError(null);
        try {
            const result: any = await invoke('load_dataset_preview_command', {
                datasetPath: path,
                offset: 0,
                limit: 100,
                split: 'train'
            });

            if (result.error) {
                setPreviewError(result.error);
                setPreviewData([]);
                setPreviewColumns([]);
                setTotalRows(0);
            } else {
                setPreviewData(result.rows);
                setTotalRows(result.totalCount ?? result.total_count ?? 0);
                setEdits({}); // Clear edits on load

                // Build columns from schema
                const cols: ColumnDef[] = result.columns.map((key: string) => ({
                    key,
                    label: key,
                    type: 'auto' as const,
                    // width removed to allow flex
                }));
                setPreviewColumns(cols);
            }
        } catch (e) {
            setPreviewError(String(e));
            addLogMessage(`Error loading dataset: ${e}`);
        }
        setIsLoading(false);
    }, [addLogMessage]);

    // Handlers
    const handleColumnResize = useCallback((columnKey: string, newWidth: number) => {
        setPreviewColumns(prev => prev.map(col =>
            col.key === columnKey ? { ...col, width: newWidth } : col
        ));
    }, []);

    const handleColumnSort = useCallback((columnKey: string) => {
        if (sortColumn === columnKey) {
            setSortDirection(prev => prev === 'asc' ? 'desc' : 'asc');
        } else {
            setSortColumn(columnKey);
            setSortDirection('asc');
        }
    }, [sortColumn]);

    const handleCellChange = useCallback((rowIndex: number, columnKey: string, newValue: any) => {
        setEdits(prev => ({
            ...prev,
            [rowIndex]: {
                ...(prev[rowIndex] || {}),
                [columnKey]: newValue
            }
        }));
    }, []);

    const handleSaveEdits = async () => {
        if (!selectedDataset) return;
        setIsSaving(true);
        try {
            // Convert edits map to array for backend
            const editList = Object.entries(edits).map(([rowIndex, rowEdits]) => ({
                rowIndex: parseInt(rowIndex),
                data: {
                    ...previewData[parseInt(rowIndex)],
                    ...rowEdits
                }
            }));

            const result: any = await invoke('apply_dataset_edits_command', {
                dataset_path: selectedDataset,
                edits: JSON.stringify(editList)
            });

            if (result.success) {
                addLogMessage(`Saved ${result.count} edits to ${selectedDataset}`);
                setEdits({});
                // Reload preview to confirm
                loadDatasetPreview(selectedDataset);
            } else {
                addLogMessage(`Error saving edits: ${result.error}`);
            }
        } catch (e) {
            addLogMessage(`Failed to save edits: ${e}`);
        }
        setIsSaving(false);
    };

    const handleImport = async () => {
        try {
            const selected = await open({
                multiple: false,
                filters: [{ name: 'Datasets', extensions: ['jsonl', 'json', 'csv', 'parquet'] }]
            });
            if (selected) {
                addLogMessage(`Importing dataset: ${selected}`);
                setSelectedDataset(selected as string);
                loadDatasetPreview(selected as string);
            }
        } catch (e) {
            addLogMessage(`Import failed: ${e}`);
        }
    };

    const handleNewDataset = async () => {
        console.log("handleNewDataset called");
        addLogMessage("Initiating new dataset creation...");
        try {
            // Generate a friendlier name, e.g. "Untitled Dataset 1"
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            const name = `Untitled-Dataset-${timestamp}`;

            console.log("Invoking create_dataset_command with name:", name);
            const newPath: string = await invoke('create_dataset_command', { name });
            console.log("create_dataset_command returned:", newPath);

            addLogMessage("Dataset created at: " + newPath);

            console.log("Reloading resources...");
            await loadResources();
            console.log("Resources reloaded");

            console.log("Selecting dataset:", newPath);
            setSelectedDataset(newPath);
            setActiveTab('generate');
        } catch (e) {
            console.error("handleNewDataset error:", e);
            addLogMessage(`Failed to create dataset: ${e}`);
        }
    };

    const handleRunRecipe = async (recipe: any) => {
        addLogMessage(`Starting generation: ${recipe.name}`);
        try {
            const result: any = await invoke('generate_dataset_command', {
                recipe: JSON.stringify(recipe),
                output_path: selectedDataset || 'generated_dataset.jsonl'
            });
            if (result.success) {
                addLogMessage(`Generation complete! Saved to ${result.output_path}`);
                loadResources();
                if (selectedDataset) loadDatasetPreview(selectedDataset);
            } else {
                addLogMessage(`Generation failed: ${result.error}`);
            }
        } catch (e) {
            addLogMessage(`Failed to run recipe: ${e}`);
        }
    };

    const handleFixDataset = async () => {
        if (!selectedDataset) return;
        const confirm = await window.confirm('This will scan the dataset and automatically remove invalid images and fix structure issues. A backup will be created. Continue?');
        if (!confirm) return;

        addLogMessage(`Scanning and fixing ${selectedDataset}...`);
        setIsLoading(true);
        try {
            const result: any = await invoke('fix_dataset_command', {
                datasetDir: selectedDataset.replace('dataset.jsonl', '').replace(/\\$/, '') // passing directory
            });

            if (result.success) {
                addLogMessage(`Fix complete! Output: ${result.output_path}`);
                loadDatasetPreview(result.output_path || selectedDataset);
            } else {
                addLogMessage(`Fix failed: ${result.error}`);
            }
        } catch (e) {
            addLogMessage(`Error during auto-fix: ${e}`);
        }
        setIsLoading(false);
    };

    const [bulkEditConfig, setBulkEditConfig] = useState({
        column: 'instruction',
        find: '',
        replace: '',
        isRegex: false
    });

    const handleBulkEdit = async () => {
        if (!selectedDataset) return;
        if (!bulkEditConfig.column || !bulkEditConfig.find) {
            addLogMessage("Please specify column and find pattern");
            return;
        }

        setIsSaving(true);
        addLogMessage(`Running bulk replace on ${bulkEditConfig.column}...`);

        try {
            const operation = {
                type: 'replace',
                column: bulkEditConfig.column,
                find: bulkEditConfig.find,
                replace: bulkEditConfig.replace,
                is_regex: bulkEditConfig.isRegex
            };

            const result: any = await invoke('bulk_edit_dataset_command', {
                datasetPath: selectedDataset,
                operation: JSON.stringify(operation)
            });

            if (result.success) {
                addLogMessage(`Bulk edit complete! Modified ${result.count} rows.`);
                loadDatasetPreview(selectedDataset);
            } else {
                addLogMessage(`Bulk edit failed: ${result.error}`);
            }
        } catch (e) {
            addLogMessage(`Error executing bulk edit: ${e}`);
        }
        setIsSaving(false);
    };

    // Derived sorted data with edits applied
    const displayedData = React.useMemo(() => {
        const dataWithEdits = previewData.map((row, index) => {
            if (edits[index]) {
                return { ...row, ...edits[index] };
            }
            return row;
        });

        if (!sortColumn) return dataWithEdits;

        return [...dataWithEdits].sort((a, b) => {
            const aVal = a[sortColumn];
            const bVal = b[sortColumn];

            if (aVal === bVal) return 0;
            if (aVal === null || aVal === undefined) return 1;
            if (bVal === null || bVal === undefined) return -1;

            const comparison = aVal < bVal ? -1 : 1;
            return sortDirection === 'asc' ? comparison : -comparison;
        });
    }, [previewData, edits, sortColumn, sortDirection]);

    // Load preview when dataset changes
    useEffect(() => {
        if (selectedDataset) {
            loadDatasetPreview(selectedDataset);
        }
    }, [selectedDataset, loadDatasetPreview]);

    const [hoveredDataset, setHoveredDataset] = useState<string | null>(null);

    const handleRenameDataset = async (dataset: DatasetInfo, e: React.MouseEvent) => {
        e.stopPropagation();
        const newName = window.prompt("Rename dataset to:", dataset.name);
        if (!newName || newName === dataset.name) return;

        try {
            await invoke('rename_resource_command', {
                resourceType: 'dataset',
                currentName: dataset.name,
                newName: newName
            });
            addLogMessage(`Renamed ${dataset.name} to ${newName}`);
            loadResources();
            if (selectedDataset === dataset.path) {
                // Approximate new path, though loadResources will refresh list
                setSelectedDataset(dataset.path.replace(dataset.name, newName));
            }
        } catch (err) {
            addLogMessage(`Rename failed: ${err}`);
        }
    };

    const handleDeleteDataset = async (dataset: DatasetInfo, e: React.MouseEvent) => {
        e.stopPropagation();

        try {
            await invoke('delete_resource_command', {
                resourceType: 'dataset',
                resourcePath: dataset.name // Backend expects filename/foldername relative to datasets dir
            });
            addLogMessage(`Deleted dataset: ${dataset.name}`);
            if (selectedDataset === dataset.path) {
                setSelectedDataset(null);
            }
            loadResources();
        } catch (err) {
            addLogMessage(`Delete failed: ${err}`);
        }
    };

    const filteredDatasets = datasets.filter(d =>
        d.name.toLowerCase().includes(searchQuery.toLowerCase())
    );

    return (
        <div style={{ display: 'flex', height: 'calc(100vh - 100px)', overflow: 'hidden' }}>
            {/* Left Sidebar - Dataset Browser */}
            <div style={{
                width: '280px',
                flexShrink: 0,
                background: 'var(--bg-surface, #121216)',
                borderRight: '1px solid var(--border-subtle, rgba(255,255,255,0.06))',
                display: 'flex',
                flexDirection: 'column'
            }}>
                {/* Sidebar Header */}
                <div style={{
                    padding: '16px',
                    borderBottom: '1px solid var(--border-subtle, rgba(255,255,255,0.06))'
                }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
                        <Database size={20} style={{ color: 'var(--accent-primary)' }} />
                        <h3 style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-main)' }}>Datasets</h3>
                        <button
                            onClick={() => loadResources()}
                            style={{
                                marginLeft: 'auto',
                                background: 'none',
                                border: 'none',
                                color: 'var(--text-muted)',
                                cursor: 'pointer',
                                padding: '4px'
                            }}
                            title="Refresh"
                        >
                            <RefreshCw size={14} />
                        </button>
                    </div>
                    <div style={{ position: 'relative' }}>
                        <Search size={14} style={{
                            position: 'absolute',
                            left: '10px',
                            top: '50%',
                            transform: 'translateY(-50%)',
                            color: 'var(--text-muted)'
                        }} />
                        <input
                            type="text"
                            placeholder="Search datasets..."
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            style={{
                                width: '100%',
                                padding: '8px 12px 8px 32px',
                                background: 'rgba(255,255,255,0.05)',
                                border: '1px solid rgba(255,255,255,0.08)',
                                borderRadius: '8px',
                                color: 'var(--text-main)',
                                fontSize: '13px'
                            }}
                        />
                    </div>
                </div>

                {/* Dataset List */}
                <div style={{ flex: 1, overflowY: 'auto', padding: '8px' }}>
                    {filteredDatasets.length === 0 ? (
                        <div style={{
                            padding: '24px 16px',
                            textAlign: 'center',
                            color: 'var(--text-muted)'
                        }}>
                            <FileText size={32} style={{ opacity: 0.3, marginBottom: '12px' }} />
                            <p style={{ fontSize: '13px' }}>No datasets found</p>
                            <p style={{ fontSize: '11px', opacity: 0.7 }}>Download or import datasets to get started</p>
                        </div>
                    ) : (
                        filteredDatasets.map((dataset) => (
                            <button
                                key={dataset.path}
                                onClick={() => setSelectedDataset(dataset.path)}
                                onMouseEnter={() => setHoveredDataset(dataset.path)}
                                onMouseLeave={() => setHoveredDataset(null)}
                                style={{
                                    width: '100%',
                                    padding: '10px 12px',
                                    background: selectedDataset === dataset.path
                                        ? 'rgba(59, 130, 246, 0.15)'
                                        : 'transparent',
                                    border: selectedDataset === dataset.path
                                        ? '1px solid rgba(59, 130, 246, 0.3)'
                                        : '1px solid transparent',
                                    borderRadius: '8px',
                                    textAlign: 'left',
                                    cursor: 'pointer',
                                    marginBottom: '4px',
                                    transition: 'all 0.15s',
                                    position: 'relative'
                                }}
                            >
                                <div style={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: '8px',
                                    color: selectedDataset === dataset.path ? 'var(--accent-primary)' : 'var(--text-main)',
                                    paddingRight: '40px' // Space for buttons
                                }}>
                                    <Table size={14} />
                                    <span style={{
                                        fontSize: '13px',
                                        fontWeight: 500,
                                        overflow: 'hidden',
                                        textOverflow: 'ellipsis',
                                        whiteSpace: 'nowrap'
                                    }}>
                                        {dataset.name}
                                    </span>
                                </div>
                                <div style={{
                                    fontSize: '11px',
                                    color: 'var(--text-muted)',
                                    marginTop: '4px',
                                    marginLeft: '22px'
                                }}>
                                    {dataset.format.toUpperCase()}
                                </div>

                                {hoveredDataset === dataset.path && (
                                    <div style={{
                                        position: 'absolute',
                                        right: '8px',
                                        top: '50%',
                                        transform: 'translateY(-50%)',
                                        display: 'flex',
                                        gap: '4px'
                                    }}>
                                        <div
                                            role="button"
                                            onClick={(e) => handleRenameDataset(dataset, e)}
                                            style={{
                                                padding: '4px',
                                                borderRadius: '4px',
                                                background: 'rgba(255,255,255,0.1)',
                                                color: 'var(--text-main)',
                                                cursor: 'pointer'
                                            }}
                                            title="Rename"
                                        >
                                            <Edit3 size={12} />
                                        </div>
                                        <div
                                            role="button"
                                            onClick={(e) => handleDeleteDataset(dataset, e)}
                                            style={{
                                                padding: '4px',
                                                borderRadius: '4px',
                                                background: 'rgba(239, 68, 68, 0.2)',
                                                color: '#ef4444',
                                                cursor: 'pointer'
                                            }}
                                            title="Delete"
                                        >
                                            <Trash2 size={12} />
                                        </div>
                                    </div>
                                )}
                            </button>
                        ))
                    )}
                </div>

                {/* Action Buttons */}
                <div style={{
                    padding: '12px',
                    borderTop: '1px solid var(--border-subtle, rgba(255,255,255,0.06))',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '8px'
                }}>
                    <button
                        onClick={() => {
                            console.log("Sidebar 'New Dataset' clicked");
                            handleNewDataset();
                        }}
                        style={{
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            gap: '8px',
                            padding: '10px',
                            background: 'var(--accent-gradient, linear-gradient(135deg, #3b82f6, #8b5cf6))',
                            border: 'none',
                            borderRadius: '8px',
                            color: 'white',
                            fontSize: '13px',
                            fontWeight: 600,
                            cursor: 'pointer'
                        }}
                    >
                        <Plus size={16} />
                        New Dataset
                    </button>
                    <button
                        onClick={handleImport}
                        style={{
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            gap: '8px',
                            padding: '10px',
                            background: 'rgba(255,255,255,0.05)',
                            border: '1px solid rgba(255,255,255,0.1)',
                            borderRadius: '8px',
                            color: 'var(--text-main)',
                            fontSize: '13px',
                            fontWeight: 500,
                            cursor: 'pointer'
                        }}
                    >
                        <FolderOpen size={16} />
                        Import
                    </button>
                    {selectedDataset && (
                        <button
                            onClick={handleFixDataset}
                            disabled={isLoading}
                            style={{
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                gap: '8px',
                                padding: '10px',
                                background: 'rgba(239, 68, 68, 0.1)',
                                border: '1px solid rgba(239, 68, 68, 0.3)',
                                borderRadius: '8px',
                                color: 'var(--accent-red, #ef4444)',
                                fontSize: '13px',
                                fontWeight: 500,
                                cursor: 'pointer'
                            }}
                        >
                            <Sparkles size={16} />
                            Auto-Fix Dataset
                        </button>
                    )}
                </div>
            </div>

            {/* Main Content Area */}
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
                {/* Tab Bar */}
                <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    padding: '0 16px',
                    borderBottom: '1px solid var(--border-subtle, rgba(255,255,255,0.06))',
                    background: 'var(--bg-surface, #121216)'
                }}>
                    {[
                        { id: 'view', label: 'View', icon: <Table size={14} /> },
                        { id: 'edit', label: 'Edit', icon: <Edit3 size={14} /> },
                        { id: 'generate', label: 'Generate', icon: <Sparkles size={14} /> }
                    ].map((tab) => (
                        <button
                            key={tab.id}
                            onClick={() => setActiveTab(tab.id as 'view' | 'edit' | 'generate')}
                            style={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: '6px',
                                padding: '14px 16px',
                                background: 'none',
                                border: 'none',
                                borderBottom: activeTab === tab.id
                                    ? '2px solid var(--accent-primary)'
                                    : '2px solid transparent',
                                color: activeTab === tab.id ? 'var(--accent-primary)' : 'var(--text-muted)',
                                fontSize: '13px',
                                fontWeight: 500,
                                cursor: 'pointer',
                                transition: 'all 0.15s'
                            }}
                        >
                            {tab.icon}
                            {tab.label}
                        </button>
                    ))}

                    {/* Dataset info in header */}
                    {selectedDataset && (
                        <div style={{
                            marginLeft: 'auto',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '16px',
                            fontSize: '12px',
                            color: 'var(--text-muted)'
                        }}>
                            {Object.keys(edits).length > 0 && (
                                <>
                                    <span style={{ color: 'var(--accent-warning, #f59e0b)' }}>
                                        {Object.keys(edits).length} unsaved changes
                                    </span>
                                    <button
                                        onClick={() => setEdits({})}
                                        disabled={isSaving}
                                        style={{
                                            background: 'none',
                                            border: 'none',
                                            color: 'var(--text-muted)',
                                            cursor: 'pointer',
                                            display: 'flex',
                                            alignItems: 'center',
                                            gap: '4px'
                                        }}
                                    >
                                        <RotateCcw size={14} />
                                        Discard
                                    </button>
                                    <button
                                        onClick={handleSaveEdits}
                                        disabled={isSaving}
                                        style={{
                                            background: 'var(--accent-primary)',
                                            border: 'none',
                                            borderRadius: '4px',
                                            padding: '4px 12px',
                                            color: 'white',
                                            cursor: 'pointer',
                                            display: 'flex',
                                            alignItems: 'center',
                                            gap: '4px',
                                            fontWeight: 500
                                        }}
                                    >
                                        {isSaving ? <Loader2 size={14} className="animate-spin" /> : <Save size={14} />}
                                        Save
                                    </button>
                                    <div style={{ width: '1px', height: '16px', background: 'var(--border-subtle, rgba(255,255,255,0.1))' }} />
                                </>
                            )}
                            <span>{totalRows.toLocaleString()} rows</span>
                            <span>{previewColumns.length} columns</span>
                        </div>
                    )}
                </div>

                {/* Content */}
                <div style={{ flex: 1, overflow: 'hidden', padding: '16px' }}>
                    {isLoading ? (
                        <div style={{
                            display: 'flex',
                            flexDirection: 'column',
                            alignItems: 'center',
                            justifyContent: 'center',
                            height: '100%',
                            color: 'var(--text-muted)'
                        }}>
                            <Loader2 size={32} style={{ marginBottom: '12px', animation: 'spin 1s linear infinite' }} />
                            <p>Loading dataset...</p>
                        </div>
                    ) : previewError ? (
                        <div style={{
                            display: 'flex',
                            flexDirection: 'column',
                            alignItems: 'center',
                            justifyContent: 'center',
                            height: '100%',
                            color: 'var(--accent-red, #ef4444)'
                        }}>
                            <p style={{ marginBottom: '8px' }}>Error loading dataset</p>
                            <p style={{ fontSize: '12px', opacity: 0.7, maxWidth: '400px', textAlign: 'center' }}>
                                {previewError}
                            </p>
                        </div>
                    ) : !selectedDataset ? (
                        <div style={{
                            display: 'flex',
                            flexDirection: 'column',
                            alignItems: 'center',
                            justifyContent: 'center',
                            height: '100%',
                            color: 'var(--text-muted)'
                        }}>
                            <Database size={64} style={{ opacity: 0.2, marginBottom: '24px' }} />
                            <h2 style={{ fontSize: '20px', fontWeight: 600, marginBottom: '8px', color: 'var(--text-main)' }}>
                                Data Collection & Generation
                            </h2>
                            <p style={{ fontSize: '14px', maxWidth: '400px', textAlign: 'center' }}>
                                Select a dataset from the sidebar to view and edit, or create a new synthetic dataset.
                            </p>
                            <div style={{ marginTop: '24px', display: 'flex', gap: '16px' }}>
                                <button onClick={handleImport} className="btn-secondary">Import Existing</button>
                                <button onClick={() => {
                                    console.log("Empty state 'Create New' clicked");
                                    handleNewDataset();
                                }} className="btn-primary">Create New</button>
                            </div>
                        </div>
                    ) : (activeTab === 'view' && previewData.length === 0) ? (
                        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', gap: '24px' }}>
                            <div style={{ padding: '24px', background: 'rgba(255,255,255,0.03)', borderRadius: '50%' }}>
                                <Database size={48} style={{ opacity: 0.2 }} />
                            </div>
                            <div style={{ textAlign: 'center' }}>
                                <h3 style={{ fontSize: '18px', fontWeight: 600, marginBottom: '8px' }}>Dataset is Empty</h3>
                                <p style={{ color: 'var(--text-muted)', maxWidth: '300px' }}>This dataset has no rows properly parsed. You can generate synthetic data to populate it.</p>
                            </div>
                            <button
                                onClick={() => setActiveTab('generate')}
                                style={{
                                    display: 'flex', alignItems: 'center', gap: '8px',
                                    background: 'linear-gradient(135deg, var(--accent-primary), #8b5cf6)',
                                    color: 'white', border: 'none', borderRadius: '8px', padding: '12px 24px',
                                    fontSize: '14px', fontWeight: 600, cursor: 'pointer',
                                    boxShadow: '0 4px 12px rgba(99, 102, 241, 0.3)'
                                }}
                            >
                                <Sparkles size={18} fill="currentColor" /> Generate Synthetic Data
                            </button>
                        </div>
                    ) : activeTab === 'view' ? (
                        <VirtualizedDataTable
                            data={displayedData}
                            columns={previewColumns}
                            totalRows={totalRows}
                            rowHeight={80}
                            onColumnResize={handleColumnResize}
                            onColumnSort={handleColumnSort}
                            onCellClick={(r) => setDetailRow(displayedData[r])}
                            onCellChange={handleCellChange}
                            sortColumn={sortColumn || undefined}
                            sortDirection={sortDirection}
                        />
                    ) : activeTab === 'edit' ? (
                        <div style={{ maxWidth: '600px', margin: '0 auto', padding: '32px' }}>
                            <h3 style={{ fontSize: '18px', marginBottom: '24px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                                <Edit3 size={18} /> Bulk Edit
                            </h3>

                            <div className="card" style={{ background: 'var(--bg-elevated)', padding: '24px', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
                                <div style={{ marginBottom: '16px' }}>
                                    <label style={{ display: 'block', fontSize: '12px', marginBottom: '8px', color: 'var(--text-muted)' }}>Target Column</label>
                                    <select
                                        value={bulkEditConfig.column}
                                        onChange={(e) => setBulkEditConfig(prev => ({ ...prev, column: e.target.value }))}
                                        style={{ width: '100%', padding: '8px', background: 'var(--bg-input)', border: '1px solid var(--border-subtle)', color: 'var(--text-main)', borderRadius: '4px' }}
                                    >
                                        {previewColumns.map(col => <option key={col.key} value={col.key}>{col.label}</option>)}
                                    </select>
                                </div>

                                <div style={{ marginBottom: '16px' }}>
                                    <label style={{ display: 'block', fontSize: '12px', marginBottom: '8px', color: 'var(--text-muted)' }}>Find</label>
                                    <input
                                        type="text"
                                        value={bulkEditConfig.find}
                                        onChange={(e) => setBulkEditConfig(prev => ({ ...prev, find: e.target.value }))}
                                        placeholder="Text to find..."
                                        style={{ width: '100%', padding: '8px', background: 'var(--bg-input)', border: '1px solid var(--border-subtle)', color: 'var(--text-main)', borderRadius: '4px' }}
                                    />
                                </div>

                                <div style={{ marginBottom: '16px' }}>
                                    <label style={{ display: 'block', fontSize: '12px', marginBottom: '8px', color: 'var(--text-muted)' }}>Replace With</label>
                                    <input
                                        type="text"
                                        value={bulkEditConfig.replace}
                                        onChange={(e) => setBulkEditConfig(prev => ({ ...prev, replace: e.target.value }))}
                                        placeholder="Replacement text..."
                                        style={{ width: '100%', padding: '8px', background: 'var(--bg-input)', border: '1px solid var(--border-subtle)', color: 'var(--text-main)', borderRadius: '4px' }}
                                    />
                                </div>

                                <div style={{ marginBottom: '24px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                                    <input
                                        type="checkbox"
                                        id="regex-toggle"
                                        checked={bulkEditConfig.isRegex}
                                        onChange={(e) => setBulkEditConfig(prev => ({ ...prev, isRegex: e.target.checked }))}
                                    />
                                    <label htmlFor="regex-toggle" style={{ fontSize: '13px', color: 'var(--text-main)' }}>Use Regular Expressions (Regex)</label>
                                </div>

                                <button
                                    onClick={handleBulkEdit}
                                    disabled={isSaving}
                                    style={{
                                        width: '100%',
                                        padding: '10px',
                                        background: 'var(--accent-primary)',
                                        color: 'white',
                                        border: 'none',
                                        borderRadius: '4px',
                                        fontWeight: 600,
                                        cursor: isSaving ? 'wait' : 'pointer',
                                        opacity: isSaving ? 0.7 : 1
                                    }}
                                >
                                    {isSaving ? 'Processing...' : 'Run Bulk Replace'}
                                </button>
                            </div>

                            <div style={{ marginTop: '24px', fontSize: '12px', color: 'var(--text-muted)', lineHeight: 1.5 }}>
                                <p><strong>Note:</strong> Valid regex patterns required if checked. Changes are applied directly to the file.</p>
                            </div>
                        </div>
                    ) : (
                        <RecipeBuilder
                            onSave={(recipe) => addLogMessage(`Saved recipe: ${recipe.name}`)}
                            onRun={handleRunRecipe}
                        />
                    )}
                </div>
            </div>

            {/* Detail Modal */}
            {detailRow && (
                <div style={{
                    position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
                    background: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(4px)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    zIndex: 1000
                }} onClick={() => setDetailRow(null)}>
                    <div style={{
                        width: '80%', height: '80%', background: 'var(--bg-surface)',
                        borderRadius: '12px', border: '1px solid var(--border-subtle)',
                        display: 'flex', flexDirection: 'column', overflow: 'hidden',
                        boxShadow: '0 24px 48px rgba(0,0,0,0.5)'
                    }} onClick={e => e.stopPropagation()}>
                        <div style={{
                            padding: '16px 24px', borderBottom: '1px solid var(--border-subtle)',
                            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                            background: 'var(--bg-elevated)'
                        }}>
                            <h3 style={{ fontSize: '16px', fontWeight: 600 }}>Row Details</h3>
                            <button onClick={() => setDetailRow(null)} style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer' }}>
                                <Plus size={24} style={{ transform: 'rotate(45deg)' }} />
                            </button>
                        </div>
                        <div style={{ flex: 1, overflow: 'auto', padding: '24px' }}>
                            <pre style={{
                                fontSize: '13px', lineHeight: 1.5, fontFamily: 'monospace',
                                color: 'var(--text-main)', whiteSpace: 'pre-wrap'
                            }}>
                                {JSON.stringify(detailRow, null, 2)}
                            </pre>
                        </div>
                    </div>
                </div>
            )}

            <style>{`
                @keyframes spin {
                    from { transform: rotate(0deg); }
                    to { transform: rotate(360deg); }
                }
                .btn-primary {
                    padding: 8px 16px;
                    background: var(--accent-primary);
                    color: white;
                    border: none;
                    border-radius: 4px;
                    font-weight: 500;
                    cursor: pointer;
                }
                .btn-secondary {
                    padding: 8px 16px;
                    background: var(--bg-elevated);
                    color: var(--text-main);
                    border: 1px solid var(--border-subtle);
                    border-radius: 4px;
                    font-weight: 500;
                    cursor: pointer;
                }
            `}</style>
        </div>
    );
};

export default DataCollectionPage;
