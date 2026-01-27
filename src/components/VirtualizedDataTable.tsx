import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { ChevronDown, ChevronRight, Image as ImageIcon, FileText, MessageSquare, Code, ArrowUpDown } from 'lucide-react';


// Types
interface DataRow {
    [key: string]: any;
    __rowIndex?: number;
}

interface ColumnDef {
    key: string;
    label: string;
    type: 'text' | 'image' | 'audio' | 'conversation' | 'json' | 'auto';
    width?: number;
    visible?: boolean;
    editable?: boolean;
    headerStyle?: React.CSSProperties;
    cellStyle?: React.CSSProperties;
}

interface VirtualizedDataTableProps {
    data: DataRow[];
    columns: ColumnDef[];
    totalRows: number;
    rowHeight?: number;
    onLoadMore?: (offset: number, limit: number) => Promise<DataRow[]>;
    onCellClick?: (rowIndex: number, columnKey: string, value: any) => void;
    onCellChange?: (rowIndex: number, columnKey: string, newValue: any) => void;
    onColumnResize?: (columnKey: string, newWidth: number) => void;
    onColumnSort?: (columnKey: string) => void;
    sortColumn?: string;
    sortDirection?: 'asc' | 'desc';
}

// Smart type detection
const detectColumnType = (value: any): ColumnDef['type'] => {
    if (value === null || value === undefined) return 'text';
    if (typeof value === 'string') {
        if (value.startsWith('data:image') || value.match(/\.(png|jpg|jpeg|gif|webp)$/i)) return 'image';
        if (value.startsWith('data:audio') || value.match(/\.(mp3|wav|ogg|flac)$/i)) return 'audio';
        try {
            const parsed = JSON.parse(value);
            if (Array.isArray(parsed) && parsed.length > 0 && parsed[0].role) return 'conversation';
            if (typeof parsed === 'object') return 'json';
        } catch { }
        return 'text';
    }
    if (Array.isArray(value)) {
        if (value.length > 0 && value[0].role && value[0].content) return 'conversation';
        return 'json';
    }
    if (typeof value === 'object') return 'json';
    return 'text';
};

// Cell Renderers
// Editable Text Cell
const EditableTextCell: React.FC<{
    value: string;
    onChange: (val: string) => void;
    onCommit: () => void;
    autoFocus?: boolean;
}> = ({ value, onChange, onCommit, autoFocus }) => {
    return (
        <textarea
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onBlur={onCommit}
            onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    onCommit();
                }
            }}
            autoFocus={autoFocus}
            style={{
                width: '100%',
                height: '100%',
                background: 'var(--bg-elevated)',
                color: 'var(--text-main)',
                border: '1px solid var(--accent-primary)',
                outline: 'none',
                resize: 'none',
                fontFamily: 'inherit',
                padding: '8px'
            }}
        />
    );
};

const TextCell: React.FC<{
    value: string;
    expanded?: boolean;
    onToggle?: () => void;
    isEditing?: boolean;
    onEditStart?: () => void;
    onEditChange?: (val: string) => void;
    onEditCommit?: () => void;
}> = ({ value, expanded, onToggle, isEditing, onEditStart, onEditChange, onEditCommit }) => {
    if (isEditing && onEditChange && onEditCommit) {
        return <EditableTextCell value={value} onChange={onEditChange} onCommit={onEditCommit} autoFocus />;
    }

    const truncated = value?.length > 150;
    const displayValue = expanded || !truncated ? value : value?.substring(0, 150) + '...';

    return (
        <div
            style={{
                padding: '8px 12px',
                fontSize: '13px',
                lineHeight: 1.5,
                wordBreak: 'break-word',
                cursor: 'text',
                color: 'var(--text-main, #e4e4e7)',
                minHeight: '100%'
            }}
            onClick={truncated ? onToggle : undefined}
            onDoubleClick={onEditStart}
        >
            {displayValue}
            {truncated && !expanded && <span style={{ color: 'var(--accent-primary)', marginLeft: '4px' }}>more</span>}
        </div>
    );
};

const ImageCell: React.FC<{ value: string }> = ({ value }) => {
    const [error, setError] = useState(false);

    if (error || !value) {
        return (
            <div style={{ padding: '8px', display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--text-muted)' }}>
                <ImageIcon size={16} />
                <span style={{ fontSize: '12px' }}>Image</span>
            </div>
        );
    }

    return (
        <div style={{ padding: '4px' }}>
            <img
                src={value}
                alt="Cell image"
                style={{
                    maxWidth: '80px',
                    maxHeight: '60px',
                    borderRadius: '4px',
                    objectFit: 'cover',
                    cursor: 'zoom-in',
                    border: '1px solid var(--border-subtle)',
                    background: 'var(--bg-subtle)'
                }}
                onClick={() => window.open(value, '_blank')} // Simple preview for now
                onError={() => setError(true)}
                loading="lazy"
            />
        </div>
    );
};

const AudioCell: React.FC<{ value: string }> = ({ value }) => {
    return (
        <div style={{ padding: '8px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <audio controls style={{ height: '32px', maxWidth: '200px' }}>
                <source src={value} />
            </audio>
        </div>
    );
};

const ConversationCell: React.FC<{ value: any[] }> = ({ value }) => {
    const [expanded, setExpanded] = useState(false);
    const messages = Array.isArray(value) ? value : [];
    const preview = messages.slice(0, 2);

    return (
        <div style={{ padding: '8px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px' }}>
                <MessageSquare size={14} style={{ color: 'var(--accent-primary)' }} />
                <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{messages.length} messages</span>
                <button
                    onClick={() => setExpanded(!expanded)}
                    style={{
                        background: 'none',
                        border: 'none',
                        color: 'var(--accent-primary)',
                        cursor: 'pointer',
                        fontSize: '11px',
                        padding: '2px 6px'
                    }}
                >
                    {expanded ? 'Collapse' : 'Expand'}
                </button>
            </div>
            {(expanded ? messages : preview).map((msg, i) => (
                <div key={i} style={{ marginBottom: '8px' }}>
                    <span style={{ fontWeight: 600, color: msg.role === 'user' ? '#60a5fa' : '#34d399', fontSize: '12px' }}>
                        {msg.role}:
                    </span>{' '}
                    <div style={{ color: 'var(--text-main)', marginTop: '2px', whiteSpace: 'pre-wrap', fontSize: '12px' }}>
                        {(() => {
                            const content = typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content);
                            return !expanded && content.length > 150 ? content.substring(0, 150) + '...' : content;
                        })()}
                    </div>
                </div>
            ))}
        </div>
    );
};

const JSONCell: React.FC<{ value: any }> = ({ value }) => {
    const [expanded, setExpanded] = useState(false);
    const jsonStr = typeof value === 'string' ? value : JSON.stringify(value, null, 2);

    return (
        <div style={{ padding: '8px' }}>
            <div
                style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                    cursor: 'pointer',
                    color: 'var(--accent-primary)'
                }}
                onClick={() => setExpanded(!expanded)}
            >
                <Code size={14} />
                <span style={{ fontSize: '11px' }}>JSON</span>
                {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
            </div>
            {expanded && (
                <pre style={{
                    marginTop: '8px',
                    padding: '8px',
                    background: 'rgba(0,0,0,0.3)',
                    borderRadius: '4px',
                    fontSize: '11px',
                    overflow: 'auto',
                    maxHeight: '200px',
                    color: '#a3e635'
                }}>
                    {jsonStr}
                </pre>
            )}
        </div>
    );
};

// Cell dispatcher
const DataCell: React.FC<{
    value: any;
    type: ColumnDef['type'];
    isEditing?: boolean;
    onEditStart: () => void;
    onEditChange: (val: any) => void;
    onEditCommit: () => void;
}> = ({ value, type, isEditing, onEditStart, onEditChange, onEditCommit }) => {
    const detectedType = type === 'auto' ? detectColumnType(value) : type;

    switch (detectedType) {
        case 'image':
            return <ImageCell value={value} />;
        case 'audio':
            return <AudioCell value={value} />;
        case 'conversation':
            // TODO: Editable Conversation Cell
            return <ConversationCell value={value} />;
        case 'json':
            // TODO: Editable JSON Cell (could use TextCell with JSON.stringify/parse)
            // For now, treat as text if editing
            if (isEditing) {
                const jsonStr = typeof value === 'string' ? value : JSON.stringify(value, null, 2);
                return (
                    <EditableTextCell
                        value={jsonStr}
                        onChange={(v) => {
                            try {
                                onEditChange(JSON.parse(v));
                            } catch {
                                // Allow invalid JSON while typing? Maybe store raw string.
                                // For simplicity here, just pass raw if we can't parse, or handle upstream
                                onEditChange(v);
                            }
                        }}
                        onCommit={onEditCommit}
                        autoFocus
                    />
                );
            }
            return <JSONCell value={value} />;
        default:
            return (
                <TextCell
                    value={String(value ?? '')}
                    isEditing={isEditing}
                    onEditStart={onEditStart}
                    onEditChange={onEditChange}
                    onEditCommit={onEditCommit}
                />
            );
    }
};

// Main Component
export const VirtualizedDataTable: React.FC<VirtualizedDataTableProps> = ({
    data,
    columns,
    totalRows,
    rowHeight = 60,
    onLoadMore,
    onCellClick,
    onCellChange,
    onColumnResize,
    onColumnSort,
    sortColumn,
    sortDirection
}) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const [scrollTop, setScrollTop] = useState(0);
    const [containerHeight, setContainerHeight] = useState(600);
    const resizingRef = useRef<{ key: string, startX: number, startWidth: number } | null>(null);

    // Editing state
    const [editingCell, setEditingCell] = useState<{ rowIndex: number, columnKey: string } | null>(null);
    const [editValue, setEditValue] = useState<any>(null);

    // Calculate visible range
    const visibleStart = Math.floor(scrollTop / rowHeight);
    const visibleCount = Math.ceil(containerHeight / rowHeight) + 2; // Buffer
    const endIndex = Math.min(totalRows, visibleStart + visibleCount);
    // Alias for compatibility with existing render logic if needed, or update usage
    const visibleEnd = Math.min(visibleStart + visibleCount, data.length);

    // Handle scroll
    const handleScroll = useCallback((e: React.UIEvent<HTMLDivElement>) => {
        setScrollTop(e.currentTarget.scrollTop);

        // Infinite scroll trigger
        if (onLoadMore && data.length < totalRows && endIndex >= data.length - 5) {
            onLoadMore(data.length, 100);
        }
    }, [endIndex, data.length, totalRows, onLoadMore]);

    // Resize observer
    useEffect(() => {
        const container = containerRef.current;
        if (!container) return;

        const resizeObserver = new ResizeObserver((entries) => {
            for (const entry of entries) {
                setContainerHeight(entry.contentRect.height);
            }
        });

        resizeObserver.observe(container);
        return () => resizeObserver.disconnect();
    }, []);

    // Column resizing handlers
    useEffect(() => {
        const handleMouseMove = (e: MouseEvent) => {
            if (resizingRef.current && onColumnResize) {
                const diff = e.clientX - resizingRef.current.startX;
                const newWidth = Math.max(50, resizingRef.current.startWidth + diff);
                onColumnResize(resizingRef.current.key, newWidth);
            }
        };

        const handleMouseUp = () => {
            resizingRef.current = null;
            document.body.style.cursor = 'default';
        };

        document.addEventListener('mousemove', handleMouseMove);
        document.addEventListener('mouseup', handleMouseUp);
        return () => {
            document.removeEventListener('mousemove', handleMouseMove);
            document.removeEventListener('mouseup', handleMouseUp);
        };
    }, [onColumnResize]);

    const visibleColumns = useMemo(() => columns.filter(c => c.visible !== false), [columns]);

    const startResize = (e: React.MouseEvent, key: string, width: number) => {
        e.stopPropagation();
        resizingRef.current = { key, startX: e.clientX, startWidth: width };
        document.body.style.cursor = 'col-resize';
    };

    return (
        <div
            ref={containerRef}
            style={{
                height: '100%',
                overflow: 'auto',
                background: 'var(--bg-surface, #121216)',
                borderRadius: '12px',
                border: '1px solid var(--border-subtle, rgba(255,255,255,0.06))'
            }}
            onScroll={handleScroll}
        >
            {/* Header */}
            <div
                style={{
                    display: 'flex',
                    position: 'sticky',
                    top: 0,
                    zIndex: 10,
                    background: 'var(--bg-elevated, #1a1a1f)',
                    borderBottom: '1px solid var(--border-subtle, rgba(255,255,255,0.08))'
                }}
            >
                {visibleColumns.map((col) => (
                    <div
                        key={col.key}
                        style={{
                            flex: col.width ? `1 1 ${col.width}px` : (col.key === 'messages' || col.type === 'conversation' ? '3 1 300px' : '1 1 100px'),
                            minWidth: col.width || 100,
                            fontSize: '11px',
                            fontWeight: 700,
                            color: col.key === sortColumn ? 'var(--accent-primary)' : 'var(--text-muted)',
                            textTransform: 'uppercase',
                            letterSpacing: '0.05em',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '8px',
                            borderRight: '1px solid var(--border-subtle, rgba(255,255,255,0.04))',
                            position: 'relative',
                            userSelect: 'none',
                            cursor: 'pointer',
                            ...col.headerStyle
                        }}
                        onClick={() => onColumnSort?.(col.key)}
                    >
                        <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis' }}>{col.label}</span>
                        {col.key === sortColumn && (
                            <ArrowUpDown
                                size={12}
                                style={{
                                    transform: sortDirection === 'desc' ? 'rotate(180deg)' : 'none',
                                    transition: 'transform 0.2s',
                                    color: 'var(--accent-primary)'
                                }}
                            />
                        )}

                        {/* Resize Handle */}
                        <div
                            style={{
                                position: 'absolute',
                                right: 0,
                                top: 0,
                                bottom: 0,
                                width: '4px',
                                cursor: 'col-resize',
                                background: 'transparent',
                                zIndex: 2
                            }}
                            onMouseDown={(e) => startResize(e, col.key, col.width || 150)}
                            onClick={(e) => e.stopPropagation()}
                        />
                    </div>
                ))}
            </div>

            {/* Virtual Content */}
            <div style={{ height: totalRows * rowHeight, position: 'relative' }}>
                {data.slice(visibleStart, visibleEnd).map((row, i) => {
                    const actualIndex = visibleStart + i;
                    return (
                        <div
                            key={actualIndex}
                            style={{
                                display: 'flex',
                                position: 'absolute',
                                top: actualIndex * rowHeight,
                                left: 0,
                                right: 0,
                                height: rowHeight,
                                borderBottom: '1px solid var(--border-subtle, rgba(255,255,255,0.04))',
                                background: actualIndex % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.01)'
                            }}
                        >
                            {visibleColumns.map((col) => (
                                <div
                                    key={col.key}
                                    style={{
                                        flex: col.width ? `0 0 ${col.width}px` : (col.key === 'messages' || col.type === 'conversation' ? '2' : '1'),
                                        minWidth: '50px',
                                        overflow: 'hidden',
                                        borderRight: '1px solid var(--border-subtle, rgba(255,255,255,0.04))',
                                        cursor: onCellClick ? 'pointer' : 'default',
                                        ...col.cellStyle
                                    }}
                                    onClick={() => onCellClick?.(actualIndex, col.key, row[col.key])}
                                >
                                    <DataCell
                                        value={editingCell?.rowIndex === actualIndex && editingCell.columnKey === col.key ? editValue : row[col.key]}
                                        type={col.type}
                                        isEditing={editingCell?.rowIndex === actualIndex && editingCell.columnKey === col.key}
                                        onEditStart={() => {
                                            if (col.editable !== false) {
                                                setEditingCell({ rowIndex: actualIndex, columnKey: col.key });
                                                setEditValue(row[col.key]);
                                            }
                                        }}
                                        onEditChange={setEditValue}
                                        onEditCommit={() => {
                                            if (editingCell) {
                                                onCellChange?.(editingCell.rowIndex, editingCell.columnKey, editValue);
                                                setEditingCell(null);
                                                setEditValue(null);
                                            }
                                        }}
                                    />
                                </div>
                            ))}
                        </div>
                    );
                })}
            </div>

            {/* Loading indicator / empty state */}
            {data.length === 0 && (
                <div style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    padding: '60px 20px',
                    color: 'var(--text-muted)'
                }}>
                    <FileText size={48} style={{ opacity: 0.3, marginBottom: '16px' }} />
                    <p style={{ fontSize: '14px' }}>No data to display</p>
                    <p style={{ fontSize: '12px', opacity: 0.6 }}>Select a dataset to view its contents</p>
                </div>
            )}
        </div>
    );
};

export default VirtualizedDataTable;
export type { DataRow, ColumnDef, VirtualizedDataTableProps };
