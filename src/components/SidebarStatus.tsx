import React, { useState } from 'react';
import {
    Cpu,
    BrainCircuit,
    ChevronUp,
    ChevronDown,
    Loader2,
    Square
} from 'lucide-react';
import '../styles/SidebarDownloads.css'; // Reusing styles

interface SidebarStatusProps {
    loadedModels: { name: string; type?: string; path: string; serverId?: string }[];
    isTraining: boolean;
    projectName?: string;
    isCollapsed: boolean;
    onStopModel?: (path: string, serverId?: string) => void;
}

export const SidebarStatus: React.FC<SidebarStatusProps> = ({
    loadedModels,
    isTraining,
    projectName,
    isCollapsed,
    onStopModel
}) => {
    const [isExpanded, setIsExpanded] = useState(true);

    if (loadedModels.length === 0 && !isTraining) {
        return null;
    }

    return (
        <div className="sidebar-downloads" style={{ borderTop: 'none', marginTop: '4px' }}>
            <div
                className="sidebar-downloads-header"
                onClick={() => setIsExpanded(!isExpanded)}
            >
                <div className="downloads-header-content">
                    {isCollapsed ? (
                        <div className="downloads-collapsed-indicator">
                            <ActivityIndicator active={loadedModels.length > 0 || isTraining} />
                        </div>
                    ) : (
                        <>
                            <span className="downloads-title">
                                Activity
                            </span>
                            <span className="downloads-toggle">
                                {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                            </span>
                        </>
                    )}
                </div>
            </div>

            {!isCollapsed && isExpanded && (
                <div className="sidebar-downloads-body" style={{ maxHeight: '200px' }}>
                    {/* Loaded Models */}
                    {loadedModels.map((model, idx) => (
                        <div key={idx} className="download-item">
                            <div className="download-item-header">
                                <div className="download-item-info">
                                    <Cpu size={14} className="download-icon active" style={{ color: '#10b981' }} />
                                    <span className="download-name" style={{ color: '#ecfeff' }}>
                                        {model.name.split('/').pop()}
                                    </span>
                                </div>
                                <span style={{ fontSize: '10px', color: 'rgba(255,255,255,0.4)', background: 'rgba(255,255,255,0.05)', padding: '2px 6px', borderRadius: '4px' }}>
                                    {model.type || 'unknown'}
                                </span>
                                {onStopModel && (
                                    <button
                                        className="sidebar-stop-btn"
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            onStopModel(model.path, model.serverId);
                                        }}
                                        title="Stop Server"
                                        style={{
                                            marginLeft: 'auto',
                                            background: 'none',
                                            border: 'none',
                                            cursor: 'pointer',
                                            color: '#9ca3af', // Neutral gray
                                            padding: '4px',
                                            display: 'flex',
                                            alignItems: 'center',
                                            justifyContent: 'center',
                                            borderRadius: '4px',
                                            transition: 'all 0.2s'
                                        }}
                                        onMouseEnter={(e) => {
                                            e.currentTarget.style.color = '#ef4444';
                                            e.currentTarget.style.background = 'rgba(239,68,68,0.1)';
                                        }}
                                        onMouseLeave={(e) => {
                                            e.currentTarget.style.color = '#9ca3af';
                                            e.currentTarget.style.background = 'none';
                                        }}
                                    >
                                        <Square size={12} fill="currentColor" />
                                    </button>
                                )}
                            </div>
                        </div>
                    ))}

                    {/* Active Training */}
                    {isTraining && (
                        <div className="download-item">
                            <div className="download-item-header">
                                <div className="download-item-info">
                                    <BrainCircuit size={14} className="download-icon active" style={{ color: '#a78bfa' }} />
                                    <span className="download-name" style={{ color: '#f5f3ff' }}>
                                        {projectName || 'Training...'}
                                    </span>
                                </div>
                                <Loader2 size={12} className="animate-spin" style={{ color: '#a78bfa' }} />
                            </div>
                            <div className="download-progress-container">
                                <span className="download-progress-text" style={{ fontSize: '10px', opacity: 0.7 }}>
                                    Job in progress
                                </span>
                            </div>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

const ActivityIndicator: React.FC<{ active: boolean }> = ({ active }) => {
    return (
        <div style={{ position: 'relative' }}>
            <Cpu size={16} style={{ color: active ? '#10b981' : '#71717a' }} />
            {active && (
                <span style={{
                    position: 'absolute',
                    top: -2,
                    right: -2,
                    width: '6px',
                    height: '6px',
                    background: '#10b981',
                    borderRadius: '50%',
                    boxShadow: '0 0 4px #10b981'
                }} />
            )}
        </div>
    );
};
