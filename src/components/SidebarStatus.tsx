import React, { useState } from 'react';
import {
    Cpu,
    BrainCircuit,
    Wrench,
    ChevronUp,
    ChevronDown,
    Loader2,
    Square
} from 'lucide-react';
import '../styles/SidebarDownloads.css'; // Reusing styles

import { motion, AnimatePresence } from 'framer-motion';

interface SidebarStatusProps {
    loadedModels: { name: string; type?: string; path: string; serverId?: string }[];
    isTraining: boolean;
    projectName?: string;
    isConverting?: boolean;
    conversionLabel?: string;
    conversionProgress?: number;
    isCollapsed: boolean;
    onStopModel?: (path: string, serverId?: string) => void;
}

export const SidebarStatus: React.FC<SidebarStatusProps> = ({
    loadedModels,
    isTraining,
    projectName,
    isConverting = false,
    conversionLabel = 'Converting...',
    conversionProgress = 0,
    isCollapsed,
    onStopModel
}) => {
    const [isExpanded, setIsExpanded] = useState(true);
    const [isPopoutOpen, setIsPopoutOpen] = useState(false);

    if (loadedModels.length === 0 && !isTraining && !isConverting) {
        return null;
    }

    const renderBody = () => (
        <div className="sidebar-downloads-body" style={{ maxHeight: isCollapsed ? '300px' : '200px' }}>
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
                                    color: '#9ca3af',
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
                            <BrainCircuit size={14} className="download-icon active" style={{ color: '#60a5fa' }} />
                            <span className="download-name" style={{ color: '#f5f3ff' }}>
                                {projectName || 'Training...'}
                            </span>
                        </div>
                        <Loader2 size={12} className="animate-spin" style={{ color: '#60a5fa' }} />
                    </div>
                    <div className="download-progress-container">
                        <span className="download-progress-text" style={{ fontSize: '10px', opacity: 0.7 }}>
                            Job in progress
                        </span>
                    </div>
                </div>
            )}

            {/* Active Conversion */}
            {isConverting && (
                <div className="download-item">
                    <div className="download-item-header">
                        <div className="download-item-info">
                            <Wrench size={14} className="download-icon active" style={{ color: '#4ade80' }} />
                            <span className="download-name" style={{ color: '#f0fdf4' }}>
                                {conversionLabel}
                            </span>
                        </div>
                        {conversionProgress > 0 && <span style={{ fontSize: '10px', color: '#4ade80', fontWeight: 'bold' }}>{conversionProgress}%</span>}
                        <Loader2 size={12} className="animate-spin" style={{ color: '#4ade80' }} />
                    </div>
                    <div className="download-progress-container">
                        <div className="download-progress-bar">
                            <motion.div
                                className="download-progress-fill"
                                initial={{ width: "0%" }}
                                animate={{ width: `${conversionProgress || 0}%` }}
                                transition={{ duration: 0.5 }}
                                style={{ background: '#4ade80' }}
                            />
                        </div>
                        <span className="download-progress-text" style={{ fontSize: '10px', opacity: 0.7 }}>
                            {conversionProgress > 0 ? 'Processing...' : 'Initializing...'}
                        </span>
                    </div>
                </div>
            )}
        </div>
    );

    if (isCollapsed) {
        return (
            <div style={{ position: 'relative', display: 'flex', justifyContent: 'center', padding: '12px 0' }}>
                <div
                    className="downloads-collapsed-indicator"
                    onClick={() => setIsPopoutOpen(!isPopoutOpen)}
                    style={{ cursor: 'pointer' }}
                >
                    <ActivityIndicator active={loadedModels.length > 0 || isTraining || isConverting} />
                </div>
                <AnimatePresence>
                    {isPopoutOpen && (
                        <motion.div
                            className="sidebar-downloads-popout"
                            initial={{ opacity: 0, x: -20, scale: 0.95 }}
                            animate={{ opacity: 1, x: 12, scale: 1 }}
                            exit={{ opacity: 0, x: -10, scale: 0.95 }}
                            transition={{ type: 'spring', damping: 20, stiffness: 300 }}
                            style={{ bottom: '10px' }}
                        >
                            <div className="flex items-center justify-between mb-3">
                                <span className="text-xs font-bold text-gray-400">Activity</span>
                                <button onClick={() => setIsPopoutOpen(false)} style={{ background: 'none', border: 'none', color: '#71717a', cursor: 'pointer' }}>
                                    <ChevronDown size={14} />
                                </button>
                            </div>
                            {renderBody()}
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>
        );
    }

    return (
        <div className="sidebar-downloads" style={{ borderTop: 'none', marginTop: '4px' }}>
            <div
                className="sidebar-downloads-header"
                onClick={() => setIsExpanded(!isExpanded)}
            >
                <div className="downloads-header-content">
                    <span className="downloads-title">
                        Activity
                    </span>
                    <span className="downloads-toggle">
                        {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                    </span>
                </div>
            </div>

            {isExpanded && renderBody()}
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
