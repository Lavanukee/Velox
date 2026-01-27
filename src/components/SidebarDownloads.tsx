import React, { useState } from 'react';
import {
    Download,
    CheckCircle,
    XCircle,
    ChevronUp,
    ChevronDown,
    X
} from 'lucide-react';
import { DownloadTask } from '../types';
import '../styles/SidebarDownloads.css';

import { motion, AnimatePresence } from 'framer-motion';

interface SidebarDownloadsProps {
    tasks: DownloadTask[];
    isCollapsed: boolean;
}

export const SidebarDownloads: React.FC<SidebarDownloadsProps> = ({ tasks, isCollapsed }) => {
    const [isExpanded, setIsExpanded] = useState(true);
    const [isPopoutOpen, setIsPopoutOpen] = useState(false);

    const activeTasks = tasks.filter(task => task.status === 'downloading' || task.status === 'pending');
    const completedOrErrorTasks = tasks.filter(task => task.status === 'completed' || task.status === 'error' || task.status === 'cancelled');

    if (tasks.length === 0) {
        return null;
    }

    const handleCancel = (task: DownloadTask) => {
        if (task.onCancel) {
            task.onCancel();
        }
    };

    const formatBytes = (bytes?: number) => {
        if (bytes === undefined || bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    };

    const formatETA = (seconds?: number) => {
        if (seconds === undefined || seconds <= 0) return '';
        if (seconds < 60) return `${Math.round(seconds)}s left`;
        const mins = Math.floor(seconds / 60);
        const secs = Math.round(seconds % 60);
        return `${mins}m ${secs}s left`;
    };

    const renderBody = () => (
        <div className="sidebar-downloads-body">
            {activeTasks.length === 0 && completedOrErrorTasks.length === 0 && (
                <p className="downloads-empty">No downloads</p>
            )}

            {/* Active Downloads */}
            {activeTasks.map(task => (
                <div key={task.id} className="download-item">
                    <div className="download-item-header">
                        <div className="download-item-info">
                            <Download size={14} className="download-icon active" />
                            <span className="download-name">{task.name}</span>
                        </div>
                        <button
                            className="download-cancel-btn"
                            onClick={() => handleCancel(task)}
                            title="Cancel download"
                            aria-label="Cancel"
                        >
                            <X size={14} />
                        </button>
                    </div>

                    <div className="download-progress-container">
                        <div className="download-progress-row">
                            <div className="download-progress-bar">
                                <div
                                    className="download-progress-fill"
                                    style={{ width: `${task.progress}%` }}
                                />
                            </div>
                        </div>
                        <div className="download-stats-row">
                            <span className="download-progress-text">{task.progress.toFixed(1)}%</span>
                            {task.downloaded_bytes !== undefined && task.total_bytes !== undefined && (
                                <span className="download-bytes">
                                    {formatBytes(task.downloaded_bytes)} / {formatBytes(task.total_bytes)}
                                </span>
                            )}
                        </div>
                    </div>

                    {(task.speed_bps || task.eta_seconds) && (
                        <div className="download-extra-info">
                            <span className="download-speed">
                                {task.speed_bps ? `${formatBytes(task.speed_bps)}/s` : ''}
                            </span>
                            <span className="download-eta">
                                {formatETA(task.eta_seconds)}
                            </span>
                        </div>
                    )}
                </div>
            ))}

            {/* Completed/Error Downloads */}
            {completedOrErrorTasks.map(task => (
                <div key={task.id} className={`download-item download-item-${task.status}`}>
                    <div className="download-item-header">
                        <div className="download-item-info">
                            {task.status === 'completed' && (
                                <CheckCircle size={14} className="download-icon completed" />
                            )}
                            {task.status === 'error' && (
                                <XCircle size={14} className="download-icon error" />
                            )}
                            {task.status === 'cancelled' && (
                                <X size={14} className="download-icon cancelled" />
                            )}
                            <span className="download-name">{task.name}</span>
                        </div>
                        <span className={`download-status-badge ${task.status}`}>
                            {task.status === 'completed' ? 'Done' : task.status === 'error' ? 'Failed' : 'Cancelled'}
                        </span>
                    </div>
                </div>
            ))}
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
                    <Download size={18} />
                    {activeTasks.length > 0 && (
                        <span className="download-badge">{activeTasks.length}</span>
                    )}
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
                                <span className="text-xs font-bold text-gray-400">Downloads</span>
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
        <div className="sidebar-downloads">
            <div className="sidebar-downloads-header" onClick={() => setIsExpanded(!isExpanded)}>
                <div className="downloads-header-content">
                    <span className="downloads-title">
                        {activeTasks.length > 0 ? `Downloads (${activeTasks.length})` : 'Downloads'}
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
