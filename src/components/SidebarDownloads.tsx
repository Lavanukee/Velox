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

interface SidebarDownloadsProps {
    tasks: DownloadTask[];
    isCollapsed: boolean;
}

export const SidebarDownloads: React.FC<SidebarDownloadsProps> = ({ tasks, isCollapsed }) => {
    const [isExpanded, setIsExpanded] = useState(true);

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

    return (
        <div className="sidebar-downloads">
            {/* Downloads Header */}
            <div
                className="sidebar-downloads-header"
                onClick={() => setIsExpanded(!isExpanded)}
            >
                <div className="downloads-header-content">
                    {isCollapsed ? (
                        <div className="downloads-collapsed-indicator">
                            <Download size={16} />
                            {activeTasks.length > 0 && (
                                <span className="download-badge">{activeTasks.length}</span>
                            )}
                        </div>
                    ) : (
                        <>
                            <span className="downloads-title">
                                {activeTasks.length > 0 ? `Downloads (${activeTasks.length})` : 'Downloads'}
                            </span>
                            <span className="downloads-toggle">
                                {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                            </span>
                        </>
                    )}
                </div>
            </div>

            {/* Downloads Body */}
            {!isCollapsed && isExpanded && (
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
                                <div className="download-progress-bar">
                                    <div
                                        className="download-progress-fill"
                                        style={{ width: `${task.progress}%` }}
                                    />
                                </div>
                                <span className="download-progress-text">{task.progress}%</span>
                            </div>
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
            )}
        </div>
    );
};
