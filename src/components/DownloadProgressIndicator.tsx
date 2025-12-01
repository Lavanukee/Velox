import React, { useState } from 'react';
import { Download, CheckCircle, XCircle, ChevronUp, ChevronDown } from 'lucide-react';
import '../styles/DownloadProgressIndicator.css'; // New CSS file

interface DownloadTask {
    id: string;
    name: string;
    progress: number; // 0-100
    status: 'pending' | 'downloading' | 'completed' | 'error' | 'cancelled';
    type: string; // e.g., 'model', 'dataset', 'binary'
}

interface DownloadProgressIndicatorProps {
    tasks: DownloadTask[];
}

const DownloadProgressIndicator: React.FC<DownloadProgressIndicatorProps> = ({ tasks }) => {
    const [isCollapsed, setIsCollapsed] = useState(false);
    const activeTasks = tasks.filter(task => task.status !== 'completed' && task.status !== 'error' && task.status !== 'cancelled');
    const completedOrErrorTasks = tasks.filter(task => task.status === 'completed' || task.status === 'error');

    if (tasks.length === 0) {
        return null; // Don't render if no tasks
    }

    const toggleCollapse = () => {
        setIsCollapsed(!isCollapsed);
    };

    return (
        <div className="download-indicator-container">
            <div className="download-indicator-header" onClick={toggleCollapse}>
                <h3>
                    {activeTasks.length > 0 ? `Active Downloads (${activeTasks.length})` : 'Downloads'}
                </h3>
                <div className="header-actions">
                    {isCollapsed ? <ChevronDown size={18} /> : <ChevronUp size={18} />}
                </div>
            </div>

            {!isCollapsed && (
                <div className="download-indicator-body">
                    {activeTasks.length === 0 && completedOrErrorTasks.length === 0 && (
                        <p className="no-downloads-message">No active or recent downloads.</p>
                    )}

                    {activeTasks.map(task => (
                        <div key={task.id} className="download-task-item">
                            <div className="task-info">
                                <Download size={18} className="task-icon downloading" />
                                <span className="task-name">{task.name}</span>
                                <span className="task-progress">{task.progress}%</span>
                            </div>
                            <div className="progress-bar-background">
                                <div
                                    className="progress-bar-fill"
                                    style={{ width: `${task.progress}%` }}
                                ></div>
                            </div>
                        </div>
                    ))}

                    {completedOrErrorTasks.map(task => (
                        <div key={task.id} className={`download-task-item task-${task.status}`}>
                            <div className="task-info">
                                {task.status === 'completed' && <CheckCircle size={18} className="task-icon completed" />}
                                {task.status === 'error' && <XCircle size={18} className="task-icon error" />}
                                <span className="task-name">{task.name}</span>
                                <span className="task-status-text">{task.status === 'completed' ? 'Completed' : 'Failed'}</span>
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};

export default DownloadProgressIndicator;