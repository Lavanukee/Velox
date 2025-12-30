import React, { useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import {
    LayoutDashboard,
    Wrench,
    Database,
    BrainCircuit,
    MessageSquare,
    TerminalSquare,
    Settings,
    PanelLeft,
    Sun,
    Moon
} from 'lucide-react';
import { AppView, DownloadTask } from '../types';
import { SidebarDownloads } from './SidebarDownloads';
import { SidebarStatus } from './SidebarStatus';
import { useApp } from '../context/AppContext';
import { useAppState } from '../context/AppStateContext';
import '../styles/components.css';

interface SidebarProps {
    currentView: AppView;
    onNavigate: (view: AppView) => void;
    downloadTasks?: DownloadTask[];
}

export const Sidebar: React.FC<SidebarProps> = ({ currentView, onNavigate, downloadTasks = [] }) => {
    const [isCollapsed, setIsCollapsed] = useState(false);
    const { colorMode, toggleColorMode, ftIsTraining, ftProjectName } = useApp();
    const { state, unloadModel } = useAppState();
    const loadedModels = state.inference.loadedModels;

    const handleStopModel = async (path: string, serverId?: string) => {
        try {
            // Determine slot from serverId (default to 0)
            let slotId = 0;
            if (serverId === 'server-1') slotId = 1;

            await invoke('stop_llama_server_command', { slotId });
            unloadModel(path);
        } catch (error) {
            console.error('Failed to stop model:', error);
        }
    };

    const navItems = [
        { view: AppView.Dashboard, label: 'Dashboard', icon: <LayoutDashboard size={20} /> },
        { view: AppView.Utilities, label: 'Utilities', icon: <Wrench size={20} /> },
        { view: AppView.DataCollection, label: 'Data Collection', icon: <Database size={20} /> },
        { view: AppView.FineTuning, label: 'Fine-Tuning', icon: <BrainCircuit size={20} /> },
        { view: AppView.Inference, label: 'Inference', icon: <MessageSquare size={20} /> },
        { view: AppView.Logs, label: 'Logs', icon: <TerminalSquare size={20} /> },
    ];

    return (
        <aside
            className="sidebar glass transition-all duration-300"
            style={{
                width: isCollapsed ? '72px' : '260px',
                height: '100vh',
                display: 'flex',
                flexDirection: 'column',
                borderRight: '1px solid var(--border-subtle)',
                background: 'var(--bg-surface)',
                zIndex: 50,
                transition: 'width 0.3s ease'
            }}
        >
            {/* Sidebar Toggle Area */}
            <div
                className="p-4 flex items-center border-b border-white/5"
                style={{
                    padding: '12px 16px',
                    borderBottom: '1px solid var(--border-subtle)',
                    justifyContent: isCollapsed ? 'center' : 'space-between',
                    display: 'flex',
                    minHeight: '64px'
                }}
            >
                {!isCollapsed && <span className="font-bold text-lg tracking-wider bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent">VELOX</span>}
                <button
                    onClick={() => setIsCollapsed(!isCollapsed)}
                    title={isCollapsed ? "Expand Sidebar" : "Collapse Sidebar"}
                    style={{
                        background: 'var(--bg-elevated)',
                        border: '1px solid var(--border-default)',
                        borderRadius: '6px',
                        width: '40px',
                        height: '32px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        color: 'var(--text-secondary)',
                        cursor: 'pointer',
                        transition: 'all 0.2s',
                        boxShadow: 'var(--shadow-3d-sm)'
                    }}
                    onMouseEnter={(e) => {
                        e.currentTarget.style.background = 'var(--bg-highlight)';
                        e.currentTarget.style.color = 'var(--text-main)';
                    }}
                    onMouseLeave={(e) => {
                        e.currentTarget.style.background = 'var(--bg-elevated)';
                        e.currentTarget.style.color = 'var(--text-secondary)';
                    }}
                >
                    <PanelLeft size={18} />
                </button>
            </div>

            {/* Navigation */}
            <nav
                className="flex-1 p-3 flex flex-col gap-2 overflow-y-auto"
                style={{
                    padding: isCollapsed ? '12px 8px' : '12px',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '8px',
                    overflowY: 'auto'
                }}
            >
                {navItems.map((item) => (
                    <button
                        key={item.label}
                        onClick={() => onNavigate(item.view)}
                        title={isCollapsed ? item.label : ''}
                        style={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: '12px',
                            justifyContent: isCollapsed ? 'center' : 'flex-start',
                            padding: isCollapsed ? '12px 0' : '12px 16px',
                            borderRadius: '8px',
                            border: currentView === item.view
                                ? '1px solid rgba(139, 92, 246, 0.25)'
                                : '1px solid transparent',
                            background: currentView === item.view
                                ? 'var(--bg-highlight, #222230)'
                                : 'transparent',
                            boxShadow: currentView === item.view
                                ? 'inset 0 1px 0 rgba(255,255,255,0.04), 0 2px 8px rgba(0,0,0,0.15)'
                                : 'none',
                            color: currentView === item.view ? '#c4b5fd' : '#a1a1aa',
                            cursor: 'pointer',
                            fontSize: '0.9375rem',
                            fontWeight: currentView === item.view ? 500 : 400,
                            transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
                            whiteSpace: 'nowrap',
                            overflow: 'hidden'
                        }}
                        onMouseEnter={(e) => {
                            if (currentView !== item.view) {
                                e.currentTarget.style.background = 'rgba(255, 255, 255, 0.04)';
                                e.currentTarget.style.color = '#e4e4e7';
                            }
                        }}
                        onMouseLeave={(e) => {
                            if (currentView !== item.view) {
                                e.currentTarget.style.background = 'transparent';
                                e.currentTarget.style.color = '#a1a1aa';
                            }
                        }}
                    >
                        <span style={{ display: 'flex', alignItems: 'center', flexShrink: 0 }}>
                            {item.icon}
                        </span>
                        {!isCollapsed && <span>{item.label}</span>}
                    </button>
                ))}
            </nav>

            {/* Active Status Section */}
            {!isCollapsed && (
                <SidebarStatus
                    loadedModels={loadedModels}
                    isTraining={ftIsTraining}
                    projectName={ftProjectName}
                    isCollapsed={isCollapsed}
                    onStopModel={handleStopModel}
                />
            )}
            {isCollapsed && (loadedModels.length > 0 || ftIsTraining) && (
                <div style={{ borderTop: '1px solid rgba(255, 255, 255, 0.05)' }}>
                    <SidebarStatus
                        loadedModels={loadedModels}
                        isTraining={ftIsTraining}
                        projectName={ftProjectName}
                        isCollapsed={isCollapsed}
                        onStopModel={handleStopModel}
                    />
                </div>
            )}

            {/* Downloads Section */}
            {!isCollapsed && (
                <SidebarDownloads tasks={downloadTasks} isCollapsed={isCollapsed} />
            )}
            {isCollapsed && downloadTasks.length > 0 && (
                <div style={{ borderTop: '1px solid rgba(255, 255, 255, 0.05)' }}>
                    <SidebarDownloads tasks={downloadTasks} isCollapsed={isCollapsed} />
                </div>
            )}

            {/* Settings & Theme Toggle */}
            <div
                className="p-3 border-t border-white/5"
                style={{
                    padding: isCollapsed ? '12px 8px' : '12px',
                    borderTop: '1px solid rgba(255, 255, 255, 0.05)',
                    marginTop: 'auto',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '8px'
                }}
            >
                {/* Light/Dark Mode Toggle */}
                <button
                    onClick={toggleColorMode}
                    title={isCollapsed ? (colorMode === 'dark' ? 'Light Mode' : 'Dark Mode') : ''}
                    style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '12px',
                        justifyContent: isCollapsed ? 'center' : 'flex-start',
                        padding: isCollapsed ? '12px 0' : '12px 16px',
                        borderRadius: '8px',
                        background: 'transparent',
                        color: '#a1a1aa',
                        cursor: 'pointer',
                        border: 'none',
                        width: '100%',
                        transition: 'all 0.2s'
                    }}
                    onMouseEnter={(e) => {
                        e.currentTarget.style.background = 'rgba(255, 255, 255, 0.05)';
                        e.currentTarget.style.color = 'var(--text-main)';
                    }}
                    onMouseLeave={(e) => {
                        e.currentTarget.style.background = 'transparent';
                        e.currentTarget.style.color = '#a1a1aa';
                    }}
                >
                    <span style={{ display: 'flex', alignItems: 'center', flexShrink: 0 }}>
                        {colorMode === 'dark' ? <Sun size={20} /> : <Moon size={20} />}
                    </span>
                    {!isCollapsed && <span>{colorMode === 'dark' ? 'Light Mode' : 'Dark Mode'}</span>}
                </button>

                {/* Settings Button */}
                <button
                    onClick={() => onNavigate(AppView.Settings)}
                    title={isCollapsed ? 'Settings' : ''}
                    style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '12px',
                        justifyContent: isCollapsed ? 'center' : 'flex-start',
                        padding: isCollapsed ? '12px 0' : '12px 16px',
                        borderRadius: '8px',
                        background: 'transparent',
                        color: currentView === AppView.Settings ? 'white' : '#a1a1aa',
                        cursor: 'pointer',
                        border: 'none',
                        width: '100%',
                        transition: 'all 0.2s'
                    }}
                    onMouseEnter={(e) => {
                        e.currentTarget.style.background = 'rgba(255, 255, 255, 0.05)';
                        e.currentTarget.style.color = 'white';
                    }}
                    onMouseLeave={(e) => {
                        e.currentTarget.style.background = 'transparent';
                        e.currentTarget.style.color = currentView === AppView.Settings ? 'white' : '#a1a1aa';
                    }}
                >
                    <span style={{ display: 'flex', alignItems: 'center', flexShrink: 0 }}>
                        <Settings size={20} />
                    </span>
                    {!isCollapsed && <span>Settings</span>}
                </button>
            </div>
        </aside>
    );
};
