import React, { useState } from 'react';
import {
    LayoutDashboard,
    Wrench,
    Database,
    BrainCircuit,
    MessageSquare,
    TerminalSquare,
    Settings,
    ChevronLeft,
    ChevronRight
} from 'lucide-react';
import { AppView, DownloadTask } from '../types';
import { SidebarDownloads } from './SidebarDownloads';
import '../styles/components.css';

interface SidebarProps {
    currentView: AppView;
    onNavigate: (view: AppView) => void;
    downloadTasks?: DownloadTask[];
}

export const Sidebar: React.FC<SidebarProps> = ({ currentView, onNavigate, downloadTasks = [] }) => {
    const [isCollapsed, setIsCollapsed] = useState(false);

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
                borderRight: '1px solid rgba(255, 255, 255, 0.1)',
                background: 'linear-gradient(135deg, rgba(12, 12, 16, 0.98) 0%, rgba(14, 12, 18, 0.98) 100%)',
                zIndex: 50,
                transition: 'width 0.3s ease'
            }}
        >
            {/* Collapse Button */}
            <div
                className="p-4 flex items-center justify-center border-b border-white/5"
                style={{
                    padding: '16px 20px',
                    borderBottom: '1px solid rgba(255, 255, 255, 0.05)',
                    justifyContent: 'center',
                    display: 'flex'
                }}
            >
                <button
                    onClick={() => setIsCollapsed(!isCollapsed)}
                    style={{
                        background: 'rgba(255, 255, 255, 0.08)',
                        border: '1px solid rgba(255, 255, 255, 0.15)',
                        borderRadius: '8px',
                        width: '36px',
                        height: '36px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        color: 'white',
                        cursor: 'pointer',
                        transition: 'all 0.2s',
                        marginLeft: '10px', // As requested
                        boxShadow: '0 2px 8px rgba(0,0,0,0.2)'
                    }}
                    onMouseEnter={(e) => {
                        e.currentTarget.style.background = 'rgba(255, 255, 255, 0.15)';
                        e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.3)';
                    }}
                    onMouseLeave={(e) => {
                        e.currentTarget.style.background = 'rgba(255, 255, 255, 0.08)';
                        e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.15)';
                    }}
                >
                    {isCollapsed ? <ChevronRight size={20} /> : <ChevronLeft size={20} />}
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
                            border: currentView === item.view ? '1px solid rgba(167, 139, 250, 0.3)' : '1px solid transparent',
                            background: currentView === item.view
                                ? 'linear-gradient(135deg, rgba(167, 139, 250, 0.15) 0%, rgba(125, 211, 252, 0.05) 100%)'
                                : 'transparent',
                            color: currentView === item.view ? '#c4b5fd' : '#a1a1aa',
                            cursor: 'pointer',
                            fontSize: '0.95rem',
                            fontWeight: currentView === item.view ? 500 : 400,
                            transition: 'all 0.2s',
                            whiteSpace: 'nowrap',
                            overflow: 'hidden'
                        }}
                        onMouseEnter={(e) => {
                            if (currentView !== item.view) {
                                e.currentTarget.style.background = 'rgba(255, 255, 255, 0.03)';
                                e.currentTarget.style.color = 'white';
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

            {/* Downloads Section */}
            {!isCollapsed && (
                <SidebarDownloads tasks={downloadTasks} isCollapsed={isCollapsed} />
            )}
            {isCollapsed && downloadTasks.length > 0 && (
                <div style={{ borderTop: '1px solid rgba(255, 255, 255, 0.05)' }}>
                    <SidebarDownloads tasks={downloadTasks} isCollapsed={isCollapsed} />
                </div>
            )}

            {/* Settings */}
            <div
                className="p-3 border-t border-white/5"
                style={{
                    padding: isCollapsed ? '12px 8px' : '12px',
                    borderTop: '1px solid rgba(255, 255, 255, 0.05)',
                    marginTop: 'auto' // Ensure it stays at bottom even if nav is short, though flex-1 on nav handles this. Added for safety.
                }}
            >
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
