import React from 'react';
import { Sidebar } from './Sidebar';
import { Header } from './Header';
import { AppView, DownloadTask } from '../types';

interface LayoutProps {
    children: React.ReactNode;
    currentView: AppView;
    onNavigate: (view: AppView) => void;
    title: string;
    downloadTasks?: DownloadTask[];
    isSettingUp?: boolean;
    setupProgress?: number;
}

export const Layout: React.FC<LayoutProps> = ({
    children,
    currentView,
    onNavigate,
    title,
    downloadTasks = [],
    isSettingUp = false,
    setupProgress = 0
}) => {
    return (
        <div className="flex h-screen bg-app text-white overflow-hidden" style={{ display: 'flex', height: '100vh', background: 'var(--bg-app)', color: 'var(--text-primary)', overflow: 'hidden' }}>
            <Sidebar currentView={currentView} onNavigate={onNavigate} downloadTasks={downloadTasks} />

            <div className="flex-1 flex flex-col min-w-0" style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
                {isSettingUp && (
                    <div className="w-full h-16 bg-[#0a0a0c] border-b border-accent-primary/20 relative flex items-center px-8 shadow-2xl z-50 overflow-hidden">
                        {/* Progress Background */}
                        <div
                            className="absolute top-0 left-0 h-full bg-accent-primary/10 transition-all duration-700 ease-out"
                            style={{ width: `${setupProgress}%` }}
                        />

                        {/* Progress Line */}
                        <div
                            className="absolute bottom-0 left-0 h-[3px] bg-accent-primary shadow-[0_0_15px_rgba(59,130,246,0.6)] transition-all duration-700 ease-out"
                            style={{ width: `${setupProgress}%` }}
                        />

                        <div className="relative z-10 flex items-center gap-4 animate-fade-in">
                            <div className="h-5 w-5 border-2 border-accent-primary border-t-transparent rounded-full animate-spin" />
                            <div className="flex flex-col">
                                <span className="text-sm font-bold text-white tracking-widest uppercase">App Setting Up</span>
                                <span className="text-xs text-accent-primary font-mono">
                                    {setupProgress >= 100 ? "Finalizing..." : `Initializing environment... ${Math.round(setupProgress)}%`}
                                </span>
                            </div>
                        </div>

                        {/* Shimmer Effect */}
                        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent -translate-x-full animate-[shimmer_2s_infinite]"></div>
                    </div>
                )}

                <Header title={title} />
                <main className="flex-1 overflow-y-auto p-8 scroll-smooth" style={{ flex: 1, overflowY: 'auto', padding: '32px' }}>
                    <div className="max-w-7xl mx-auto w-full animate-fade-in" style={{ maxWidth: '1400px', margin: '0 auto', width: '100%' }}>
                        {children}
                    </div>
                </main>
            </div>
        </div>
    );
};
