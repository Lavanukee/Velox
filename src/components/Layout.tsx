import React from 'react';
import { Sidebar } from './Sidebar';
import { Header } from './Header';
import { SplashOverlay } from './SplashOverlay';
import { AppView, DownloadTask } from '../types';

interface LayoutProps {
    children: React.ReactNode;
    currentView: AppView;
    onNavigate: (view: AppView) => void;
    title: string;
    downloadTasks?: DownloadTask[];
    isSettingUp?: boolean;
    setupProgress?: number;
    setupMessage?: string;
    setupLoadedBytes?: number;
    setupTotalBytes?: number;
}

export const Layout: React.FC<LayoutProps> = ({
    children,
    currentView,
    onNavigate,
    title,
    downloadTasks = [],
    isSettingUp = false,
    setupProgress = 0,
    setupMessage = "Initializing environment...",
    setupLoadedBytes = 0,
    setupTotalBytes = 0
}) => {
    return (
        <div className="flex h-screen bg-app text-white overflow-hidden" style={{ display: 'flex', height: '100vh', background: 'var(--bg-app)', color: 'var(--text-primary)', overflow: 'hidden' }}>
            <Sidebar currentView={currentView} onNavigate={onNavigate} downloadTasks={downloadTasks} />

            <div className="flex-1 flex flex-col min-w-0" style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
                <SplashOverlay
                    isVisible={isSettingUp}
                    progress={setupProgress}
                    message={setupMessage}
                    loadedBytes={setupLoadedBytes}
                    totalBytes={setupTotalBytes}
                />

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
