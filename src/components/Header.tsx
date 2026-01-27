import React from 'react';
import { useApp } from '../context/AppContext';
import { Zap, User } from 'lucide-react';

interface HeaderProps {
    title: string;
}

export const Header: React.FC<HeaderProps> = ({ title }) => {
    const { userMode, toggleUserMode, userFeatures } = useApp();

    return (
        <header className="h-16 border-b border-white/5 flex items-center justify-between px-8 glass" style={{
            height: 'var(--header-height)',
            borderBottom: '1px solid var(--border-subtle)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '0 32px',
            background: 'var(--bg-surface)',
            backdropFilter: 'blur(12px)',
            color: 'var(--text-main)'
        }}>
            <h1 className="text-xl font-semibold tracking-wide" style={{ color: 'var(--text-main)' }}>{title}</h1>

            {userFeatures.showUserModeToggle && (
                <div
                    className="mode-toggle"
                    onClick={toggleUserMode}
                    style={{
                        position: 'relative',
                        display: 'flex',
                        alignItems: 'center',
                        background: 'var(--bg-elevated)',
                        borderRadius: '12px',
                        padding: '4px',
                        cursor: 'pointer',
                        width: '180px',
                        height: '36px',
                        border: '1px solid var(--border-default)',
                        userSelect: 'none',
                        overflow: 'hidden'
                    }}
                >
                    {/* Sliding Glass Reveal */}
                    <div style={{
                        position: 'absolute',
                        left: userMode === 'power' ? 'calc(50% + 2px)' : '4px',
                        width: 'calc(50% - 6px)',
                        height: 'calc(100% - 8px)',
                        background: 'rgba(59, 130, 246, 0.3)',
                        backdropFilter: 'blur(8px)',
                        boxShadow: '0 0 15px rgba(59, 130, 246, 0.2)',
                        borderRadius: '8px',
                        transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                        zIndex: 1,
                        border: '1px solid rgba(59, 130, 246, 0.4)'
                    }} />

                    <div style={{
                        flex: 1,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        gap: '6px',
                        fontSize: '11px',
                        fontWeight: 600,
                        zIndex: 2,
                        color: userMode !== 'power' ? 'var(--text-main)' : 'var(--text-secondary)',
                        transition: 'color 0.2s',
                        letterSpacing: '0.05em'
                    }}>
                        <User size={14} /> USER
                    </div>
                    <div style={{
                        flex: 1,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        gap: '6px',
                        fontSize: '11px',
                        fontWeight: 600,
                        zIndex: 2,
                        color: userMode === 'power' ? 'var(--text-main)' : 'var(--text-secondary)',
                        transition: 'color 0.2s',
                        letterSpacing: '0.05em'
                    }}>
                        <Zap size={14} className={userMode === 'power' ? 'text-yellow-400' : ''} /> POWER
                    </div>
                </div>
            )}
        </header>
    );
};
