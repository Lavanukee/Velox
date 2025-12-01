import React from 'react';
import { useApp } from '../context/AppContext';
import { Toggle } from './Toggle';
import { Zap, User } from 'lucide-react';

interface HeaderProps {
    title: string;
}

export const Header: React.FC<HeaderProps> = ({ title }) => {
    const { userMode, toggleUserMode } = useApp();

    return (
        <header className="h-16 border-b border-white/5 flex items-center justify-between px-8 glass" style={{
            height: 'var(--header-height)',
            borderBottom: '1px solid var(--border-subtle)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '0 32px',
            background: 'rgba(10, 10, 12, 0.6)',
            backdropFilter: 'blur(12px)'
        }}>
            <h1 className="text-xl font-semibold text-white tracking-wide">{title}</h1>

            <div className="flex items-center gap-4">
                <div className="flex items-center gap-2 bg-black/20 px-3 py-1.5 rounded-full border border-white/5" style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '12px',
                    background: 'rgba(0,0,0,0.2)',
                    padding: '6px 12px',
                    borderRadius: '99px',
                    border: '1px solid rgba(255,255,255,0.05)'
                }}>
                    <div className="flex items-center gap-2 text-sm text-gray-400">
                        {userMode === 'power' ? <Zap size={14} className="text-yellow-400" /> : <User size={14} />}
                        <span>{userMode === 'power' ? 'Power User' : 'Standard'}</span>
                    </div>
                    <Toggle checked={userMode === 'power'} onChange={toggleUserMode} />
                </div>
            </div>
        </header>
    );
};
