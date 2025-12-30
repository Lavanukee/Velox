import React from 'react';
import { motion } from 'framer-motion';
import '../styles/components.css';

interface ToggleProps {
    checked: boolean;
    onChange: (checked: boolean) => void;
    label?: string;
}

export const Toggle: React.FC<ToggleProps> = ({ checked, onChange, label }) => {
    return (
        <div
            className="flex items-center gap-3 cursor-pointer"
            onClick={() => onChange(!checked)}
            style={{ display: 'flex', alignItems: 'center', gap: '12px', cursor: 'pointer' }}
        >
            <div
                className={`toggle-switch ${checked ? 'checked' : ''}`}
                style={{
                    backgroundColor: checked ? 'rgba(139, 92, 246, 0.4)' : 'rgba(255, 255, 255, 0.08)',
                    transition: 'background-color 0.2s'
                }}
            >
                <motion.div
                    className="toggle-thumb"
                    animate={{
                        x: checked ? 20 : 0
                    }}
                    style={{
                        backgroundColor: checked ? '#a78bfa' : '#6b7280'
                    }}
                    transition={{ type: "spring", stiffness: 500, damping: 30 }}
                />
            </div>
            {label && <span className="text-secondary" style={{ fontSize: '0.875rem', userSelect: 'none' }}>{label}</span>}
        </div>
    );
};
