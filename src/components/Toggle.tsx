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
        <div className="flex items-center gap-3 cursor-pointer" onClick={() => onChange(!checked)}>
            <motion.div
                className={`toggle-switch ${checked ? 'checked' : ''}`}
                animate={{
                    backgroundColor: checked ? 'rgba(59, 130, 246, 0.5)' : 'rgba(255, 255, 255, 0.1)'
                }}
                transition={{ duration: 0.2 }}
            >
                <motion.div
                    className="toggle-thumb"
                    animate={{
                        x: checked ? 20 : 0,
                        backgroundColor: checked ? '#3b82f6' : '#94a3b8'
                    }}
                    transition={{ type: "spring", stiffness: 500, damping: 30 }}
                />
            </motion.div>
            {label && <span className="text-sm text-secondary select-none">{label}</span>}
        </div>
    );
};
