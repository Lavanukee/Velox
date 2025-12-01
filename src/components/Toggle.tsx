import React from 'react';
import '../styles/components.css';

interface ToggleProps {
    checked: boolean;
    onChange: (checked: boolean) => void;
    label?: string;
}

export const Toggle: React.FC<ToggleProps> = ({ checked, onChange, label }) => {
    return (
        <div className="flex items-center gap-3 cursor-pointer" onClick={() => onChange(!checked)}>
            <div className={`toggle-switch ${checked ? 'checked' : ''}`}>
                <div className="toggle-thumb" />
            </div>
            {label && <span className="text-sm text-secondary select-none">{label}</span>}
        </div>
    );
};
