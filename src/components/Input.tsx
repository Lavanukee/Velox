import React from 'react';
import { InfoTooltip } from './InfoTooltip';
import '../styles/components.css';

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
    label?: string;
    tooltip?: string;
    displayValue?: string;
    error?: string;
    fullWidth?: boolean;
}

export const Input: React.FC<InputProps> = ({
    label,
    tooltip,
    displayValue,
    error,
    fullWidth = true,
    className = '',
    ...props
}) => {
    return (
        <div className={`input-wrapper ${fullWidth ? 'w-full' : ''} ${className}`}>
            {label && (
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                    <label className="input-label" style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: 0 }}>
                        {label}
                        {tooltip && <InfoTooltip text={tooltip} />}
                    </label>
                    {displayValue && <span style={{ fontSize: '12px', color: '#a78bfa', fontWeight: 600 }}>{displayValue}</span>}
                </div>
            )}
            <input className="input-field" {...props} />
            {error && <span className="text-xs text-red-500 mt-1">{error}</span>}
        </div>
    );
};

interface SelectProps extends React.SelectHTMLAttributes<HTMLSelectElement> {
    label?: string;
    tooltip?: string;
    options: { value: string; label: string }[];
}

export const Select: React.FC<SelectProps> = ({
    label,
    tooltip,
    options,
    className = '',
    ...props
}) => {
    return (
        <div className={`input-wrapper ${className}`}>
            {label && (
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                    <label className="input-label" style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: 0 }}>
                        {label}
                        {tooltip && <InfoTooltip text={tooltip} />}
                    </label>
                </div>
            )}
            <select className="input-field" {...props}>
                {options.map((opt) => (
                    <option key={opt.value} value={opt.value}>
                        {opt.label}
                    </option>
                ))}
            </select>
        </div>
    );
};
