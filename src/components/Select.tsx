import React, { useState, useRef, useEffect } from 'react';
import { ChevronDown } from 'lucide-react';

interface Option {
    value: string;
    label: string;
}

interface SelectProps {
    value: string;
    onChange: (value: string) => void;
    options: Option[];
    placeholder?: string;
    disabled?: boolean;
    style?: React.CSSProperties;
    label?: string; // Optional label on top
}

export const Select: React.FC<SelectProps> = ({
    value,
    onChange,
    options,
    placeholder = 'Select...',
    disabled = false,
    style,
    label
}) => {
    const [isOpen, setIsOpen] = useState(false);
    const containerRef = useRef<HTMLDivElement>(null);

    // Close on click outside
    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
                setIsOpen(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    const selectedOption = options.find(o => o.value === value);

    return (
        <div style={{ width: '100%', ...style }} ref={containerRef}>
            {label && (
                <label style={{ display: 'block', fontSize: '13px', marginBottom: '6px', color: '#e4e4e7' }}>{label}</label>
            )}
            <div
                onClick={() => !disabled && setIsOpen(!isOpen)}
                style={{
                    position: 'relative',
                    width: '100%',
                    padding: '10px 14px',
                    background: disabled ? 'rgba(255,255,255,0.02)' : 'rgba(0,0,0,0.3)',
                    border: isOpen ? '1px solid rgba(139,92,246,0.5)' : '1px solid rgba(255,255,255,0.15)',
                    borderRadius: '10px',
                    color: value ? 'white' : '#9ca3af',
                    fontSize: '14px',
                    cursor: disabled ? 'not-allowed' : 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    transition: 'all 0.2s',
                    opacity: disabled ? 0.6 : 1
                }}
            >
                <span style={{
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    maxWidth: 'calc(100% - 24px)'
                }}>
                    {selectedOption ? selectedOption.label : placeholder}
                </span>
                <ChevronDown size={16} color="#9ca3af" style={{ transform: isOpen ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }} />
            </div>

            {isOpen && !disabled && (
                <div style={{
                    position: 'absolute',
                    marginTop: '6px',
                    width: containerRef.current?.offsetWidth,
                    maxHeight: '240px',
                    overflowY: 'auto',
                    background: '#18181b',
                    border: '1px solid rgba(255,255,255,0.1)',
                    borderRadius: '10px',
                    boxShadow: '0 10px 25px rgba(0,0,0,0.5)',
                    zIndex: 1000,
                    padding: '6px'
                }}>
                    {options.length > 0 ? options.map((option) => (
                        <div
                            key={option.value}
                            onClick={() => {
                                onChange(option.value);
                                setIsOpen(false);
                            }}
                            onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(255,255,255,0.05)'}
                            onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
                            style={{
                                padding: '8px 12px',
                                borderRadius: '6px',
                                cursor: 'pointer',
                                fontSize: '14px',
                                color: option.value === value ? '#a78bfa' : '#e4e4e7',
                                background: option.value === value ? 'rgba(139,92,246,0.1)' : 'transparent',
                                transition: 'background 0.1s'
                            }}
                        >
                            {option.label}
                        </div>
                    )) : (
                        <div style={{ padding: '12px', textAlign: 'center', color: '#71717a', fontSize: '13px' }}>
                            No options
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

export default Select;
