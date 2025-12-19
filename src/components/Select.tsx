import React, { useState, useRef, useEffect, useMemo } from 'react';
import { ChevronDown, Search, Eye } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { InfoTooltip } from './InfoTooltip';

export interface Option {
    value: string;
    label: string;
    tags?: string[];
    hasVision?: boolean;
    engine?: 'GGUF' | 'Base';
}

interface SelectProps {
    value: string;
    onChange: (value: string) => void;
    options: Option[];
    placeholder?: string;
    disabled?: boolean;
    style?: React.CSSProperties;
    label?: string;
    tooltip?: string;
    showSearch?: boolean;
    width?: string | number;
}

export const Select: React.FC<SelectProps> = ({
    value,
    onChange,
    options,
    placeholder = 'Select...',
    disabled = false,
    style,
    label,
    tooltip,
    showSearch = false,
    width = '100%'
}) => {
    const [isOpen, setIsOpen] = useState(false);
    const [searchQuery, setSearchQuery] = useState('');
    const containerRef = useRef<HTMLDivElement>(null);
    const searchInputRef = useRef<HTMLInputElement>(null);

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

    // Focus search input when opening
    useEffect(() => {
        if (isOpen && showSearch) {
            setTimeout(() => searchInputRef.current?.focus(), 100);
        } else {
            setSearchQuery('');
        }
    }, [isOpen, showSearch]);

    const filteredOptions = useMemo(() => {
        if (!searchQuery) return options;
        const lowQuery = searchQuery.toLowerCase();
        return options.filter(o =>
            o.label.toLowerCase().includes(lowQuery) ||
            o.value.toLowerCase().includes(lowQuery) ||
            o.tags?.some(t => t.toLowerCase().includes(lowQuery))
        );
    }, [options, searchQuery]);

    const selectedOption = options.find(o => o.value === value);

    const toggleOpen = () => {
        if (!disabled) setIsOpen(!isOpen);
    };

    return (
        <div style={{ width, position: 'relative', ...style }} ref={containerRef}>
            {label && (
                <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px', marginBottom: '6px', color: '#e4e4e7', fontWeight: 500 }}>
                    {label}
                    {tooltip && <InfoTooltip text={tooltip} />}
                </label>
            )}
            <div
                onClick={toggleOpen}
                style={{
                    position: 'relative',
                    width: '100%',
                    padding: '10px 14px',
                    background: disabled ? 'rgba(255,255,255,0.02)' : 'rgba(0,0,0,0.3)',
                    border: isOpen ? '1px solid rgba(139,92,246,0.6)' : '1px solid rgba(255,255,255,0.12)',
                    borderRadius: '12px',
                    color: value ? 'white' : '#9ca3af',
                    fontSize: '14px',
                    cursor: disabled ? 'not-allowed' : 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
                    boxShadow: isOpen ? '0 0 0 3px rgba(139,92,246,0.1)' : 'none',
                    opacity: disabled ? 0.6 : 1
                }}
            >
                <span style={{
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    maxWidth: 'calc(100% - 24px)',
                    fontWeight: selectedOption ? 500 : 400
                }}>
                    {selectedOption ? selectedOption.label : placeholder}
                </span>
                <ChevronDown size={16} color="#9ca3af" style={{ transform: isOpen ? 'rotate(180deg)' : 'none', transition: 'transform 0.3s cubic-bezier(0.4, 0, 0.2, 1)' }} />
            </div>

            <AnimatePresence>
                {isOpen && !disabled && (
                    <motion.div
                        initial={{ opacity: 0, y: -4, scale: 0.98 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: -4, scale: 0.98 }}
                        transition={{ duration: 0.15, ease: [0.4, 0, 0.2, 1] }}
                        style={{
                            position: 'absolute',
                            marginTop: '8px',
                            width: '100%',
                            minWidth: '280px', // Ensure it doesn't get too narrow
                            maxHeight: '350px',
                            background: '#121214',
                            border: '1px solid rgba(255,255,255,0.12)',
                            borderRadius: '14px',
                            boxShadow: '0 20px 40px rgba(0,0,0,0.6), 0 0 0 1px rgba(255,255,255,0.05)',
                            zIndex: 2000,
                            padding: '8px',
                            left: 0,
                            display: 'flex',
                            flexDirection: 'column'
                        }}
                    >
                        {showSearch && (
                            <div style={{ position: 'relative', marginBottom: '8px', padding: '0 4px' }}>
                                <Search size={14} style={{ position: 'absolute', left: '16px', top: '50%', transform: 'translateY(-50%)', color: '#71717a' }} />
                                <input
                                    ref={searchInputRef}
                                    type="text"
                                    placeholder="Search models..."
                                    value={searchQuery}
                                    onChange={(e) => setSearchQuery(e.target.value)}
                                    style={{
                                        width: '100%',
                                        padding: '8px 12px 8px 34px',
                                        background: 'rgba(255,255,255,0.05)',
                                        border: '1px solid rgba(255,255,255,0.1)',
                                        borderRadius: '8px',
                                        color: 'white',
                                        fontSize: '13px',
                                        outline: 'none',
                                    }}
                                />
                            </div>
                        )}

                        <div style={{ overflowY: 'auto', flex: 1, paddingRight: '4px' }} className="custom-scrollbar">
                            {filteredOptions.length > 0 ? filteredOptions.map((option) => (
                                <div
                                    key={option.value}
                                    onClick={() => {
                                        onChange(option.value);
                                        setIsOpen(false);
                                    }}
                                    onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(255,255,255,0.06)'}
                                    onMouseLeave={(e) => e.currentTarget.style.background = option.value === value ? 'rgba(139,92,246,0.12)' : 'transparent'}
                                    style={{
                                        padding: '10px 12px',
                                        borderRadius: '8px',
                                        cursor: 'pointer',
                                        fontSize: '14px',
                                        color: option.value === value ? '#c4b5fd' : '#e4e4e7',
                                        background: option.value === value ? 'rgba(139,92,246,0.12)' : 'transparent',
                                        transition: 'all 0.1s',
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'space-between',
                                        marginBottom: '2px'
                                    }}
                                >
                                    <span style={{
                                        flex: 1,
                                        whiteSpace: 'nowrap',
                                        overflow: 'hidden',
                                        textOverflow: 'ellipsis',
                                        fontWeight: option.value === value ? 500 : 400
                                    }}>
                                        {option.label}
                                    </span>

                                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginLeft: '12px', flexShrink: 0 }}>
                                        {option.hasVision && (
                                            <div style={{
                                                padding: '4px',
                                                background: 'rgba(234,179,8,0.15)',
                                                borderRadius: '6px',
                                                color: '#eab308',
                                                display: 'flex'
                                            }}>
                                                <Eye size={12} />
                                            </div>
                                        )}
                                        {option.engine && (
                                            <div style={{
                                                fontSize: '10px',
                                                fontWeight: 700,
                                                padding: '2px 6px',
                                                background: option.engine === 'GGUF' ? 'rgba(59,130,246,0.15)' : 'rgba(139,92,246,0.15)',
                                                color: option.engine === 'GGUF' ? '#60a5fa' : '#a78bfa',
                                                borderRadius: '4px',
                                                letterSpacing: '0.02em'
                                            }}>
                                                {option.engine}
                                            </div>
                                        )}
                                    </div>
                                </div>
                            )) : (
                                <div style={{ padding: '24px 12px', textAlign: 'center', color: '#71717a', fontSize: '13px' }}>
                                    No models found
                                </div>
                            )}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};

export default Select;
