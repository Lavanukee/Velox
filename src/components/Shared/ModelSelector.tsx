import React, { useEffect, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { ChevronDown, Loader2 } from 'lucide-react';

interface ModelSelectorProps {
    value: string;
    onChange: (value: string) => void;
    placeholder?: string;
}

export const ModelSelector: React.FC<ModelSelectorProps> = ({ value, onChange, placeholder = "Select a model..." }) => {
    const [models, setModels] = useState<string[]>([]);
    const [loading, setLoading] = useState(false);


    useEffect(() => {
        let mounted = true;
        const loadModels = async () => {
            setLoading(true);
            try {
                const ggufs = await invoke<string[]>('list_gguf_models_command');
                if (mounted) setModels(ggufs || []);
            } catch (e) {
                console.error(e);
            } finally {
                if (mounted) setLoading(false);
            }
        };
        loadModels();
        return () => { mounted = false; };
    }, []);

    return (
        <div style={{ position: 'relative' }}>
            <select
                value={value}
                onChange={(e) => onChange(e.target.value)}
                disabled={loading}
                style={{
                    width: '100%',
                    padding: '10px 12px',
                    paddingRight: '32px',
                    background: 'var(--bg-input, #09090b)',
                    border: '1px solid var(--border-input, #27272a)',
                    borderRadius: '6px',
                    color: 'var(--text-main)',
                    fontSize: '13px',
                    appearance: 'none',
                    cursor: 'pointer'
                }}
            >
                <option value="" disabled>{loading ? "Loading..." : placeholder}</option>
                {models.map(m => (
                    <option key={m} value={m}>{m}</option>
                ))}
            </select>
            <div style={{ position: 'absolute', right: '12px', top: '50%', transform: 'translateY(-50%)', pointerEvents: 'none', color: 'var(--text-muted)' }}>
                {loading ? <Loader2 size={14} className="animate-spin" /> : <ChevronDown size={14} />}
            </div>
            {models.length === 0 && !loading && (
                <div style={{ marginTop: '4px', fontSize: '11px', color: 'var(--accent-warning)' }}>
                    No local GGUF models found.
                </div>
            )}
        </div>
    );
};
