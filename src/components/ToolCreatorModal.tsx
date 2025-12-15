import React, { useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { X, Save, Code, FileJson, Type, AlertCircle } from 'lucide-react';

interface ToolCreatorModalProps {
    isOpen: boolean;
    onClose: () => void;
    onSaveSuccess: () => void;
}

const ToolCreatorModal: React.FC<ToolCreatorModalProps> = ({ isOpen, onClose, onSaveSuccess }) => {
    const [name, setName] = useState('');
    const [description, setDescription] = useState('');
    const [parameters, setParameters] = useState('{\n    "arg_name": {\n        "type": "string",\n        "description": "Argument description"\n    }\n}');
    const [code, setCode] = useState('def run(**kwargs):\n    # Access arguments via kwargs\n    # Example: arg = kwargs.get("arg_name")\n    \n    return {\n        "result": "Success",\n        "data": "..."\n    }');
    const [error, setError] = useState<string | null>(null);
    const [isSaving, setIsSaving] = useState(false);

    if (!isOpen) return null;

    const handleSave = async () => {
        setError(null);
        setIsSaving(true);
        try {
            // Validate JSON
            try {
                JSON.parse(parameters);
            } catch (e) {
                throw new Error("Parameters must be valid JSON");
            }

            if (!name.trim()) throw new Error("Tool name is required");
            if (!code.trim()) throw new Error("Python code is required");

            const fileContent = `
from typing import Any, Dict

# Tool Definition
TOOL_DEF = {
    'name': '${name}',
    'description': '${description}',
    'parameters': ${parameters}
}

${code}
`;

            await invoke('save_custom_tool_command', {
                toolName: name,
                toolCode: fileContent
            });

            onSaveSuccess();
            onClose();
            // Reset form
            setName('');
            setDescription('');
            setCode('def run(**kwargs):\n    # Access arguments via kwargs\n    # Example: arg = kwargs.get("arg_name")\n    \n    return {\n        "result": "Success",\n        "data": "..."\n    }');
        } catch (err: any) {
            setError(err.message || String(err));
        } finally {
            setIsSaving(false);
        }
    };

    return (
        <div style={{
            position: 'fixed', inset: 0, zIndex: 1000,
            background: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(4px)',
            display: 'flex', alignItems: 'center', justifyContent: 'center'
        }}>
            <div style={{
                width: '800px', maxWidth: '95vw', height: '85vh',
                background: '#18181b', borderRadius: '16px',
                border: '1px solid rgba(255,255,255,0.1)',
                display: 'flex', flexDirection: 'column', overflow: 'hidden',
                boxShadow: '0 20px 50px rgba(0,0,0,0.5)'
            }}>
                {/* Header */}
                <div style={{
                    padding: '20px 24px', borderBottom: '1px solid rgba(255,255,255,0.08)',
                    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                    background: 'rgba(255,255,255,0.02)'
                }}>
                    <h2 style={{ fontSize: '18px', fontWeight: 600, color: 'white', display: 'flex', alignItems: 'center', gap: '10px' }}>
                        <Code size={20} className="text-blue-400" /> Create Custom Tool
                    </h2>
                    <button onClick={onClose} style={{ background: 'none', border: 'none', color: '#9ca3af', cursor: 'pointer' }}>
                        <X size={20} />
                    </button>
                </div>

                {/* Content */}
                <div style={{ flex: 1, overflowY: 'auto', padding: '24px', display: 'flex', flexDirection: 'column', gap: '20px' }}>

                    {error && (
                        <div style={{ padding: '12px', background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.2)', borderRadius: '8px', color: '#fca5a5', display: 'flex', gap: '8px', alignItems: 'center' }}>
                            <AlertCircle size={16} /> {error}
                        </div>
                    )}

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                        <div>
                            <label style={{ display: 'block', fontSize: '12px', fontWeight: 500, color: '#9ca3af', marginBottom: '8px' }}>Tool Name (snake_case)</label>
                            <div style={{ position: 'relative' }}>
                                <Type size={14} style={{ position: 'absolute', left: '12px', top: '50%', transform: 'translateY(-50%)', color: '#6b7280' }} />
                                <input
                                    value={name} onChange={e => setName(e.target.value)}
                                    placeholder="e.g. calculate_fibonacci"
                                    style={{ width: '100%', padding: '10px 12px 10px 36px', background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', color: 'white', outline: 'none' }}
                                />
                            </div>
                        </div>
                        <div>
                            <label style={{ display: 'block', fontSize: '12px', fontWeight: 500, color: '#9ca3af', marginBottom: '8px' }}>Description</label>
                            <input
                                value={description} onChange={e => setDescription(e.target.value)}
                                placeholder="What does this tool do?"
                                style={{ width: '100%', padding: '10px 12px', background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', color: 'white', outline: 'none' }}
                            />
                        </div>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', flex: 1 }}>
                        <div style={{ display: 'flex', flexDirection: 'column' }}>
                            <label style={{ fontSize: '12px', fontWeight: 500, color: '#9ca3af', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                                <FileJson size={14} /> Parameters (JSON Schema)
                            </label>
                            <textarea
                                value={parameters} onChange={e => setParameters(e.target.value)}
                                style={{
                                    flex: 1, width: '100%', padding: '12px',
                                    background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(255,255,255,0.1)',
                                    borderRadius: '8px', color: '#a5b4fc', fontFamily: 'monospace', fontSize: '13px',
                                    resize: 'none', outline: 'none', lineHeight: 1.5
                                }}
                            />
                        </div>
                        <div style={{ display: 'flex', flexDirection: 'column' }}>
                            <label style={{ fontSize: '12px', fontWeight: 500, color: '#9ca3af', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                                <Code size={14} /> Python Implementation
                            </label>
                            <textarea
                                value={code} onChange={e => setCode(e.target.value)}
                                style={{
                                    flex: 1, width: '100%', padding: '12px',
                                    background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(255,255,255,0.1)',
                                    borderRadius: '8px', color: '#86efac', fontFamily: 'monospace', fontSize: '13px',
                                    resize: 'none', outline: 'none', lineHeight: 1.5
                                }}
                            />
                        </div>
                    </div>
                </div>

                {/* Footer */}
                <div style={{
                    padding: '16px 24px', borderTop: '1px solid rgba(255,255,255,0.08)',
                    display: 'flex', justifyContent: 'flex-end', gap: '12px',
                    background: 'rgba(255,255,255,0.02)'
                }}>
                    <button
                        onClick={onClose}
                        style={{ padding: '8px 16px', borderRadius: '8px', background: 'transparent', border: '1px solid rgba(255,255,255,0.1)', color: 'white', cursor: 'pointer' }}
                    >
                        Cancel
                    </button>
                    <button
                        onClick={handleSave}
                        disabled={isSaving}
                        style={{
                            padding: '8px 16px', borderRadius: '8px',
                            background: '#3b82f6', border: 'none',
                            color: 'white', fontWeight: 500, cursor: 'pointer',
                            display: 'flex', alignItems: 'center', gap: '8px',
                            opacity: isSaving ? 0.7 : 1
                        }}
                    >
                        {isSaving ? 'Saving...' : <><Save size={16} /> Save Tool</>}
                    </button>
                </div>
            </div>
        </div>
    );
};

export default ToolCreatorModal;
