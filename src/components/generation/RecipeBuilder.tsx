import React, { useState } from 'react';
import {
    MessageSquare, Image as ImageIcon, Trash2,
    FolderOpen,
    FileText, X, Database, Sparkles, Save, Play, Settings, ArrowDown
} from 'lucide-react';
import { GenerationRecipe, GenerationStep, DEFAULT_RECIPE, InputSource } from '../../types/GenerationRecipe';
import { ModelSelector } from '../Shared/ModelSelector';
import { DragDropEditor, DragDropEditorHandle } from './DragDropEditor';
import { open } from '@tauri-apps/plugin-dialog';

// Helper Component for Step Icons
export const StepIcon: React.FC<{ type: string }> = ({ type }) => {
    switch (type) {
        case 'conversation-turn': return <MessageSquare size={16} />;
        case 'image-generation': return <ImageIcon size={16} />;
        default: return <Settings size={16} />;
    }
};

interface RecipeBuilderProps {
    onSave?: (recipe: GenerationRecipe) => void;
    onRun?: (recipe: GenerationRecipe) => void;
}

export const RecipeBuilder: React.FC<RecipeBuilderProps> = ({ onSave, onRun }) => {
    const [recipe, setRecipe] = useState<GenerationRecipe>({ ...DEFAULT_RECIPE, sources: [] });
    const [selectedStepId, setSelectedStepId] = useState<string | null>(null);
    const editorRef = React.useRef<DragDropEditorHandle>(null);

    // --- Actions ---

    const addSource = async (type: 'file' | 'folder') => {
        try {
            const selected = await open({
                directory: type === 'folder',
                multiple: false,
                title: type === 'folder' ? "Select Folder" : "Select File"
            });

            if (selected && typeof selected === 'string') {
                const name = selected.split(/[\\/]/).pop() || 'Untitled Source';
                const newSource: InputSource = {
                    id: crypto.randomUUID(),
                    name: `Source ${String.fromCharCode(65 + recipe.sources.length)} (${name})`, // Source A, B, C...
                    type,
                    path: selected,
                    iterationMode: 'sequential', // Default
                    allowRepetition: false
                };
                setRecipe(p => ({ ...p, sources: [...p.sources, newSource] }));
            }
        } catch (e) {
            console.error("Failed to add source", e);
        }
    };

    const removeSource = (id: string) => {
        setRecipe(p => ({ ...p, sources: p.sources.filter(s => s.id !== id) }));
    };

    const addStep = (role: 'system' | 'user' | 'assistant') => {
        const newStep: GenerationStep = {
            id: crypto.randomUUID(),
            type: 'conversation-turn',
            name: `${role.charAt(0).toUpperCase() + role.slice(1)} Message`,
            config: {
                role, // Add custom role property to config for internal tracking
                userTemplate: [''],
                assistantTemplate: ['']
            }
        };
        setRecipe(prev => ({ ...prev, steps: [...prev.steps, newStep] }));
        setSelectedStepId(newStep.id);
    };

    const removeStep = (id: string) => {
        setRecipe(prev => ({
            ...prev,
            steps: prev.steps.filter(s => s.id !== id)
        }));
        if (selectedStepId === id) setSelectedStepId(null);
    };

    const updateStepConfig = (id: string, updates: any) => {
        setRecipe(prev => ({
            ...prev,
            steps: prev.steps.map(s => s.id === id ? { ...s, config: { ...s.config, ...updates } } : s)
        }));
    };

    const selectedStep = recipe.steps.find(s => s.id === selectedStepId);

    // --- Drag and Drop Helpers ---
    const handleDragStart = (e: React.DragEvent, type: string, payload: string) => {
        e.dataTransfer.setData('velocity/type', type);
        e.dataTransfer.setData('velocity/payload', payload);
        // Fallback for some environments/browsers
        e.dataTransfer.setData('text/plain', JSON.stringify({ type, payload }));
    };

    const [showPresets, setShowPresets] = useState(false);

    const presets = [
        {
            name: "Web Dev Distiller",
            description: "Generate single-file HTML apps from random ideas.",
            recipe: {
                ...DEFAULT_RECIPE,
                name: "Web Dev Distiller",
                steps: [
                    { id: 's1', type: 'conversation-turn', name: 'System', config: { role: 'system', userTemplate: ['You are an expert web developer. Write single-file HTML apps.'] } },
                    { id: 's2', type: 'conversation-turn', name: 'User Task', config: { role: 'user', userTemplate: ['Write a website for: ', { type: 'generator', prompt: ['Random app idea'] }] } },
                    { id: 's3', type: 'conversation-turn', name: 'Assistant Reply', config: { role: 'assistant', assistantTemplate: [{ type: 'generator', prompt: [''] }] } }
                ]
            }
        },
        {
            name: "Instruction Tuning",
            description: "Standard instruction-response pairs.",
            recipe: {
                ...DEFAULT_RECIPE,
                name: "Instruction Tuning",
                steps: [
                    { id: 's4', type: 'conversation-turn', name: 'Instruction', config: { role: 'user', userTemplate: ['Answer this: ', { type: 'generator', prompt: ['Random question'] }] } },
                    { id: 's5', type: 'conversation-turn', name: 'Response', config: { role: 'assistant', assistantTemplate: [{ type: 'generator', prompt: [''] }] } }
                ]
            }
        }
    ];

    return (
        <div style={{ display: 'flex', height: '100%', background: 'var(--bg-subtle, #18181b)', position: 'relative' }}>

            {showPresets && (
                <div style={{
                    position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
                    background: 'rgba(0,0,0,0.7)', zIndex: 10,
                    display: 'flex', alignItems: 'center', justifyContent: 'center'
                }} onClick={() => setShowPresets(false)}>
                    <div style={{
                        width: '400px', background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)',
                        borderRadius: '8px', padding: '24px', boxShadow: '0 4px 20px rgba(0,0,0,0.3)'
                    }} onClick={e => e.stopPropagation()}>
                        <h3 style={{ fontSize: '18px', fontWeight: 600, marginBottom: '16px' }}>Load Preset</h3>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                            {presets.map((p, i) => (
                                <button
                                    key={i}
                                    onClick={() => {
                                        // Load preset
                                        // Preserve sources if needed? Or replace?
                                        // Usually replace recipe structure but maybe keep sources?
                                        // For now, replace entirely but keep sources if existing? 
                                        // User might want fresh start.
                                        setRecipe({ ...p.recipe, sources: recipe.sources } as GenerationRecipe);
                                        setShowPresets(false);
                                    }}
                                    style={{
                                        textAlign: 'left', padding: '12px',
                                        background: 'var(--bg-elevated)', border: '1px solid var(--border-subtle)',
                                        borderRadius: '6px', cursor: 'pointer',
                                        display: 'flex', flexDirection: 'column', gap: '4px'
                                    }}
                                >
                                    <span style={{ fontWeight: 600, color: 'var(--text-main)' }}>{p.name}</span>
                                    <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>{p.description}</span>
                                </button>
                            ))}
                        </div>
                        <button
                            onClick={() => setShowPresets(false)}
                            style={{ marginTop: '16px', width: '100%', padding: '8px', background: 'transparent', border: '1px solid var(--border-subtle)', color: 'var(--text-muted)', borderRadius: '6px', cursor: 'pointer' }}
                        >
                            Cancel
                        </button>
                    </div>
                </div>
            )}

            {/* LEFT: Timeline & Sources */}
            <div style={{ width: '300px', padding: '16px', borderRight: '1px solid var(--border-subtle)', display: 'flex', flexDirection: 'column', background: 'var(--bg-surface)' }}>
                {/* Header / Presets */}
                <div style={{ marginBottom: '16px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
                        <h3 style={{ fontSize: '14px', fontWeight: 600 }}>Recipe</h3>
                        <div style={{ display: 'flex', gap: '4px' }}>
                            <button
                                onClick={() => onSave?.(recipe)}
                                style={{ padding: '4px', cursor: 'pointer', color: 'var(--text-main)', background: 'none', border: 'none' }}
                                title="Save Recipe"
                            >
                                <Save size={16} />
                            </button>
                            <button
                                onClick={() => setShowPresets(true)}
                                style={{ padding: '4px', cursor: 'pointer', color: 'var(--text-main)', background: 'none', border: 'none' }}
                                title="Load Preset"
                            >
                                <FolderOpen size={16} />
                            </button>
                        </div>
                    </div>

                    <input
                        type="text"
                        value={recipe.name}
                        onChange={(e) => setRecipe(p => ({ ...p, name: e.target.value }))}
                        style={{ background: 'var(--bg-input)', border: '1px solid var(--border-input)', borderRadius: '4px', padding: '6px', color: 'var(--text-main)', width: '100%', fontSize: '13px' }}
                        placeholder="Recipe Name"
                    />
                </div>

                {/* Timeline */}
                <div style={{ flex: 1, overflowY: 'auto', marginBottom: '16px' }}>
                    <h4 style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '8px', textTransform: 'uppercase' }}>Pipeline Steps</h4>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                        {recipe.steps.length === 0 ? (
                            <div style={{ padding: '16px', textAlign: 'center', border: '1px dashed var(--border-subtle)', borderRadius: '6px', color: 'var(--text-muted)', fontSize: '12px' }}>
                                No steps
                            </div>
                        ) : (
                            recipe.steps.map((step, index) => (
                                <React.Fragment key={step.id}>
                                    {index > 0 && <div style={{ display: 'flex', justifyContent: 'center' }}><ArrowDown size={12} style={{ color: 'var(--text-muted)' }} /></div>}
                                    <div
                                        onClick={() => setSelectedStepId(step.id)}
                                        style={{
                                            background: selectedStepId === step.id ? 'var(--bg-elevated)' : 'transparent',
                                            border: selectedStepId === step.id ? '1px solid var(--accent-primary)' : '1px solid var(--border-subtle)',
                                            borderRadius: '6px',
                                            padding: '10px',
                                            cursor: 'pointer',
                                            display: 'flex', alignItems: 'center', gap: '8px'
                                        }}
                                    >
                                        <div style={{
                                            width: '18px', height: '18px', borderRadius: '50%',
                                            background: step.config.role === 'user' ? '#3b82f6' : step.config.role === 'assistant' ? '#10b981' : '#f59e0b',
                                            display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'white', fontSize: '9px', fontWeight: 700
                                        }}>
                                            {step.config.role ? step.config.role[0].toUpperCase() : 'S'}
                                        </div>
                                        <span style={{ fontWeight: 500, fontSize: '13px', flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{step.name}</span>
                                        <button
                                            onClick={(e) => { e.stopPropagation(); removeStep(step.id); }}
                                            style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', padding: 0 }}
                                        >
                                            <Trash2 size={12} />
                                        </button>
                                    </div>
                                </React.Fragment>
                            ))
                        )}
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '4px', marginTop: '4px' }}>
                            <button onClick={() => addStep('system')} style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border-subtle)', borderRadius: '4px', padding: '6px', fontSize: '10px', cursor: 'pointer', color: 'var(--text-main)' }}>+ Sys</button>
                            <button onClick={() => addStep('user')} style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border-subtle)', borderRadius: '4px', padding: '6px', fontSize: '10px', cursor: 'pointer', color: 'var(--text-main)' }}>+ User</button>
                            <button onClick={() => addStep('assistant')} style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border-subtle)', borderRadius: '4px', padding: '6px', fontSize: '10px', cursor: 'pointer', color: 'var(--text-main)' }}>+ Asst</button>
                        </div>
                    </div>
                </div>

                {/* Sources (Moved here) */}
                <div style={{ flex: 1, overflowY: 'auto', borderTop: '1px solid var(--border-subtle)', paddingTop: '16px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
                        <h4 style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Sources</h4>
                        <div style={{ display: 'flex', gap: '4px' }}>
                            <button onClick={() => addSource('file')} title="Add File" style={{ padding: '2px', cursor: 'pointer', color: 'var(--text-muted)', background: 'none', border: 'none' }}><FileText size={14} /></button>
                            <button onClick={() => addSource('folder')} title="Add Folder" style={{ padding: '2px', cursor: 'pointer', color: 'var(--text-muted)', background: 'none', border: 'none' }}><FolderOpen size={14} /></button>
                        </div>
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                        {recipe.sources.map(src => (
                            <div key={src.id} style={{ padding: '8px', background: 'var(--bg-elevated)', borderRadius: '6px', border: '1px solid var(--border-subtle)', fontSize: '12px' }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '6px' }}>
                                    {src.type === 'folder' ? <FolderOpen size={12} style={{ color: '#f59e0b' }} /> : <FileText size={12} style={{ color: '#3b82f6' }} />}
                                    <span style={{ fontWeight: 600, flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{src.name}</span>
                                    <button onClick={() => removeSource(src.id)} style={{ padding: 0, background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)' }}><X size={12} /></button>
                                </div>
                                <div style={{ display: 'flex', gap: '4px' }}>
                                    {['sequential', 'random'].map(mode => (
                                        <button
                                            key={mode}
                                            onClick={() => setRecipe(p => ({ ...p, sources: p.sources.map(s => s.id === src.id ? { ...s, iterationMode: mode as any } : s) }))}
                                            style={{
                                                flex: 1,
                                                padding: '2px', borderRadius: '3px', border: '1px solid',
                                                borderColor: src.iterationMode === mode ? 'var(--accent-primary)' : 'var(--border-subtle)',
                                                background: src.iterationMode === mode ? 'rgba(59, 130, 246, 0.1)' : 'transparent',
                                                color: src.iterationMode === mode ? 'var(--accent-primary)' : 'var(--text-muted)',
                                                cursor: 'pointer', fontSize: '9px', textTransform: 'capitalize'
                                            }}
                                            title={`Set iteration mode to ${mode}`}
                                        >
                                            {mode}
                                        </button>
                                    ))}
                                </div>
                            </div>
                        ))}
                        {recipe.sources.length === 0 && (
                            <p style={{ fontSize: '11px', color: 'var(--text-muted)', textAlign: 'center', fontStyle: 'italic' }}>No sources added</p>
                        )}
                    </div>
                </div>

                {/* Footer Actions */}
                <div style={{ marginTop: 'auto', paddingTop: '16px', borderTop: '1px solid var(--border-subtle)' }}>
                    <button
                        onClick={() => onRun?.(recipe)}
                        style={{ width: '100%', background: 'var(--accent-primary)', borderRadius: '6px', padding: '10px', color: 'white', border: 'none', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', fontWeight: 600 }}
                    >
                        <Play size={16} fill="currentColor" /> Generate
                    </button>
                </div>
            </div>

            {/* CENTER/RIGHT: Editor Area */}
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', background: 'var(--bg-subtle)' }}>
                {selectedStep ? (
                    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
                        {/* Toolbar */}
                        <div style={{
                            padding: '12px 24px',
                            borderBottom: '1px solid var(--border-subtle)',
                            background: 'var(--bg-surface)',
                            display: 'flex',
                            gap: '12px',
                            alignItems: 'center'
                        }}>
                            <span style={{ fontSize: '12px', color: 'var(--text-muted)', fontWeight: 600, marginRight: '4px' }}>INSERT BLOCKS:</span>

                            <button
                                onClick={() => editorRef.current?.insertBlock({ type: 'generator', prompt: [''] })}
                                style={{
                                    display: 'flex', alignItems: 'center', gap: '6px',
                                    padding: '6px 12px', background: 'rgba(99, 102, 241, 0.1)',
                                    border: '1px solid var(--accent-primary)', borderRadius: '6px',
                                    fontSize: '12px', cursor: 'pointer', color: 'var(--accent-primary)',
                                    transition: 'all 0.2s'
                                }}
                                title="Click to insert Generator"
                            >
                                <Sparkles size={14} /> Generator
                            </button>

                            <div style={{ width: '1px', height: '24px', background: 'var(--border-subtle)' }} />

                            {recipe.sources.map(src => (
                                <button
                                    key={src.id}
                                    onClick={() => editorRef.current?.insertBlock({ type: 'source_data', sourceId: src.id })}
                                    style={{
                                        display: 'flex', alignItems: 'center', gap: '6px',
                                        padding: '6px 12px', background: 'rgba(16, 185, 129, 0.1)',
                                        border: '1px solid var(--accent-success, #10b981)', borderRadius: '6px',
                                        fontSize: '12px', cursor: 'pointer', color: 'var(--accent-success, #10b981)',
                                        transition: 'all 0.2s'
                                    }}
                                    title={`Click to insert ${src.name}`}
                                >
                                    <Database size={14} /> {src.name}
                                </button>
                            ))}
                            {recipe.sources.length === 0 && <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>(Add sources in sidebar)</span>}
                        </div>

                        {/* Workspace */}
                        <div style={{ flex: 1, padding: '24px', overflowY: 'auto' }}>
                            <div style={{ maxWidth: '800px', margin: '0 auto' }}>
                                <div style={{ marginBottom: '16px', display: 'flex', justifyContent: 'space-between', alignItems: 'end' }}>
                                    <label style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-main)' }}>
                                        {selectedStep.config.role === 'user' ? 'User Message' : 'Assistant Response'}
                                    </label>
                                    <button onClick={() => setSelectedStepId(null)} style={{ fontSize: '12px', color: 'var(--accent-primary)', background: 'none', border: 'none', cursor: 'pointer' }}>
                                        Close Editor
                                    </button>
                                </div>

                                <DragDropEditor
                                    ref={editorRef}
                                    value={selectedStep.config.role === 'user' ? (selectedStep.config.userTemplate || []) : (selectedStep.config.assistantTemplate || [])}
                                    onChange={(val) => updateStepConfig(selectedStep.id, selectedStep.config.role === 'user' ? { userTemplate: val } : { assistantTemplate: val })}
                                    availableSources={recipe.sources}
                                    placeholder={`Typing...`}
                                    style={{ minHeight: '400px', border: 'none', background: 'transparent', padding: '0', fontSize: '15px' }}
                                />

                                {/* Model Selector for Assistant Steps */}
                                {selectedStep.config.role === 'assistant' && (
                                    <div style={{ marginTop: '24px', background: 'var(--bg-surface)', padding: '16px', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
                                        <label style={{ display: 'block', marginBottom: '8px', fontSize: '12px', color: 'var(--text-muted)' }}>Generation Model</label>
                                        <ModelSelector
                                            value={selectedStep.config.model || ''}
                                            onChange={(val) => updateStepConfig(selectedStep.id, { model: val })}
                                        />
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                ) : (
                    <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', flexDirection: 'column', gap: '16px', opacity: 0.5 }}>
                        <Settings size={48} />
                        <p>Select a step to edit or add a new one.</p>
                    </div>
                )}
            </div>
        </div>
    );
};

export default RecipeBuilder;
