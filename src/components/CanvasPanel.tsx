import React, { useState, useEffect } from 'react';
import { Highlight, themes } from 'prism-react-renderer';
import mermaid from 'mermaid';
import { MarkdownRenderer } from './MarkdownRenderer';
import {
    Edit3, Copy, Check, X,
    Plus, PenTool
} from 'lucide-react';

// Initialize mermaid
mermaid.initialize({
    startOnLoad: false,
    theme: 'dark',
    securityLevel: 'loose',
});

export type CanvasMode = 'code' | 'markdown' | 'mermaid';

export interface CanvasArtifact {
    id: string;
    title: string;
    content: string;
    mode: CanvasMode;
    language?: string;
    createdAt: number;
}

interface CanvasPanelProps {
    artifacts: CanvasArtifact[];
    onArtifactsChange: (artifacts: CanvasArtifact[]) => void;
    isVisible: boolean;
    onToggleVisibility: () => void;
    onAnalyzeWithAI?: (content: string, title: string) => void;
}

// Language detection
const detectLanguage = (content: string, hint?: string): string => {
    if (hint) return hint;
    if (content.includes('def ') || content.includes('import ')) return 'python';
    if (content.includes('function ') || content.includes('const ') || content.includes('=>')) return 'javascript';
    if (content.includes('fn ') || content.includes('let mut')) return 'rust';
    if (content.includes('<html') || content.includes('<!DOCTYPE')) return 'html';
    if (content.includes('{') && content.includes('}') && content.includes(':')) return 'json';
    return 'text';
};

// Mermaid renderer
const MermaidDiagram: React.FC<{ content: string }> = ({ content }) => {
    const [svg, setSvg] = useState('');
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const render = async () => {
            try {
                const id = `mermaid-${Date.now()}`;
                const { svg } = await mermaid.render(id, content);
                setSvg(svg);
                setError(null);
            } catch (e: any) {
                setError(e.message || 'Failed to render');
            }
        };
        if (content.trim()) render();
    }, [content]);

    if (error) {
        return (
            <div style={{ padding: 16, background: 'rgba(239,68,68,0.1)', borderRadius: 8, color: '#f87171', fontSize: 12 }}>
                <strong>Diagram Error:</strong> {error}
            </div>
        );
    }

    return (
        <div
            style={{ display: 'flex', justifyContent: 'center', padding: 16, overflow: 'auto' }}
            dangerouslySetInnerHTML={{ __html: svg }}
        />
    );
};

// Code viewer with syntax highlighting
const CodeViewer: React.FC<{
    content: string;
    language: string;
    isEditing: boolean;
    onChange: (content: string) => void;
}> = ({ content, language, isEditing, onChange }) => {
    const lines = content.split('\n');

    if (isEditing) {
        return (
            <div style={{ position: 'relative' }}>
                <div style={{
                    position: 'absolute', left: 0, top: 0, width: 40,
                    background: 'rgba(0,0,0,0.4)', borderRight: '1px solid rgba(255,255,255,0.1)',
                    height: '100%', fontSize: 12, color: '#4b5563', fontFamily: 'monospace', paddingTop: 12
                }}>
                    {lines.map((_, i) => (
                        <div key={i} style={{ padding: '0 8px', textAlign: 'right', lineHeight: '1.5' }}>{i + 1}</div>
                    ))}
                </div>
                <textarea
                    value={content}
                    onChange={(e) => onChange(e.target.value)}
                    spellCheck={false}
                    style={{
                        width: '100%', minHeight: 300, resize: 'vertical',
                        background: 'rgba(0,0,0,0.3)', border: 'none', outline: 'none',
                        color: '#e5e7eb', fontFamily: 'ui-monospace, monospace', fontSize: 13,
                        lineHeight: '1.5', padding: '12px 12px 12px 52px', tabSize: 4
                    }}
                    onKeyDown={(e) => {
                        if (e.key === 'Tab') {
                            e.preventDefault();
                            const start = e.currentTarget.selectionStart;
                            const end = e.currentTarget.selectionEnd;
                            const newVal = content.substring(0, start) + '    ' + content.substring(end);
                            onChange(newVal);
                        }
                    }}
                />
            </div>
        );
    }

    return (
        <Highlight theme={themes.nightOwl} code={content} language={language as any}>
            {({ style, tokens, getLineProps, getTokenProps }) => (
                <pre style={{ ...style, margin: 0, padding: 12, background: 'transparent', fontSize: 13, overflow: 'auto' }}>
                    {tokens.map((line, i) => (
                        <div key={i} {...getLineProps({ line })} style={{ display: 'flex', lineHeight: '1.5' }}>
                            <span style={{ width: 40, flexShrink: 0, color: '#4b5563', userSelect: 'none', textAlign: 'right', paddingRight: 12 }}>
                                {i + 1}
                            </span>
                            <span>{line.map((token, key) => <span key={key} {...getTokenProps({ token })} />)}</span>
                        </div>
                    ))}
                </pre>
            )}
        </Highlight>
    );
};


// Main CanvasPanel component
export const CanvasPanel: React.FC<CanvasPanelProps> = ({
    artifacts,
    onArtifactsChange,
    isVisible,
    onToggleVisibility
}) => {
    const [isEditing, setIsEditing] = useState(false);
    const [copied, setCopied] = useState(false);

    // Always use the first artifact or the only artifact
    const activeArtifact = artifacts[0];

    const updateArtifact = (updates: Partial<CanvasArtifact>) => {
        if (!activeArtifact) {
            // Create initial if none exists
            onArtifactsChange([{
                id: `artifact-${Date.now()}`,
                title: 'Untitled',
                content: '',
                mode: 'code',
                createdAt: Date.now(),
                ...updates
            }]);
            return;
        }
        onArtifactsChange(artifacts.map((a, i) => i === 0 ? { ...a, ...updates } : a));
    };

    const handleCopy = async () => {
        if (activeArtifact) {
            await navigator.clipboard.writeText(activeArtifact.content);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        }
    };


    if (!isVisible) return null; // Controlled by InferencePage now

    return (
        <div style={{
            height: '100%',
            background: 'linear-gradient(180deg, rgba(20,18,28,0.98) 0%, rgba(15,13,22,0.98) 100%)',
            display: 'flex', flexDirection: 'column',
            transition: 'width 0.3s ease',
            position: 'relative'
        }}>
            {/* Header */}
            <div style={{
                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                padding: '12px 16px', borderBottom: '1px solid rgba(255,255,255,0.08)',
                background: 'rgba(0,0,0,0.2)'
            }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <PenTool size={16} style={{ color: '#60a5fa' }} />
                    <span style={{ fontSize: 14, fontWeight: 600 }}>Writing Area</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                    <button
                        onClick={onToggleVisibility}
                        style={{ padding: 6, background: 'transparent', border: 'none', color: '#9ca3af', cursor: 'pointer', borderRadius: '6px' }}
                        onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(255,255,255,0.05)'}
                        onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
                    >
                        <X size={16} />
                    </button>
                </div>
            </div>

            {/* Toolbar */}
            <div style={{
                display: 'flex', alignItems: 'center', gap: 8,
                padding: '8px 12px', borderBottom: '1px solid rgba(255,255,255,0.06)',
                background: 'rgba(0,0,0,0.15)'
            }}>
                {activeArtifact && (
                    <select
                        value={activeArtifact.mode}
                        onChange={(e) => updateArtifact({ mode: e.target.value as CanvasMode })}
                        style={{
                            padding: '4px 8px', fontSize: 11, background: 'rgba(255,255,255,0.05)',
                            border: '1px solid rgba(255,255,255,0.1)', borderRadius: 4,
                            color: '#e5e7eb', cursor: 'pointer'
                        }}
                    >
                        <option value="code">Code</option>
                        <option value="markdown">Markdown</option>
                        <option value="mermaid">Diagram</option>
                    </select>
                )}

                <div style={{ flex: 1 }} />

                <button
                    onClick={() => setIsEditing(!isEditing)}
                    style={{
                        padding: '4px 10px', fontSize: 11, display: 'flex', alignItems: 'center', gap: 4,
                        background: isEditing ? 'rgba(139,92,246,0.15)' : 'rgba(255,255,255,0.05)',
                        border: `1px solid ${isEditing ? 'rgba(139,92,246,0.3)' : 'rgba(255,255,255,0.1)'}`,
                        borderRadius: 4, color: isEditing ? '#60a5fa' : '#9ca3af', cursor: 'pointer'
                    }}
                >
                    <Edit3 size={12} /> {isEditing ? 'Done' : 'Edit'}
                </button>

                <button
                    onClick={handleCopy}
                    style={{
                        padding: '4px 10px', fontSize: 11, display: 'flex', alignItems: 'center', gap: 4,
                        background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)',
                        borderRadius: 4, color: copied ? '#22c55e' : '#9ca3af', cursor: 'pointer'
                    }}
                >
                    {copied ? <Check size={12} /> : <Copy size={12} />}
                    {copied ? 'Copied!' : 'Copy'}
                </button>
            </div>

            {/* Content Area */}
            <div style={{ flex: 1, overflow: 'auto' }}>
                {activeArtifact ? (
                    <>
                        {isEditing && (
                            <input
                                value={activeArtifact.title}
                                onChange={(e) => updateArtifact({ title: e.target.value })}
                                placeholder="Page title..."
                                style={{
                                    width: '100%', padding: '12px 16px', fontSize: 15, fontWeight: 600,
                                    background: 'rgba(0,0,0,0.2)', border: 'none', borderBottom: '1px solid rgba(255,255,255,0.08)',
                                    color: '#e5e7eb', outline: 'none'
                                }}
                            />
                        )}

                        {activeArtifact.mode === 'code' && (
                            <CodeViewer
                                content={activeArtifact.content}
                                language={detectLanguage(activeArtifact.content, activeArtifact.language)}
                                isEditing={isEditing}
                                onChange={(content) => updateArtifact({ content })}
                            />
                        )}

                        {activeArtifact.mode === 'markdown' && (
                            isEditing ? (
                                <textarea
                                    value={activeArtifact.content}
                                    onChange={(e) => updateArtifact({ content: e.target.value })}
                                    placeholder="Start writing..."
                                    style={{
                                        width: '100%', height: 'calc(100% - 50px)', minHeight: 400, resize: 'none',
                                        background: 'transparent', border: 'none', outline: 'none',
                                        color: '#e5e7eb', fontSize: 15, lineHeight: 1.7, padding: 20,
                                        fontFamily: 'inherit'
                                    }}
                                />
                            ) : (
                                <div style={{ padding: 20 }}>
                                    <MarkdownRenderer content={activeArtifact.content} />
                                </div>
                            )
                        )}

                        {activeArtifact.mode === 'mermaid' && (
                            isEditing ? (
                                <textarea
                                    value={activeArtifact.content}
                                    onChange={(e) => updateArtifact({ content: e.target.value })}
                                    placeholder="Mermaid code..."
                                    style={{
                                        width: '100%', height: 'calc(100% - 50px)', minHeight: 400, resize: 'none',
                                        background: 'rgba(0,0,0,0.1)', border: 'none', outline: 'none',
                                        color: '#e5e7eb', fontFamily: 'monospace', fontSize: 13, padding: 20
                                    }}
                                />
                            ) : (
                                <MermaidDiagram content={activeArtifact.content} />
                            )
                        )}
                    </>
                ) : (
                    <div style={{
                        height: '100%', display: 'flex', flexDirection: 'column',
                        alignItems: 'center', justifyContent: 'center', color: '#6b7280', gap: 16
                    }}>
                        <PenTool size={48} style={{ opacity: 0.1 }} />
                        <p style={{ fontSize: 14 }}>The writing area is empty</p>
                        <button
                            onClick={() => updateArtifact({ title: 'New Document', content: '' })}
                            style={{
                                padding: '10px 20px', fontSize: 13, background: 'rgba(139,92,246,0.15)',
                                border: '1px solid rgba(139,92,246,0.3)', borderRadius: 8,
                                color: '#60a5fa', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 8
                            }}
                        >
                            <Plus size={16} /> Start Writing
                        </button>
                    </div>
                )}
            </div>
        </div>
    );
};

export default CanvasPanel;
