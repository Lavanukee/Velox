import React, { useState, useEffect, useRef } from 'react';
import { Highlight, themes } from 'prism-react-renderer';
import mermaid from 'mermaid';
import { MarkdownRenderer } from './MarkdownRenderer';
import {
    FileCode, FileText, Image, Edit3, Copy, Check, X,
    Replace
} from 'lucide-react';

// Initialize mermaid
mermaid.initialize({
    startOnLoad: false,
    theme: 'dark',
    securityLevel: 'loose',
});

export type CanvasMode = 'code' | 'markdown' | 'mermaid';

interface CanvasProps {
    content: string;
    mode: CanvasMode;
    language?: string;
    title?: string;
    onContentChange?: (content: string) => void;
    isEditable?: boolean;
    streamingEdit?: {
        startLine: number;
        endLine: number;
        newContent: string;
        isActive: boolean;
    } | null;
}

// Language detection from file extension or content
const detectLanguage = (content: string, hint?: string): string => {
    if (hint) return hint;
    if (content.includes('def ') || content.includes('import ')) return 'python';
    if (content.includes('function ') || content.includes('const ') || content.includes('=>')) return 'javascript';
    if (content.includes('fn ') || content.includes('let mut')) return 'rust';
    if (content.includes('<html') || content.includes('<!DOCTYPE')) return 'html';
    if (content.includes('{') && content.includes('}') && content.includes(':')) return 'json';
    return 'text';
};

// Mermaid diagram component
const MermaidDiagram: React.FC<{ content: string }> = ({ content }) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const [svg, setSvg] = useState('');
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const renderDiagram = async () => {
            try {
                const id = `mermaid-${Date.now()}`;
                const { svg } = await mermaid.render(id, content);
                setSvg(svg);
                setError(null);
            } catch (e: any) {
                setError(e.message || 'Failed to render diagram');
            }
        };

        if (content.trim()) {
            renderDiagram();
        }
    }, [content]);

    if (error) {
        return (
            <div style={{ padding: '16px', background: 'rgba(239,68,68,0.1)', borderRadius: '8px', color: '#f87171' }}>
                <p style={{ margin: 0, fontSize: '12px' }}>Diagram Error: {error}</p>
                <pre style={{ margin: '8px 0 0', fontSize: '11px', color: '#a1a1aa' }}>{content}</pre>
            </div>
        );
    }

    return (
        <div
            ref={containerRef}
            style={{
                background: 'rgba(255,255,255,0.02)',
                borderRadius: '8px',
                padding: '16px',
                display: 'flex',
                justifyContent: 'center',
                overflow: 'auto'
            }}
            dangerouslySetInnerHTML={{ __html: svg }}
        />
    );
};

// Code editor with syntax highlighting
const CodeEditor: React.FC<{
    content: string;
    language: string;
    isEditing: boolean;
    onChange: (content: string) => void;
    streamingEdit?: CanvasProps['streamingEdit'];
}> = ({ content, language, isEditing, onChange, streamingEdit }) => {
    const textareaRef = useRef<HTMLTextAreaElement>(null);
    const lines = content.split('\n');

    // Apply streaming edit highlighting
    const getLineStyle = (lineNumber: number): React.CSSProperties => {
        if (streamingEdit?.isActive &&
            lineNumber >= streamingEdit.startLine &&
            lineNumber <= streamingEdit.endLine) {
            return {
                background: 'rgba(34,197,94,0.15)',
                borderLeft: '3px solid #22c55e',
                paddingLeft: '9px',
            };
        }
        return { paddingLeft: '12px' };
    };

    if (isEditing) {
        return (
            <div style={{ position: 'relative' }}>
                <div style={{
                    position: 'absolute',
                    left: 0,
                    top: 0,
                    width: '40px',
                    background: 'rgba(0,0,0,0.3)',
                    borderRight: '1px solid rgba(255,255,255,0.1)',
                    height: '100%',
                    display: 'flex',
                    flexDirection: 'column',
                    fontSize: '12px',
                    color: '#6b7280',
                    fontFamily: 'monospace',
                    paddingTop: '12px',
                }}>
                    {lines.map((_, i) => (
                        <div key={i} style={{ padding: '0 8px', textAlign: 'right', lineHeight: '1.5' }}>
                            {i + 1}
                        </div>
                    ))}
                </div>
                <textarea
                    ref={textareaRef}
                    value={content}
                    onChange={(e) => onChange(e.target.value)}
                    spellCheck={false}
                    style={{
                        width: '100%',
                        minHeight: '200px',
                        height: `${Math.max(200, lines.length * 24 + 24)}px`,
                        resize: 'vertical',
                        background: 'rgba(0,0,0,0.3)',
                        border: 'none',
                        outline: 'none',
                        color: '#e5e7eb',
                        fontFamily: 'ui-monospace, monospace',
                        fontSize: '13px',
                        lineHeight: '1.5',
                        padding: '12px 12px 12px 52px',
                        tabSize: 4,
                    }}
                    onKeyDown={(e) => {
                        if (e.key === 'Tab') {
                            e.preventDefault();
                            const start = e.currentTarget.selectionStart;
                            const end = e.currentTarget.selectionEnd;
                            const newValue = content.substring(0, start) + '    ' + content.substring(end);
                            onChange(newValue);
                            setTimeout(() => {
                                textareaRef.current?.setSelectionRange(start + 4, start + 4);
                            }, 0);
                        }
                    }}
                />
            </div>
        );
    }

    return (
        <Highlight theme={themes.nightOwl} code={content} language={language as any}>
            {({ style, tokens, getLineProps, getTokenProps }) => (
                <pre style={{
                    ...style,
                    margin: 0,
                    padding: 0,
                    background: 'transparent',
                    fontSize: '13px',
                    overflow: 'auto',
                }}>
                    {tokens.map((line, i) => (
                        <div
                            key={i}
                            {...getLineProps({ line })}
                            style={{
                                display: 'flex',
                                lineHeight: '1.5',
                                ...getLineStyle(i + 1),
                            }}
                        >
                            <span style={{
                                width: '40px',
                                flexShrink: 0,
                                color: '#4b5563',
                                userSelect: 'none',
                                textAlign: 'right',
                                paddingRight: '12px',
                            }}>
                                {i + 1}
                            </span>
                            <span>
                                {line.map((token, key) => (
                                    <span key={key} {...getTokenProps({ token })} />
                                ))}
                            </span>
                        </div>
                    ))}
                </pre>
            )}
        </Highlight>
    );
};

export const Canvas: React.FC<CanvasProps> = ({
    content,
    mode,
    language,
    title,
    onContentChange,
    isEditable = true,
    streamingEdit,
}) => {
    const [isEditing, setIsEditing] = useState(false);
    const [localContent, setLocalContent] = useState(content);
    const [copied, setCopied] = useState(false);
    const detectedLanguage = detectLanguage(content, language);

    useEffect(() => {
        setLocalContent(content);
    }, [content]);

    // Apply streaming edit
    useEffect(() => {
        if (streamingEdit?.isActive && streamingEdit.newContent) {
            const lines = localContent.split('\n');
            const before = lines.slice(0, streamingEdit.startLine - 1);
            const after = lines.slice(streamingEdit.endLine);
            const newLines = [...before, ...streamingEdit.newContent.split('\n'), ...after];
            setLocalContent(newLines.join('\n'));
        }
    }, [streamingEdit]);

    const handleCopy = async () => {
        await navigator.clipboard.writeText(localContent);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    const handleSave = () => {
        setIsEditing(false);
        onContentChange?.(localContent);
    };

    const handleCancel = () => {
        setIsEditing(false);
        setLocalContent(content);
    };

    const getModeIcon = () => {
        switch (mode) {
            case 'code': return <FileCode size={14} />;
            case 'markdown': return <FileText size={14} />;
            case 'mermaid': return <Image size={14} />;
        }
    };

    return (
        <div style={{
            border: '1px solid rgba(255,255,255,0.1)',
            borderRadius: '12px',
            overflow: 'hidden',
            background: 'rgba(15,15,20,0.8)',
            margin: '12px 0',
        }}>
            {/* Header */}
            <div style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '10px 14px',
                borderBottom: '1px solid rgba(255,255,255,0.08)',
                background: 'rgba(0,0,0,0.3)',
            }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    {getModeIcon()}
                    <span style={{ fontSize: '13px', fontWeight: 500 }}>
                        {title || (mode === 'code' ? detectedLanguage.toUpperCase() : mode.charAt(0).toUpperCase() + mode.slice(1))}
                    </span>
                    {mode === 'code' && (
                        <span style={{
                            fontSize: '11px',
                            color: '#6b7280',
                            background: 'rgba(255,255,255,0.05)',
                            padding: '2px 6px',
                            borderRadius: '4px'
                        }}>
                            {localContent.split('\n').length} lines
                        </span>
                    )}
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    {isEditing ? (
                        <>
                            <button onClick={handleSave} style={{
                                padding: '4px 10px',
                                background: 'rgba(34,197,94,0.15)',
                                border: '1px solid rgba(34,197,94,0.3)',
                                borderRadius: '6px',
                                color: '#22c55e',
                                cursor: 'pointer',
                                fontSize: '12px',
                                display: 'flex',
                                alignItems: 'center',
                                gap: '4px'
                            }}>
                                <Check size={12} /> Save
                            </button>
                            <button onClick={handleCancel} style={{
                                padding: '4px 10px',
                                background: 'rgba(255,255,255,0.05)',
                                border: '1px solid rgba(255,255,255,0.1)',
                                borderRadius: '6px',
                                color: '#9ca3af',
                                cursor: 'pointer',
                                fontSize: '12px',
                                display: 'flex',
                                alignItems: 'center',
                                gap: '4px'
                            }}>
                                <X size={12} /> Cancel
                            </button>
                        </>
                    ) : (
                        <>
                            {isEditable && (
                                <button onClick={() => setIsEditing(true)} style={{
                                    padding: '4px 8px',
                                    background: 'rgba(255,255,255,0.05)',
                                    border: '1px solid rgba(255,255,255,0.1)',
                                    borderRadius: '6px',
                                    color: '#9ca3af',
                                    cursor: 'pointer',
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: '4px',
                                    fontSize: '12px'
                                }}>
                                    <Edit3 size={12} /> Edit
                                </button>
                            )}
                            <button onClick={handleCopy} style={{
                                padding: '4px 8px',
                                background: 'rgba(255,255,255,0.05)',
                                border: '1px solid rgba(255,255,255,0.1)',
                                borderRadius: '6px',
                                color: copied ? '#22c55e' : '#9ca3af',
                                cursor: 'pointer',
                                display: 'flex',
                                alignItems: 'center',
                                gap: '4px',
                                fontSize: '12px'
                            }}>
                                {copied ? <Check size={12} /> : <Copy size={12} />}
                                {copied ? 'Copied' : 'Copy'}
                            </button>
                        </>
                    )}
                </div>
            </div>

            {/* Content */}
            <div style={{ padding: mode === 'code' ? '0' : '16px', maxHeight: '500px', overflow: 'auto' }}>
                {mode === 'code' && (
                    <CodeEditor
                        content={localContent}
                        language={detectedLanguage}
                        isEditing={isEditing}
                        onChange={setLocalContent}
                        streamingEdit={streamingEdit}
                    />
                )}
                {mode === 'markdown' && (
                    isEditing ? (
                        <textarea
                            value={localContent}
                            onChange={(e) => setLocalContent(e.target.value)}
                            style={{
                                width: '100%',
                                minHeight: '200px',
                                background: 'transparent',
                                border: 'none',
                                outline: 'none',
                                color: '#e5e7eb',
                                fontSize: '14px',
                                lineHeight: 1.6,
                                resize: 'vertical',
                            }}
                        />
                    ) : (
                        <MarkdownRenderer content={localContent} />
                    )
                )}
                {mode === 'mermaid' && (
                    isEditing ? (
                        <textarea
                            value={localContent}
                            onChange={(e) => setLocalContent(e.target.value)}
                            placeholder="Enter mermaid diagram code..."
                            style={{
                                width: '100%',
                                minHeight: '150px',
                                background: 'rgba(0,0,0,0.2)',
                                border: '1px solid rgba(255,255,255,0.1)',
                                borderRadius: '8px',
                                outline: 'none',
                                color: '#e5e7eb',
                                fontFamily: 'monospace',
                                fontSize: '13px',
                                padding: '12px',
                                resize: 'vertical',
                            }}
                        />
                    ) : (
                        <MermaidDiagram content={localContent} />
                    )
                )}
            </div>

            {/* Streaming edit indicator */}
            {streamingEdit?.isActive && (
                <div style={{
                    padding: '8px 14px',
                    borderTop: '1px solid rgba(255,255,255,0.08)',
                    background: 'rgba(34,197,94,0.1)',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    fontSize: '12px',
                    color: '#22c55e'
                }}>
                    <Replace size={14} />
                    <span>Editing lines {streamingEdit.startLine}-{streamingEdit.endLine}...</span>
                </div>
            )}
        </div>
    );
};

export default Canvas;
