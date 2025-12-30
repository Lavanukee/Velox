import React, { useState, useEffect, useRef, useMemo } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { useApp } from '../context/AppContext';
import { useAppState } from '../context/AppStateContext';
import { Button } from '../components/Button';
import { Input } from '../components/Input';
import { MarkdownRenderer } from '../components/MarkdownRenderer';
import { CanvasPanel } from '../components/CanvasPanel';
import ToolCreatorModal from '../components/ToolCreatorModal';
import { Select } from '../components/Select';
import {
    Settings2, Trash2, Edit2, RotateCcw,
    ChevronDown, Cpu, Zap, HardDrive, Layers, Terminal,
    Loader2, CheckCircle2, XCircle, Power, Paperclip, X,
    Settings, Wrench, BarChart2, Clock, PenTool, Search, Code, Brain, Activity,
    ArrowUp, PlusCircle, Play
} from 'lucide-react';
import type { Option } from '../components/Select';

const ThinkingBlock: React.FC<{ content: string }> = ({ content }) => {
    const [expanded, setExpanded] = useState(false);
    return (
        <div style={{ marginBottom: '12px' }}>
            <button
                onClick={() => setExpanded(!expanded)}
                style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    padding: '8px 12px',
                    background: 'rgba(139,92,246,0.15)',
                    border: '1px solid rgba(139,92,246,0.3)',
                    borderRadius: '8px',
                    color: '#a78bfa',
                    cursor: 'pointer',
                    fontSize: '13px',
                    width: '100%',
                    textAlign: 'left'
                }}
            >
                <Brain size={14} />
                <span style={{ flex: 1 }}>Thinking...</span>
                <ChevronDown size={14} style={{ transform: expanded ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }} />
            </button>
            {expanded && (
                <div style={{
                    marginTop: '8px',
                    padding: '12px',
                    background: 'rgba(0,0,0,0.3)',
                    borderRadius: '8px',
                    fontSize: '13px',
                    color: '#a1a1aa',
                    lineHeight: 1.6,
                    whiteSpace: 'pre-wrap',
                    borderLeft: '3px solid rgba(139,92,246,0.5)'
                }}>
                    {content}
                </div>
            )}
        </div>
    );
};

const ToolCallBlock: React.FC<{ content: string }> = ({ content }) => {
    const [expanded, setExpanded] = useState(false);
    // Extract tool name if possible
    const nameMatch = content.match(/<name>([\s\S]*?)<\/name>/i);
    const toolName = nameMatch ? nameMatch[1].trim() : 'Tool Call';

    return (
        <div style={{ marginBottom: '12px' }}>
            <button
                onClick={() => setExpanded(!expanded)}
                style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    padding: '8px 12px',
                    background: 'rgba(59,130,246,0.15)',
                    border: '1px solid rgba(59,130,246,0.3)',
                    borderRadius: '8px',
                    color: '#60a5fa',
                    cursor: 'pointer',
                    fontSize: '13px',
                    width: '100%',
                    textAlign: 'left'
                }}
            >
                <Wrench size={14} />
                <span style={{ flex: 1 }}>Calling Tool: <strong>{toolName}</strong></span>
                <ChevronDown size={14} style={{ transform: expanded ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }} />
            </button>
            {expanded && (
                <div style={{
                    marginTop: '8px',
                    padding: '12px',
                    background: 'rgba(0,0,0,0.3)',
                    borderRadius: '8px',
                    fontSize: '13px',
                    color: '#34d399', // Greenish for code/args
                    lineHeight: 1.6,
                    whiteSpace: 'pre-wrap',
                    borderLeft: '3px solid rgba(59,130,246,0.5)',
                    fontFamily: 'monospace'
                }}>
                    {content}
                </div>
            )}
        </div>
    );
};

const ToolResultBlock: React.FC<{ content: string }> = ({ content }) => {
    const [expanded, setExpanded] = useState(false);
    return (
        <div style={{ marginBottom: '12px' }}>
            <button
                onClick={() => setExpanded(!expanded)}
                style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    padding: '8px 12px',
                    background: 'rgba(16,185,129,0.15)',
                    border: '1px solid rgba(16,185,129,0.3)',
                    borderRadius: '8px',
                    color: '#34d399',
                    cursor: 'pointer',
                    fontSize: '13px',
                    width: '100%',
                    textAlign: 'left'
                }}
            >
                <CheckCircle2 size={14} />
                <span style={{ flex: 1 }}>Tool Result</span>
                <ChevronDown size={14} style={{ transform: expanded ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }} />
            </button>
            {expanded && (
                <div style={{
                    marginTop: '8px',
                    padding: '12px',
                    background: 'rgba(0,0,0,0.3)',
                    borderRadius: '8px',
                    fontSize: '12px',
                    color: '#e5e7eb',
                    lineHeight: 1.6,
                    whiteSpace: 'pre-wrap',
                    borderLeft: '3px solid rgba(16,185,129,0.5)'
                }}>
                    {content}
                </div>
            )}
        </div>
    );
};

// Parse message content for various tags
type ContentBlock =
    | { type: 'text'; content: string }
    | { type: 'thinking'; content: string }
    | { type: 'tool_call'; content: string }
    | { type: 'tool_result'; content: string };

const parseMessageContentBlocks = (text: string): ContentBlock[] => {
    const blocks: ContentBlock[] = [];

    // Combined regex for all tags
    // Matches <think>...</think>, <tool_call>...</tool_call>, <tool_result>...</tool_result>
    const tagRegex = /<(think|tool_call|tool_result)>([\s\S]*?)<\/\1>/gi;

    let match;
    let lastIndex = 0;

    while ((match = tagRegex.exec(text)) !== null) {
        // Add text before the tag
        const textBefore = text.slice(lastIndex, match.index).trim();
        if (textBefore) {
            blocks.push({ type: 'text', content: textBefore });
        }

        const tagName = match[1].toLowerCase();
        const content = match[2].trim();

        if (tagName === 'think') {
            blocks.push({ type: 'thinking', content });
        } else if (tagName === 'tool_call') {
            blocks.push({ type: 'tool_call', content });
        } else if (tagName === 'tool_result') {
            blocks.push({ type: 'tool_result', content });
        }

        lastIndex = tagRegex.lastIndex;
    }

    // Add remaining text
    const textAfter = text.slice(lastIndex).trim();
    if (textAfter) {
        blocks.push({ type: 'text', content: textAfter });
    }

    if (blocks.length === 0 && text.trim()) {
        blocks.push({ type: 'text', content: text });
    }

    return blocks;
};

interface MessageItemProps {
    msg: any;
    idx: number;
    isSecondary: boolean;
    editingIndex: number | null;
    editText: string;
    setEditText: (t: string) => void;
    handleEditSave: (i: number) => void;
    handleEditStart: (i: number) => void;
    handleDeleteMessage: (i: number) => void;
    setEditingIndex: (i: number | null) => void;
    streamMetrics: any;
    infShowMetrics: boolean;
    isLast: boolean;
}

const MessageItem: React.FC<MessageItemProps> = ({
    msg, idx, editingIndex, editText, setEditText,
    handleEditSave, handleEditStart, handleDeleteMessage, setEditingIndex,
    streamMetrics, infShowMetrics, isLast
}) => {
    const isUser = msg.sender === 'user';
    const [hovering, setHovering] = useState(false);

    return (
        <div
            style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: isUser ? 'flex-end' : 'flex-start', // Model left-aligned, user right-aligned
                width: '100%',
                marginBottom: '16px'
            }}
            onMouseEnter={() => setHovering(true)}
            onMouseLeave={() => setHovering(false)}
        >
            {editingIndex === idx ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', width: isUser ? '70%' : '90%' }}>
                    <textarea
                        value={editText}
                        onChange={(e) => setEditText(e.target.value)}
                        style={{ background: 'rgba(0,0,0,0.4)', border: '1px solid rgba(255,255,255,0.2)', borderRadius: '10px', padding: '12px', color: 'white', minHeight: '80px', resize: 'vertical', width: '100%' }}
                    />
                    <div style={{ display: 'flex', gap: '8px' }}>
                        <Button size="sm" onClick={() => handleEditSave(idx)}>Save</Button>
                        <Button size="sm" variant="ghost" onClick={() => setEditingIndex(null)}>Cancel</Button>
                    </div>
                </div>
            ) : (
                <>
                    {/* Message Content */}
                    <div style={{
                        maxWidth: isUser ? '70%' : '90%',
                        padding: isUser ? '14px 18px' : '16px 0',
                        borderRadius: isUser ? '16px' : '0',
                        // User: gradient border effect using background trick
                        background: isUser
                            ? 'linear-gradient(135deg, rgba(10,10,14,1) 0%, rgba(20,18,28,1) 100%)'
                            : 'transparent',
                        border: isUser
                            ? '1px solid transparent'
                            : 'none',
                        backgroundImage: isUser
                            ? 'linear-gradient(135deg, rgba(10,10,14,1), rgba(20,18,28,1)), linear-gradient(135deg, rgba(40,40,50,0.6) 0%, rgba(139,92,246,0.3) 100%)'
                            : 'none',
                        backgroundOrigin: 'border-box',
                        backgroundClip: isUser ? 'padding-box, border-box' : 'unset',
                        color: '#e5e7eb',
                        textAlign: isUser ? 'right' : 'left',
                    }}>
                        {msg.sender === 'bot' ? (() => {
                            const blocks = parseMessageContentBlocks(msg.text);
                            return (
                                <>
                                    {blocks.map((block, bIdx) => {
                                        if (block.type === 'thinking') return <ThinkingBlock key={bIdx} content={block.content} />;
                                        if (block.type === 'tool_call') return <ToolCallBlock key={bIdx} content={block.content} />;
                                        if (block.type === 'tool_result') return <ToolResultBlock key={bIdx} content={block.content} />;
                                        return <MarkdownRenderer key={bIdx} content={block.content} />;
                                    })}
                                </>
                            );
                        })() : (
                            <p style={{ whiteSpace: 'pre-wrap', lineHeight: 1.6, margin: 0 }}>{msg.text}</p>
                        )}
                        {msg.sender === 'bot' && !msg.isStreaming && streamMetrics && isLast && infShowMetrics && (
                            <div style={{ fontSize: '11px', color: '#6b7280', marginTop: '10px', paddingTop: '8px', borderTop: '1px solid rgba(255,255,255,0.1)', display: 'flex', gap: '12px', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'center' }}>
                                <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}><Zap size={12} /> {streamMetrics.tokens_per_second.toFixed(1)} t/s</span>
                                <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}><BarChart2 size={12} /> {streamMetrics.total_tokens} tokens</span>
                                <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}><Clock size={12} /> Prompt: {streamMetrics.prompt_eval_time_ms.toFixed(0)}ms</span>
                                <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}><Clock size={12} /> Eval: {streamMetrics.eval_time_ms.toFixed(0)}ms</span>
                            </div>
                        )}
                    </div>

                    {/* Edit/Delete Buttons - Icon only, circular, hover effect */}
                    <div style={{
                        display: 'flex',
                        gap: '8px',
                        marginTop: '8px',
                        justifyContent: isUser ? 'flex-end' : 'center',
                        opacity: hovering ? 1 : 0,
                        transition: 'opacity 0.2s'
                    }}>
                        <button
                            onClick={() => handleEditStart(idx)}
                            style={{
                                width: '28px',
                                height: '28px',
                                padding: 0,
                                background: 'rgba(255,255,255,0.05)',
                                border: '1px solid rgba(255,255,255,0.1)',
                                borderRadius: '50%',
                                color: 'rgba(255,255,255,0.5)',
                                cursor: 'pointer',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                transition: 'all 0.2s'
                            }}
                            className="icon-btn"
                        >
                            <Edit2 size={12} />
                        </button>
                        <button
                            onClick={() => handleDeleteMessage(idx)}
                            style={{
                                width: '28px',
                                height: '28px',
                                padding: 0,
                                background: 'rgba(239,68,68,0.1)',
                                border: '1px solid rgba(239,68,68,0.2)',
                                borderRadius: '50%',
                                color: 'rgba(239,68,68,0.7)',
                                cursor: 'pointer',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                transition: 'all 0.2s'
                            }}
                            className="icon-btn-danger"
                        >
                            <Trash2 size={12} />
                        </button>
                    </div>
                </>
            )}
        </div>
    );
};

interface InferencePageProps {
    modelConfig: any;
    addLogMessage: (message: string) => void;
}

interface CheckpointInfo {
    name: string;
    path: string;
    is_final: boolean;
    step_number: number | null;
}

interface ProjectLoraInfo {
    project_name: string;
    checkpoints: CheckpointInfo[];
}

// Helper for IDs
const generateId = () => Date.now().toString(36) + Math.random().toString(36).substr(2);

const InferencePage: React.FC<InferencePageProps> = ({ modelConfig, addLogMessage }) => {
    const { loadModel, state } = useAppState();
    const loadedModels = state.inference.loadedModels;

    const {
        userMode,
        chatMessages, setChatMessages,
        inputMessage, setInputMessage,
        selectedBaseModel, setSelectedBaseModel,
        selectedLoraAdapter, setSelectedLoraAdapter,
        temperature, setTemperature,
        contextSize, setContextSize,
        topP, setTopP,
        topK, setTopK,
        systemPrompt, setSystemPrompt,
        isServerRunning, setIsServerRunning,
        infFlashAttn, setInfFlashAttn,
        infNoMmap, setInfNoMmap,
        infGpuLayers, setInfGpuLayers,
        infBatchSize, setInfBatchSize,
        infUbatchSize, setInfUbatchSize,
        infThreads, setInfThreads,
        infServerStatus, setInfServerStatus,
        infShowMetrics, setInfShowMetrics,
        infEnableWebSearch, setInfEnableWebSearch,
        infEnableCodeExec, setInfEnableCodeExec,
        infEnableCanvas, setInfEnableCanvas,
        infCanvasVisible, setInfCanvasVisible,
        infCanvasArtifacts, setInfCanvasArtifacts,
        infInferenceEngine, setInfInferenceEngine,
        isEngineUpdating, setupProgressPercent, setupMessage,
        // setLoadedModels, loadedModels, // Removed from AppContext
        evaluationMode, setEvaluationMode,
        arenaSelectedModels, setArenaSelectedModels,
        arenaScores, setArenaScores,
        arenaCurrentPair, setArenaCurrentPair,
        benchmarkMessages, setBenchmarkMessages,
        selectedBenchmarkModel, setSelectedBenchmarkModel
    } = useApp();

    const isBenchmarking = evaluationMode === 'compare';
    const isBlindTest = evaluationMode === 'arena';

    const [showTools, setShowTools] = useState(false);

    const messagesEndRef = useRef<HTMLDivElement>(null);
    const textareaRef = useRef<HTMLTextAreaElement>(null);

    // Resources
    const [modelOptions, setModelOptions] = useState<Option[]>([]);
    const [projectLoras, setProjectLoras] = useState<ProjectLoraInfo[]>([]);
    const [selectedProject, setSelectedProject] = useState('');
    const [selectedCheckpoint, setSelectedCheckpoint] = useState('');
    const [benchmarkLoraAdapter, setBenchmarkLoraAdapter] = useState('');

    // Backend Logs / Progress
    const [promptProgress, setPromptProgress] = useState<number | null>(null); // 0-100 or null

    const {
        streamMetrics, isPromptProcessing,
        isSending: isSendingGlobal, setIsSending, setIsPromptProcessing
    } = useApp();

    // Vision
    const [pendingImage, setPendingImage] = useState<string | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const [showToolCreator, setShowToolCreator] = useState(false);

    // UI State
    // const [isSending, setIsSending] = useState(false); // Replaced by global
    // Alias it for easier refactoring if needed, or just use isSendingGlobal
    const isSending = isSendingGlobal;

    const [showSidebar, setShowSidebar] = useState(true);
    const [showAdvanced, setShowAdvanced] = useState(false);
    const [showCommandPreview, setShowCommandPreview] = useState(false);
    const [editingIndex, setEditingIndex] = useState<number | null>(null);
    const [editText, setEditText] = useState('');

    // Prompt Input State

    const [secondaryBenchStatus, setSecondaryBenchStatus] = useState<'idle' | 'loading' | 'ready' | 'error'>('idle');
    const [secondaryBenchPort] = useState(8081);
    const [arenaRevealed, setArenaRevealed] = useState(false);
    const [showToolsDropup, setShowToolsDropup] = useState(false);

    // Derived states
    // const isBenchmarking = evaluationMode === 'compare'; // Already derived above
    // const isBlindTest = evaluationMode === 'arena'; // Already derived above
    // const [revealIdentity, setRevealIdentity] = useState(false); // Unused for now

    // We already have 'isBenchmarking' (view toggle)
    // We already have 'benchmarkMessages' (chat history for B)
    // Actually, I removed them intentionally to verify. Let's keep them removed if unused, or restore if used.
    // The previous error logs showed benchmarkStreamMetrics usage.
    const [benchmarkStreamMetrics, setBenchmarkStreamMetrics] = useState<{
        tokens_per_second: number;
        prompt_eval_time_ms: number;
        eval_time_ms: number;
        total_tokens: number;
    } | null>(null);

    // Layout Resizing
    const [canvasWidth, setCanvasWidth] = useState(450);
    const [settingsWidth, setSettingsWidth] = useState(320);
    const [isResizingCanvas, setIsResizingCanvas] = useState(false);
    const [isResizingSettings, setIsResizingSettings] = useState(false);

    // Refs for event handlers to access current state without re-binding
    const stateRef = React.useRef({
        canvasWidth,
        settingsWidth,
        showSidebar,
        isResizingCanvas,
        isResizingSettings
    });

    // Update refs when state changes
    useEffect(() => {
        stateRef.current = {
            canvasWidth,
            settingsWidth,
            showSidebar,
            isResizingCanvas,
            isResizingSettings
        };
    }, [canvasWidth, settingsWidth, showSidebar, isResizingCanvas, isResizingSettings]);

    // Handle resizing
    useEffect(() => {
        const handleMouseMove = (e: MouseEvent) => {
            const { isResizingSettings, isResizingCanvas, showSidebar, settingsWidth } = stateRef.current;

            if (isResizingSettings) {
                // Settings is on the right. Width = WindowWidth - MouseX
                const newWidth = window.innerWidth - e.clientX;
                if (newWidth > 250 && newWidth < 600) {
                    setSettingsWidth(newWidth);
                }
            } else if (isResizingCanvas) {
                // Canvas is middle.
                const rightEdge = window.innerWidth - (showSidebar ? settingsWidth : 0);
                const newWidth = rightEdge - e.clientX;
                if (newWidth > 300 && newWidth < 1200) {
                    setCanvasWidth(newWidth);
                }
            }
        };

        const handleMouseUp = () => {
            setIsResizingCanvas(false);
            setIsResizingSettings(false);
            document.body.style.cursor = 'default';
        };

        if (isResizingCanvas || isResizingSettings) {
            window.addEventListener('mousemove', handleMouseMove);
            window.addEventListener('mouseup', handleMouseUp);
            document.body.style.cursor = 'col-resize';
            document.body.style.userSelect = 'none';
        }

        return () => {
            window.removeEventListener('mousemove', handleMouseMove);
            window.removeEventListener('mouseup', handleMouseUp);
            document.body.style.cursor = 'default';
            document.body.style.userSelect = 'auto';
        };
    }, [isResizingCanvas, isResizingSettings]);


    // Re-load resources when engine changes
    useEffect(() => {
        loadResources();
    }, [infInferenceEngine]);

    // Sync UI selection with global state when a model is loaded
    useEffect(() => {
        const globalSelected = state.inference.selectedModelPath;
        if (globalSelected && globalSelected !== selectedBaseModel) {
            setSelectedBaseModel(globalSelected);
        }
    }, [state.inference.selectedModelPath]);

    // Max GPU Layers (Dynamic)
    const [maxGpuLayers, setMaxGpuLayers] = useState(200);

    // Listen to logs for dynamic GPU layer detection
    useEffect(() => {
        const unlisten = listen('log', (event: any) => {
            const logMsg = event.payload as string;
            const match = /offloaded (\d+)\/(\d+) layers/.exec(logMsg);
            if (match) {
                const total = parseInt(match[2]);
                setMaxGpuLayers(total);
                if (infGpuLayers > total) setInfGpuLayers(total);
            }
        });
        return () => {
            unlisten.then(f => f());
        };
    }, [infGpuLayers]);

    const handleClearConversation = async () => {
        try {
            await invoke('clear_chat_history_command');
            setChatMessages([]);
            addLogMessage('Chat history cleared');
        } catch (error) {
            addLogMessage(`Error clearing chat history: ${error}`);
        }
    };

    const handleEditStart = (index: number) => {
        setEditingIndex(index);
        setEditText(chatMessages[index].text);
    };

    const handleEditSave = (index: number) => {
        setChatMessages(prev => prev.map((msg, i) =>
            i === index ? { ...msg, text: editText } : msg
        ));
        setEditingIndex(null);
        setEditText('');
    };

    const handleDeleteMessage = (index: number) => {
        setChatMessages(prev => prev.filter((_, i) => i !== index));
        if (isBenchmarking) {
            setBenchmarkMessages(prev => prev.filter((_, i) => i !== index));
        }
    };

    // Auto-scroll
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [chatMessages]);

    // AUTO-START: When model changes, start server automatically
    // Also auto-switch engine based on model type
    // Use ref to prevent race condition causing double starts
    const isAutoStartingRef = useRef(false);
    useEffect(() => {
        const restart = async () => {
            if (selectedBaseModel) {
                // Prevent duplicate starts from rapid re-renders
                if (isAutoStartingRef.current) {
                    addLogMessage(`[DEBUG] Auto-start already in progress, skipping.`);
                    return;
                }

                // Determine correct engine based on model type
                const selectedOption = modelOptions.find(o => o.value === selectedBaseModel);
                const isGGUF = selectedOption?.engine === 'GGUF' || selectedBaseModel.endsWith('.gguf');
                const correctEngine = isGGUF ? 'llamacpp' : 'transformers';

                // Auto-switch engine if needed
                if (infInferenceEngine !== correctEngine) {
                    addLogMessage(`[AUTO] Switching engine to ${correctEngine} for ${isGGUF ? 'GGUF' : 'safetensors'} model.`);
                    setInfInferenceEngine(correctEngine);
                    // Engine switch triggers resource reload, server start will happen on next effect
                    return;
                }

                // Check if this model is already running globally
                const isAlreadyRunning = loadedModels.some(m => m.name === selectedBaseModel && m.type === correctEngine);

                // Restart if model not running or server in bad state
                if (!isAlreadyRunning || infServerStatus === 'idle' || infServerStatus === 'error') {
                    isAutoStartingRef.current = true;
                    addLogMessage(`[AUTO] Starting/Restarting server for model: ${selectedBaseModel}`);
                    await handleStartServer();
                    // Reset lock after a delay to allow the server to fully start
                    setTimeout(() => { isAutoStartingRef.current = false; }, 2000);
                } else {
                    addLogMessage(`[DEBUG] Server is already running ${selectedBaseModel}, skipping auto-start.`);
                }
            }
        };
        restart();
    }, [selectedBaseModel, modelOptions, infInferenceEngine]);

    // Update LoRA when checkpoint changes
    useEffect(() => {
        if (selectedProject && selectedCheckpoint) {
            const project = projectLoras.find(p => p.project_name === selectedProject);
            if (project) {
                const checkpoint = project.checkpoints.find(c => c.name === selectedCheckpoint);
                if (checkpoint) {
                    setSelectedLoraAdapter(checkpoint.path);
                }
            }
        }
    }, [selectedCheckpoint, selectedProject, projectLoras]);

    const loadResources = async () => {
        try {
            const resources: any[] = await invoke('list_all_resources_command');

            // Map resources to Select options with enriched data
            const options: Option[] = resources
                .filter(r => r.type === 'gguf' || r.type === 'model')
                .filter(r => !r.is_mmproj) // Don't list mmproj in main list
                .map(r => {
                    const isGGUF = r.type === 'gguf';
                    let name = r.name;
                    let quantTag = '';

                    if (isGGUF) {
                        // Extract quant tag if possible (e.g., Q4_K_M)
                        // Filename usually: "User-Repo-Q4_K_M.gguf" or "model.Q4_K_M.gguf"
                        // Or just simple parsing from the name if the backend provides it cleaned.
                        // Based on typical GGUF naming:
                        const parts = r.path.split(/[-._]/);
                        const possibleQuant = parts.find((p: string) => p.startsWith('Q') && p.length <= 8 && /\d/.test(p));
                        if (possibleQuant) quantTag = possibleQuant;

                        // Clean name: replace -- with /
                        // Remove "data/models/" prefix if present in name (it shouldn't be, usually stripped)
                        name = r.name.replace('data/models/', '').replace(/--/g, '/');
                    } else {
                        // Transformers model: usually "author/repo"
                        name = r.name.replace(/--/g, '/');
                    }

                    // Allow ReactNode in label if component supports it, otherwise string
                    // Checking Select.tsx, usually it renders label. 
                    // Let's assume for now we construct a nice string or use a custom format if Select supports it.
                    // If Select expects string label, we can't emit JSX. 
                    // Let's rely on text format: "Author/Repo [Q4_K_M]"
                    const label = isGGUF ? `${name} ${quantTag ? `[${quantTag}]` : ''}` : name;

                    // Check if this model has a matching mmproj (vision)
                    const hasVision = resources.some(res => res.is_mmproj && res.path.includes(name.split('/')[1] || name));

                    return {
                        value: isGGUF ? r.path : r.name, // base models use name for transformers
                        label: label,
                        engine: isGGUF ? 'GGUF' : 'Base' as any,
                        hasVision: hasVision
                    };
                });

            setModelOptions(options);

            const projects: ProjectLoraInfo[] = await invoke('list_loras_by_project_command');
            setProjectLoras(projects);
        } catch (error) {
            addLogMessage(`Error loading resources: ${error}`);
        }
    };

    const checkServerHealth = async (port: number) => {
        try {
            const isHealthy = await invoke('check_server_health_command', { port });
            return isHealthy as boolean;
        } catch (e) {
            return false;
        }
    };

    const startServerInstance = async (
        slot: 0 | 1,
        model: string,
        _requestedPort: number, // Ignore requested port, enforce slot-based port
        engine: 'llamacpp' | 'transformers',
        loraPath?: string | null
    ) => {
        const isPrimary = slot === 0;
        const setStatus = isPrimary ? setInfServerStatus : setSecondaryBenchStatus;
        const logPrefix = isPrimary ? '[Primary]' : '[Secondary]';

        // Enforce slot-based ports to prevent conflicts
        // Slot 0 -> 8080, Slot 1 -> 8081
        const port = 8080 + slot;

        if (!model) {
            addLogMessage(`${logPrefix} Start called but no model selected.`);
            return;
        }

        addLogMessage(`${logPrefix} Starting server (Engine: ${engine}, Model: ${model}, Port: ${port})`);
        setStatus('loading');

        try {
            if (engine === 'transformers') {
                if (!isPrimary) {
                    addLogMessage(`${logPrefix} Transformers engine not yet supported for secondary slot.`);
                    setStatus('error');
                    return;
                }
                await invoke('start_transformers_server_command', {
                    modelPath: model,
                    port: port
                });
            } else {
                await invoke('start_llama_server_command', {
                    modelPath: model,
                    mmprojPath: '',
                    loraPath: loraPath || '',
                    gpuLayers: infGpuLayers,
                    ctxSize: contextSize,
                    batchSize: infBatchSize,
                    ubatchSize: infUbatchSize,
                    threads: infThreads,
                    flashAttn: infFlashAttn,
                    noMmap: infNoMmap,
                    port: port,
                    slotId: slot
                });
            }

            // Verify health
            let attempts = 0;
            const maxAttempts = 15;
            const checkInterval = setInterval(async () => {
                attempts++;
                const isHealthy = await checkServerHealth(port);
                if (isHealthy) {
                    clearInterval(checkInterval);
                    setStatus('ready');
                    addLogMessage(`${logPrefix} Server ready on port ${port}`);
                    if (isPrimary) setIsServerRunning(true);
                    // Register the loaded model globally with the CORRECT port
                    loadModel({
                        path: model,
                        name: model,
                        type: engine,
                        serverId: `server-${slot}`,
                        serverPort: port,
                        hasMMProj: false
                    });
                } else if (attempts >= maxAttempts) {
                    clearInterval(checkInterval);
                    setStatus('error');
                    addLogMessage(`${logPrefix} Server failed to respond after ${maxAttempts} attempts.`);
                }
            }, 1000);

        } catch (error) {
            addLogMessage(`${logPrefix} Error starting server: ${error}`);
            setStatus('error');
        }
    };

    const handleStartServer = async () => {
        // Start Primary
        await startServerInstance(0, selectedBaseModel, modelConfig?.serverPort || 8080, infInferenceEngine, selectedLoraAdapter);

        // If Benchmarking/Blind Test is active and secondary model is selected, start it too
        if ((isBenchmarking || isBlindTest) && selectedBenchmarkModel) {
            // Force GGUF for secondary for now? Or auto-detect?
            // Simple generic auto-detect:
            const isGGUF = selectedBenchmarkModel.endsWith('.gguf');
            // Only support llama/GGUF for secondary for now as per plan
            if (isGGUF) {
                // Delay slightly to avoid port race or resource spike?
                setTimeout(() => {
                    startServerInstance(1, selectedBenchmarkModel, secondaryBenchPort, 'llamacpp', benchmarkLoraAdapter);
                }, 1000);
            } else {
                addLogMessage('[Secondary] Only GGUF models are currently supported for secondary slot.');
            }
        }
    };

    const handleStopServer = async (clearSelection = false) => {
        try {
            // Stop both slots by ID (not port)
            await invoke('stop_llama_server_command', { slotId: 0 });
            await invoke('stop_llama_server_command', { slotId: 1 });
            setIsServerRunning(false);
            setInfServerStatus('idle');
            if (clearSelection) {
                setSelectedBaseModel(''); // Only clear if explicitly ejecting
                // Also remove from loadedModels in context to reflect 'Eject' immediately
                const currentModel = loadedModels.find(m => m.name === selectedBaseModel);
                if (currentModel) {
                    // We need a way to unload from context. We'll rely on the polling/sync or add unloadModel to destructuring
                }
            }
            setSecondaryBenchStatus('idle');
            setArenaCurrentPair(null);
            setArenaRevealed(false);
        } catch (error) {
            addLogMessage(`Error stopping server: ${error}`);
        }
    };

    const handleStartArena = async () => {
        if (arenaSelectedModels.length < 2) {
            addLogMessage('Select at least 2 models for the Arena pool.');
            return;
        }

        // Randomly pick two distinct models from the pool
        const pool = [...arenaSelectedModels];
        const idx1 = Math.floor(Math.random() * pool.length);
        const model1 = pool.splice(idx1, 1)[0];
        const idx2 = Math.floor(Math.random() * pool.length);
        const model2 = pool.splice(idx2, 1)[0];

        setArenaCurrentPair([model1, model2]);
        setArenaRevealed(false);
        handleClearConversation();

        // Start both servers
        setInfServerStatus('loading');
        setSecondaryBenchStatus('loading');

        try {
            // Start model1 on primary slot (0)
            await invoke('start_llama_server_command', {
                modelPath: model1,
                mmprojPath: '',
                loraPath: null,
                port: modelConfig?.serverPort || 8080,
                gpuLayers: infGpuLayers > 0 ? infGpuLayers : null,
                ctxSize: contextSize,
                batchSize: infBatchSize,
                ubatchSize: infUbatchSize,
                threads: infThreads > 0 ? infThreads : null,
                flashAttn: infFlashAttn,
                noMmap: infNoMmap,
                slotId: 0
            });
            setInfServerStatus('ready');

            // Start model2 on secondary slot (1)
            await invoke('start_llama_server_command', {
                modelPath: model2,
                mmprojPath: '',
                loraPath: null,
                port: secondaryBenchPort,
                gpuLayers: infGpuLayers > 0 ? infGpuLayers : null,
                ctxSize: contextSize,
                batchSize: infBatchSize,
                ubatchSize: infUbatchSize,
                threads: infThreads > 0 ? infThreads : null,
                flashAttn: infFlashAttn,
                noMmap: infNoMmap,
                slotId: 1
            });
            setSecondaryBenchStatus('ready');
            setIsServerRunning(true);
        } catch (error) {
            setInfServerStatus('error');
            setSecondaryBenchStatus('error');
            addLogMessage(`Arena start failed: ${error}`);
        }
    };

    const handleArenaVote = (winner: 'left' | 'right' | 'tie' | 'followup') => {
        if (!arenaCurrentPair || winner === 'followup') {
            if (winner === 'followup') {
                // Just keep current pair and reveal if not already
                setArenaRevealed(true);
            }
            return;
        }

        const [modelA, modelB] = arenaCurrentPair;
        setArenaScores(prev => {
            const next = { ...prev };
            if (!next[modelA]) next[modelA] = { wins: 0, ties: 0, total: 0 };
            if (!next[modelB]) next[modelB] = { wins: 0, ties: 0, total: 0 };

            next[modelA].total += 1;
            next[modelB].total += 1;

            if (winner === 'left') next[modelA].wins += 1;
            if (winner === 'right') next[modelB].wins += 1;
            if (winner === 'tie') {
                next[modelA].ties += 0.5;
                next[modelB].ties += 0.5;
            }
            return next;
        });

        setArenaRevealed(true);
    };

    const handleStopGeneration = async () => {
        try {
            setIsSending(false);
            setIsPromptProcessing(false);
            addLogMessage('Generation stopped by user.');
        } catch (e) {
            console.error(e);
        }
    };

    const handleRestartServer = async () => {
        if (isServerRunning) {
            await handleStopServer();
            // Wait a bit to ensure port release?
            setTimeout(() => handleStartServer(), 1000);
        } else {
            handleStartServer();
        }
    };

    // Build enhanced system prompt with tool/canvas instructions
    const buildEnhancedSystemPrompt = () => {
        let enhanced = systemPrompt;
        const additions: string[] = [];

        // Add tool calling instructions if any tools are enabled
        if (infEnableWebSearch || infEnableCodeExec) {
            additions.push(`
## Available Tools
You have access to the following tools. To use a tool, respond with a tool call in this exact XML format:

<tool_call>
<name>tool_name</name>
<arguments>{"arg1": "value1"}</arguments>
</tool_call>

After making a tool call, STOP and wait for the result before continuing.
`);
            if (infEnableWebSearch) {
                additions.push(`### web_search
Search the web for information.
Arguments: {"query": "search query", "max_results": 3}
Example:
<tool_call>
<name>web_search</name>
<arguments>{"query": "current weather in San Francisco"}</arguments>
</tool_call>`);
            }
            if (infEnableCodeExec) {
                additions.push(`### execute_python_code
Execute Python code (sandboxed, no file/network access).
Arguments: {"code": "python code here"}
Example:
<tool_call>
<name>execute_python_code</name>
<arguments>{"code": "print(2 + 2)"}</arguments>
</tool_call>`);
            }
        }

        // Add canvas instructions if enabled
        if (infEnableCanvas) {
            additions.push(`
## Canvas Mode
When providing code or documents, you can create interactive canvases that the user can edit.

To CREATE or OVERWRITE the canvas content, use:
<canvas_content type="code|markdown|mermaid" language="python">
... content ...
</canvas_content>

To perform a FIND AND REPLACE edit on existing content, use:
<canvas_edit>
<start_line>5</start_line>
<end_line>10</end_line>
<new_content>
... replacement lines ...
</new_content>
</canvas_edit>

For new code blocks that shouldn't open a separate canvas, use standard markdown code fences.
`);
        }

        if (additions.length > 0) {
            enhanced += '\n\n---\n' + additions.join('\n\n');
        }

        return enhanced;
    };

    const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            const file = e.target.files[0];
            const reader = new FileReader();
            reader.onload = (ev) => {
                if (ev.target?.result) {
                    setPendingImage(ev.target.result as string);
                }
            };
            reader.readAsDataURL(file);
        }
    };

    const handleRemoveImage = () => {
        setPendingImage(null);
        if (fileInputRef.current) fileInputRef.current.value = '';
    };

    const handleSendMessage = async (messageToSend?: string) => {
        const msg = messageToSend || inputMessage;
        if ((!msg.trim() && !pendingImage) || !isServerRunning) return;

        if (!messageToSend) {
            setInputMessage('');
            setPendingImage(null); // Clear image after sending
            // Reset textarea height
            if (textareaRef.current) {
                textareaRef.current.style.height = '44px';
            }

            const timestamp = Date.now();
            setChatMessages(prev => [...prev, {
                id: generateId(),
                text: msg,
                sender: 'user',
                timestamp,
                image: pendingImage
            }]);

            if (isBenchmarking) {
                setBenchmarkMessages(prev => [...prev, {
                    id: generateId(),
                    text: msg,
                    sender: 'user',
                    timestamp,
                    image: pendingImage
                }]);
            }
        }

        setIsSending(true);
        setIsPromptProcessing(true);
        setPromptProgress(0);
        // setStreamMetrics(null); // Managed globally now
        setBenchmarkStreamMetrics(null);

        // Construct payload
        let payload: any = msg;
        if (pendingImage) {
            payload = [
                { type: "text", text: msg },
                { type: "image_url", image_url: { url: pendingImage } }
            ];
        }

        const commonParams = {
            host: '127.0.0.1',
            systemPrompt: buildEnhancedSystemPrompt(),
            temperature,
            topP,
            topK,
            ctxSize: contextSize,
        };

        try {
            // Resolving correct port based on selected model
            const targetModel = loadedModels.find(m => m.name === selectedBaseModel);
            const targetPort = targetModel?.serverPort || 8080;
            const isVision = targetModel?.hasMMProj || false;


            // Re-constructing logic:
            const historyToSync = chatMessages.map(m => {
                if (m.sender === 'bot') {
                    return { role: 'assistant', content: m.text };
                }
                let content: any = m.text;
                if (m.image) {
                    if (isVision) {
                        content = [
                            { type: 'text', text: m.text },
                            { type: 'image_url', image_url: { url: m.image } }
                        ];
                    } else {
                        content = m.text + " [Image Omitted]";
                    }
                }
                return { role: 'user', content };
            });

            // Current message payload sanitization
            let cleanPayload: any = msg;
            if (pendingImage) {
                if (isVision) {
                    cleanPayload = [
                        { type: 'text', text: msg },
                        { type: 'image_url', image_url: { url: pendingImage } }
                    ];
                } else {
                    cleanPayload = msg + " [Image Omitted]";
                }
            }

            const primaryRequest = invoke('send_chat_message_streaming_command', {
                ...commonParams,
                port: targetPort,
                message: cleanPayload,
                label: null, // default stream
                full_history: historyToSync
            });

            if (isBenchmarking || isBlindTest) {
                if (!selectedBenchmarkModel) {
                    addLogMessage('[Secondary] No secondary model selected for benchmark.');
                    await primaryRequest;
                } else {
                    const benchmarkRequest = invoke('send_chat_message_streaming_command', {
                        ...commonParams,
                        port: secondaryBenchPort,
                        message: payload,
                        label: 'benchmark'
                    });

                    await Promise.all([primaryRequest, benchmarkRequest]);
                }
            } else {
                await primaryRequest;
            }
            setIsSending(false);
        } catch (error) {
            addLogMessage(`Error sending message: ${error}`);
            setChatMessages(prev => [...prev, { id: generateId(), text: 'Error: Failed to send message.', sender: 'system', timestamp: Date.now() }]);
            setIsSending(false);
        }
    };


    // Generate command preview
    const commandPreview = useMemo(() => {
        if (!selectedBaseModel) return '';
        const parts = ['llama-server'];
        parts.push(`--model "${selectedBaseModel}"`);
        parts.push(`--port ${modelConfig?.serverPort || 8080}`);
        parts.push(`--ctx-size ${contextSize}`);
        parts.push(`--n-gpu-layers ${infGpuLayers > 0 ? infGpuLayers : 999}`);
        parts.push(`--batch-size ${infBatchSize}`);
        parts.push(`--ubatch-size ${infUbatchSize}`);
        if (infFlashAttn) parts.push('--flash-attn on');
        if (infNoMmap) parts.push('--no-mmap');
        if (infThreads > 0) parts.push(`--threads ${infThreads}`);
        if (selectedLoraAdapter) parts.push(`--lora "${selectedLoraAdapter}"`);
        return parts.join(' \\\n  ');
    }, [selectedBaseModel, contextSize, infGpuLayers, infBatchSize, infUbatchSize, infFlashAttn, infNoMmap, infThreads, selectedLoraAdapter, modelConfig]);

    // Status indicator component
    const StatusIndicator = () => {
        const statusConfig = {
            idle: { color: '#71717a', icon: null, text: 'No model loaded' },
            loading: { color: '#f59e0b', icon: <Loader2 size={14} className="animate-spin" />, text: 'Loading model...' },
            ready: { color: '#10b981', icon: <CheckCircle2 size={14} />, text: 'Ready' },
            error: { color: '#ef4444', icon: <XCircle size={14} />, text: 'Error' },
        };
        const config = statusConfig[infServerStatus];
        return (
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', padding: '6px 12px', background: 'rgba(255,255,255,0.05)', borderRadius: '20px', fontSize: '13px' }}>
                <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: config.color, boxShadow: infServerStatus === 'ready' ? `0 0 8px ${config.color}` : 'none' }} />
                {config.icon}
                <span style={{ color: config.color }}>{config.text}</span>
            </div>
        );
    };

    const isCanvasVisible = infEnableCanvas && infCanvasVisible;

    return (
        <div style={{ display: 'flex', height: 'calc(100vh - 100px)', overflow: 'hidden', position: 'relative' }}>
            {/* Background Update Overlay */}
            {isEngineUpdating && (
                <div style={{
                    position: 'absolute',
                    inset: 0,
                    zIndex: 2000,
                    background: 'rgba(9, 9, 11, 0.85)',
                    backdropFilter: 'blur(12px)',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '24px',
                    padding: '40px',
                    textAlign: 'center'
                }}>
                    <div style={{
                        width: '80px',
                        height: '80px',
                        borderRadius: '24px',
                        background: 'rgba(139, 92, 246, 0.1)',
                        border: '1px solid rgba(139, 92, 246, 0.2)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        marginBottom: '8px'
                    }}>
                        <Loader2 size={40} className="animate-spin text-accent-primary" />
                    </div>

                    <div>
                        <h2 style={{ fontSize: '24px', fontWeight: 700, marginBottom: '8px', background: 'var(--accent-gradient)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                            Updating Inference Engine
                        </h2>
                        <p style={{ color: 'var(--text-muted)', fontSize: '14px', maxWidth: '400px' }}>
                            We're preparing the latest version of the LlamaCPP core for peak performance.
                        </p>
                    </div>

                    <div style={{ width: '100%', maxWidth: '400px' }}>
                        <div style={{ height: '6px', width: '100%', background: 'rgba(255, 255, 255, 0.1)', borderRadius: '3px', overflow: 'hidden', marginBottom: '12px' }}>
                            <div style={{
                                height: '100%',
                                width: `${setupProgressPercent}%`,
                                background: 'var(--accent-gradient)',
                                boxShadow: '0 0 15px var(--accent-primary)',
                                transition: 'width 0.3s ease'
                            }} />
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'monospace' }}>
                            <span style={{ textTransform: 'uppercase' }}>{setupMessage || 'UPDATING...'}</span>
                            <span>{setupProgressPercent}%</span>
                        </div>
                    </div>
                </div>
            )}
            <style>{`
                @keyframes wiggle {
                    0%, 100% { transform: translateY(0); }
                    25% { transform: translateY(-3px); }
                    75% { transform: translateY(3px); }
                }
            `}</style>
            {/* Main Chat Area */}
            <div style={{
                flex: 1,
                display: 'flex',
                flexDirection: 'column',
                minWidth: '450px', // Prevent crushing on small screens
                borderRight: isCanvasVisible ? '1px solid rgba(255,255,255,0.08)' : 'none'
            }}>
                {/* Top Model Bar */}
                <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '12px',
                    padding: '12px 16px',
                    background: 'var(--bg-surface, #121216)',
                    borderBottom: '1px solid var(--border-subtle, rgba(255,255,255,0.05))',
                    boxShadow: 'var(--shadow-sm, 0 2px 4px rgba(0,0,0,0.12))',
                    flexShrink: 0
                }}>
                    {/* Evaluation Mode Toggle */}
                    <div style={{ position: 'relative', flexShrink: 0 }}>
                        <button
                            onClick={() => {
                                const modes: ('off' | 'compare' | 'arena')[] = ['off', 'compare', 'arena'];
                                const currentIdx = modes.indexOf(evaluationMode);
                                setEvaluationMode(modes[(currentIdx + 1) % 3]);
                            }}
                            style={{
                                width: '36px', height: '36px',
                                borderRadius: '8px',
                                background: evaluationMode !== 'off' ? 'rgba(167, 139, 250, 0.2)' : 'rgba(255,255,255,0.05)',
                                border: evaluationMode !== 'off' ? '1px solid rgba(167, 139, 250, 0.5)' : '1px solid rgba(255,255,255,0.1)',
                                color: evaluationMode !== 'off' ? '#a78bfa' : '#9ca3af',
                                cursor: 'pointer',
                                display: 'flex', alignItems: 'center', justifyContent: 'center',
                                transition: 'all 0.2s'
                            }}
                            title={`Evaluation Mode: ${evaluationMode.charAt(0).toUpperCase() + evaluationMode.slice(1)}`}
                        >
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <rect x="3" y="3" width="18" height="18" rx="2" stroke="currentColor" strokeOpacity="0.5" />
                                <line x1="12" y1="2" x2="12" y2="22" stroke="currentColor" strokeDasharray="2 2" />
                            </svg>
                        </button>
                        {evaluationMode !== 'off' && (
                            <span style={{
                                position: 'absolute', top: '100%', left: '0', marginTop: '4px',
                                fontSize: '10px', color: '#a78bfa', whiteSpace: 'nowrap',
                                background: 'rgba(0,0,0,0.6)', padding: '2px 6px', borderRadius: '4px',
                                zIndex: 100
                            }}>
                                {evaluationMode === 'arena' ? 'Arena Mode' : 'Compare Mode'}
                            </span>
                        )}
                    </div>

                    {/* Model Selector */}
                    <div style={{ flex: 2, maxWidth: '500px' }}>
                        <Select
                            value={selectedBaseModel}
                            onChange={(val) => {
                                // Detect engine from selected option if possible, or just use GGUF as default
                                const opt = modelOptions.find(o => o.value === val);
                                if (opt) {
                                    setInfInferenceEngine(opt.engine === 'GGUF' ? 'llamacpp' : 'transformers');
                                }
                                if (opt) {
                                    setInfInferenceEngine(opt.engine === 'GGUF' ? 'llamacpp' : 'transformers');
                                }
                                if (isServerRunning) handleStopServer(false); // Don't clear selection, we are switching
                                setSelectedBaseModel(val);
                            }}
                            options={[
                                { value: '', label: 'Select a model...' },
                                ...modelOptions
                            ]}
                            placeholder="Select a model..."
                            style={{ width: '100%' }}
                            showSearch={true}
                        />
                    </div>


                    {/* LoRA Selector */}
                    <div style={{ flex: 1, maxWidth: '250px' }}>
                        <Select
                            value={selectedProject}
                            onChange={(val) => {
                                setSelectedProject(val);
                                if (!val) {
                                    setSelectedCheckpoint('');
                                    setSelectedLoraAdapter('');
                                } else {
                                    const project = projectLoras.find(p => p.project_name === val);
                                    if (project && project.checkpoints.length > 0) {
                                        setSelectedCheckpoint(project.checkpoints[0].name);
                                        setSelectedLoraAdapter(project.checkpoints[0].path);
                                    }
                                }
                            }}
                            options={[
                                { value: '', label: 'No LoRA' },
                                ...projectLoras.map(p => ({
                                    value: p.project_name,
                                    label: `🔗 ${p.project_name}`
                                }))
                            ]}
                            placeholder="Select LoRA Adapter"
                            disabled={infServerStatus === 'loading'}
                        />
                    </div>

                    {/* Status */}
                    <StatusIndicator />

                    {/* Restart Button */}
                    <button
                        onClick={handleRestartServer}
                        style={{
                            padding: '8px',
                            background: 'rgba(255,255,255,0.05)',
                            border: '1px solid rgba(255,255,255,0.1)',
                            borderRadius: '8px',
                            color: '#e4e4e7',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '6px'
                        }}
                        title="Restart Server"
                    >
                        <RotateCcw size={14} />
                    </button>

                    {/* Eject Button */}
                    {isServerRunning && (
                        <button
                            onClick={() => handleStopServer(true)}
                            style={{
                                padding: '8px 12px',
                                background: 'rgba(239,68,68,0.15)',
                                border: '1px solid rgba(239,68,68,0.3)',
                                borderRadius: '8px',
                                color: '#f87171',
                                cursor: 'pointer',
                                display: 'flex',
                                alignItems: 'center',
                                gap: '6px',
                                fontSize: '13px'
                            }}
                        >
                            <Power size={14} /> Eject
                        </button>
                    )}

                    {/* Settings Toggle */}
                    <button
                        onClick={() => setShowSidebar(!showSidebar)}
                        style={{
                            padding: '10px',
                            background: showSidebar ? 'rgba(167,139,250,0.2)' : 'rgba(255,255,255,0.05)',
                            border: '1px solid rgba(255,255,255,0.1)',
                            borderRadius: '10px',
                            color: showSidebar ? '#a78bfa' : '#9ca3af',
                            cursor: 'pointer'
                        }}
                    >
                        <Settings2 size={18} />
                    </button>
                </div>


                {/* Main Content Area - Flex Row */}
                <div style={{ display: 'flex', flex: 1, overflow: 'hidden', position: 'relative' }}>

                    {/* 1. Chat Column */}
                    <div style={{
                        flex: 1,
                        display: 'flex',
                        flexDirection: 'column',
                        background: 'var(--bg-app, #09090b)',
                        minWidth: '300px',
                        overflow: 'hidden',
                        position: 'relative'
                    }}>

                        {/* Messages */}
                        <div style={{ flex: 1, overflowY: 'auto', padding: '24px', paddingBottom: '160px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
                            {chatMessages.length === 0 && (
                                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', opacity: 0.4 }}>
                                    <Brain size={56} style={{ marginBottom: '16px' }} />
                                    <p style={{ fontSize: '16px', color: '#9ca3af' }}>
                                        {infServerStatus === 'idle' ? 'Select a model to start chatting' :
                                            infServerStatus === 'loading' ? 'Loading model...' :
                                                'Start a conversation...'}
                                    </p>
                                </div>
                            )}

                            {/* Compare Mode: Two Columns with Model Dropdowns */}
                            {evaluationMode === 'compare' && (
                                <div style={{ display: 'flex', gap: '24px', height: '100%' }}>
                                    {/* Column A - Primary Model */}
                                    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '12px', borderRight: '1px solid rgba(255,255,255,0.05)', paddingRight: '12px' }}>
                                        <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', padding: '8px', background: 'rgba(59,130,246,0.05)', borderRadius: '8px' }}>
                                            <Select
                                                value={selectedBaseModel}
                                                onChange={(val) => setSelectedBaseModel(val)}
                                                options={modelOptions}
                                                placeholder="Select Model A"
                                            />
                                        </div>
                                        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '20px', overflowY: 'auto' }}>
                                            {chatMessages.map((msg, idx) => (
                                                <MessageItem
                                                    key={msg.id || idx} msg={msg} idx={idx} isSecondary={false}
                                                    editingIndex={editingIndex} editText={editText} setEditText={setEditText}
                                                    handleEditSave={handleEditSave} handleEditStart={handleEditStart}
                                                    handleDeleteMessage={handleDeleteMessage} setEditingIndex={setEditingIndex}
                                                    streamMetrics={streamMetrics} infShowMetrics={infShowMetrics}
                                                    isLast={idx === chatMessages.length - 1}
                                                />
                                            ))}
                                        </div>
                                    </div>
                                    {/* Column B - Secondary Model */}
                                    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '12px' }}>
                                        <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', padding: '8px', background: 'rgba(167,139,250,0.05)', borderRadius: '8px' }}>
                                            <Select
                                                value={selectedBenchmarkModel}
                                                onChange={(val) => setSelectedBenchmarkModel(val)}
                                                options={modelOptions}
                                                placeholder="Select Model B"
                                            />
                                            {/* LoRA Selector for Model B - simplified flat list for now or reuse projectLoras logic if needed */}
                                            {/* For simplicity finding ANY checkpoint from projectLoras that matches user choice */}
                                            <div style={{ marginTop: '4px' }}>
                                                <Select
                                                    value={benchmarkLoraAdapter}
                                                    onChange={(val) => setBenchmarkLoraAdapter(val)}
                                                    options={[
                                                        { value: '', label: 'No LoRA' },
                                                        ...projectLoras.flatMap(p => p.checkpoints.map(c => ({
                                                            value: c.path,
                                                            label: `${p.project_name} / ${c.name}`
                                                        })))
                                                    ]}
                                                    placeholder="Select LoRA B"
                                                />
                                            </div>
                                        </div>
                                        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '20px', overflowY: 'auto' }}>
                                            {benchmarkMessages.map((msg, idx) => (
                                                <MessageItem
                                                    key={msg.id || idx} msg={msg} idx={idx} isSecondary={true}
                                                    editingIndex={editingIndex} editText={editText} setEditText={setEditText}
                                                    handleEditSave={handleEditSave} handleEditStart={handleEditStart}
                                                    handleDeleteMessage={handleDeleteMessage} setEditingIndex={setEditingIndex}
                                                    streamMetrics={benchmarkStreamMetrics} infShowMetrics={infShowMetrics}
                                                    isLast={idx === benchmarkMessages.length - 1}
                                                />
                                            ))}
                                        </div>
                                    </div>
                                </div>
                            )}

                            {/* Arena Mode: Blind Testing with Voting */}
                            {evaluationMode === 'arena' && (
                                <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', height: '100%' }}>
                                    {/* Arena Pool and Start Button */}
                                    <div style={{ padding: '12px', background: 'rgba(167,139,250,0.05)', borderRadius: '8px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                            <span style={{ fontSize: '11px', color: '#9ca3af', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Arena Pool (select 2+ models)</span>
                                            <button
                                                onClick={handleStartArena}
                                                disabled={arenaSelectedModels.length < 2 || isSending}
                                                style={{
                                                    padding: '6px 12px', background: 'var(--accent-primary, #8b5cf6)',
                                                    border: 'none', borderRadius: '6px', color: 'white',
                                                    fontSize: '12px', fontWeight: 'bold', cursor: 'pointer',
                                                    opacity: (arenaSelectedModels.length < 2 || isSending) ? 0.5 : 1,
                                                    display: 'flex', alignItems: 'center', gap: '6px'
                                                }}
                                            >
                                                <Play size={12} fill="currentColor" /> Start Arena Run
                                            </button>
                                        </div>
                                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                                            {modelOptions.map((opt) => (
                                                <button
                                                    key={opt.value}
                                                    onClick={() => {
                                                        setArenaSelectedModels(prev =>
                                                            prev.includes(opt.value)
                                                                ? prev.filter(m => m !== opt.value)
                                                                : [...prev, opt.value]
                                                        );
                                                    }}
                                                    style={{
                                                        padding: '4px 8px', fontSize: '11px', borderRadius: '4px',
                                                        background: arenaSelectedModels.includes(opt.value) ? 'rgba(167,139,250,0.3)' : 'rgba(255,255,255,0.05)',
                                                        border: arenaSelectedModels.includes(opt.value) ? '1px solid #a78bfa' : '1px solid rgba(255,255,255,0.1)',
                                                        color: arenaSelectedModels.includes(opt.value) ? '#a78bfa' : '#9ca3af',
                                                        cursor: 'pointer'
                                                    }}
                                                >
                                                    {opt.label}
                                                </button>
                                            ))}
                                        </div>
                                    </div>

                                    {/* Scoreboard (Collapsible) */}
                                    {Object.keys(arenaScores).length > 0 && (
                                        <div style={{ padding: '10px', background: 'rgba(0,0,0,0.3)', borderRadius: '8px', fontSize: '11px' }}>
                                            <div style={{ fontWeight: 'bold', marginBottom: '8px', color: '#a78bfa' }}>Scoreboard</div>
                                            <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap' }}>
                                                {Object.entries(arenaScores).map(([model, score]) => (
                                                    <div key={model} style={{ textAlign: 'center' }}>
                                                        <div style={{ color: '#fff', fontWeight: '600' }}>{model.split('/').pop()?.split('.')[0]}</div>
                                                        <div style={{ color: '#a78bfa' }}>{score.wins + score.ties * 0.5}</div>
                                                        <div style={{ color: '#6b7280' }}>{((score.wins + score.ties * 0.5) / Math.max(score.total, 1)).toFixed(2)}</div>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    )}

                                    {/* Reveal and Voting UI */}
                                    {(chatMessages.some(m => m.sender === 'bot') || benchmarkMessages.some(m => m.sender === 'bot')) && !isSending && (
                                        <div style={{ padding: '20px', background: 'rgba(167,139,250,0.05)', border: '1px solid rgba(167,139,250,0.1)', borderRadius: '12px', display: 'flex', flexDirection: 'column', gap: '16px', alignItems: 'center' }}>
                                            <span style={{ fontSize: '14px', color: '#a78bfa', fontWeight: 500 }}>Which response was better?</span>
                                            <div style={{ display: 'flex', gap: '12px' }}>
                                                <button
                                                    onClick={() => handleArenaVote('left')}
                                                    style={{ padding: '10px 20px', background: '#3b82f6', border: 'none', borderRadius: '8px', color: 'white', fontWeight: 'bold', cursor: 'pointer' }}
                                                >
                                                    👈 Left Wins
                                                </button>
                                                <button
                                                    onClick={() => handleArenaVote('right')}
                                                    style={{ padding: '10px 20px', background: '#8b5cf6', border: 'none', borderRadius: '8px', color: 'white', fontWeight: 'bold', cursor: 'pointer' }}
                                                >
                                                    Right Wins 👉
                                                </button>
                                                <button
                                                    onClick={() => handleArenaVote('tie')}
                                                    style={{ padding: '10px 20px', background: 'rgba(255,255,255,0.1)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', color: 'white', fontWeight: 'bold', cursor: 'pointer' }}
                                                >
                                                    🤝 It's a Tie
                                                </button>
                                                <button
                                                    onClick={() => handleArenaVote('followup')}
                                                    style={{ padding: '10px 20px', background: 'transparent', border: '1px solid #10b981', borderRadius: '8px', color: '#10b981', fontWeight: 'bold', cursor: 'pointer' }}
                                                >
                                                    💬 Follow-up
                                                </button>
                                            </div>

                                            {arenaRevealed && arenaCurrentPair && (
                                                <div style={{ marginTop: '10px', padding: '10px 20px', background: 'rgba(0,0,0,0.4)', borderRadius: '8px', display: 'flex', gap: '32px' }}>
                                                    <div><span style={{ color: '#60a5fa', fontWeight: 'bold' }}>Model A:</span> <span style={{ color: '#9ca3af' }}>{arenaCurrentPair[0].split('/').pop()}</span></div>
                                                    <div><span style={{ color: '#a78bfa', fontWeight: 'bold' }}>Model B:</span> <span style={{ color: '#9ca3af' }}>{arenaCurrentPair[1].split('/').pop()}</span></div>
                                                </div>
                                            )}
                                        </div>
                                    )}

                                    {/* Arena Dual Columns */}
                                    <div style={{ display: 'flex', gap: '24px', flex: 1 }}>
                                        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '20px', borderRight: '1px solid rgba(255,255,255,0.05)', paddingRight: '12px' }}>
                                            <div style={{ textAlign: 'center', padding: '10px', background: 'rgba(59,130,246,0.1)', borderRadius: '8px' }}>
                                                <span style={{ fontSize: '14px', fontWeight: 'bold', color: '#60a5fa' }}>Model A</span>
                                            </div>
                                            {chatMessages.map((msg, idx) => (
                                                <MessageItem
                                                    key={msg.id || idx} msg={msg} idx={idx} isSecondary={false}
                                                    editingIndex={editingIndex} editText={editText} setEditText={setEditText}
                                                    handleEditSave={handleEditSave} handleEditStart={handleEditStart}
                                                    handleDeleteMessage={handleDeleteMessage} setEditingIndex={setEditingIndex}
                                                    streamMetrics={streamMetrics} infShowMetrics={infShowMetrics}
                                                    isLast={idx === chatMessages.length - 1}
                                                />
                                            ))}
                                        </div>
                                        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '20px' }}>
                                            <div style={{ textAlign: 'center', padding: '10px', background: 'rgba(167,139,250,0.1)', borderRadius: '8px' }}>
                                                <span style={{ fontSize: '14px', fontWeight: 'bold', color: '#a78bfa' }}>Model B</span>
                                            </div>
                                            {benchmarkMessages.map((msg, idx) => (
                                                <MessageItem
                                                    key={msg.id || idx} msg={msg} idx={idx} isSecondary={true}
                                                    editingIndex={editingIndex} editText={editText} setEditText={setEditText}
                                                    handleEditSave={handleEditSave} handleEditStart={handleEditStart}
                                                    handleDeleteMessage={handleDeleteMessage} setEditingIndex={setEditingIndex}
                                                    streamMetrics={benchmarkStreamMetrics} infShowMetrics={infShowMetrics}
                                                    isLast={idx === benchmarkMessages.length - 1}
                                                />
                                            ))}
                                        </div>
                                    </div>
                                </div>
                            )}

                            {/* Standard Chat: Single Column */}
                            {evaluationMode === 'off' && (
                                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '20px' }}>
                                    {chatMessages.map((msg, idx) => (
                                        <MessageItem
                                            key={msg.id || idx} msg={msg} idx={idx} isSecondary={false}
                                            editingIndex={editingIndex} editText={editText} setEditText={setEditText}
                                            handleEditSave={handleEditSave} handleEditStart={handleEditStart}
                                            handleDeleteMessage={handleDeleteMessage} setEditingIndex={setEditingIndex}
                                            streamMetrics={streamMetrics} infShowMetrics={infShowMetrics}
                                            isLast={idx === chatMessages.length - 1}
                                        />
                                    ))}
                                </div>
                            )}
                            <div ref={messagesEndRef} />
                        </div>

                        {/* Input Area (Floating Capsule) */}
                        <div style={{
                            padding: '0 40px 40px 40px',
                            display: 'flex',
                            justifyContent: 'center',
                            pointerEvents: 'none',
                            position: 'absolute', // Fixed at bottom
                            bottom: 0,
                            left: 0,
                            right: 0,
                            zIndex: 20
                        }}>
                            <div style={{
                                width: '100%',
                                maxWidth: '850px',
                                background: 'var(--bg-elevated)', // Dark grey capsule
                                borderRadius: '24px',
                                border: '1px solid var(--border-subtle)',
                                boxShadow: '0 8px 32px rgba(0,0,0,0.3)',
                                display: 'flex',
                                flexDirection: 'column',
                                position: 'relative',
                                pointerEvents: 'auto',
                                padding: '8px'
                            }}>
                                {/* Processing / Status Overlay if needed, or just wiggle inside */}
                                {isPromptProcessing && (
                                    <div style={{ position: 'absolute', top: '-30px', left: '50%', transform: 'translateX(-50%)', background: 'rgba(0,0,0,0.6)', padding: '4px 12px', borderRadius: '12px', fontSize: '12px', color: '#a78bfa', backdropFilter: 'blur(4px)' }}>
                                        Processing {promptProgress ? `${promptProgress}%` : '...'}
                                    </div>
                                )}

                                {/* Top: Text Area */}
                                <textarea
                                    ref={textareaRef}
                                    value={inputMessage}
                                    onChange={(e) => {
                                        setInputMessage(e.target.value);
                                        e.target.style.height = 'auto';
                                        e.target.style.height = Math.min(e.target.scrollHeight, 150) + 'px';
                                    }}
                                    onKeyDown={(e) => {
                                        if (e.key === 'Enter' && !e.shiftKey) {
                                            e.preventDefault();
                                            handleSendMessage();
                                        }
                                    }}
                                    placeholder={infServerStatus === 'ready' ? "Ask anything..." : "Load a model to chat"}
                                    rows={1}
                                    disabled={infServerStatus !== 'ready' || isSending}
                                    style={{
                                        width: '100%',
                                        background: 'transparent',
                                        border: 'none',
                                        color: 'var(--text-main)',
                                        fontSize: '16px',
                                        padding: '12px 16px',
                                        outline: 'none',
                                        resize: 'none',
                                        minHeight: '44px',
                                        maxHeight: '150px',
                                        lineHeight: '1.5',
                                        borderRadius: '16px'
                                    }}
                                />

                                {/* Bottom: Tools & Send */}
                                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 8px' }}>

                                    {/* Left: Attachments & Tools */}
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                        <div style={{ position: 'relative' }}>
                                            <button
                                                onClick={() => setShowToolsDropup(!showToolsDropup)}
                                                style={{
                                                    background: showToolsDropup ? 'rgba(167,139,250,0.2)' : 'transparent',
                                                    border: 'none', color: showToolsDropup ? '#a78bfa' : '#9ca3af',
                                                    cursor: 'pointer', padding: '8px', borderRadius: '50%', transition: 'all 0.2s'
                                                }}
                                                className="icon-btn"
                                            >
                                                <PlusCircle size={20} />
                                            </button>

                                            {showToolsDropup && (
                                                <div style={{
                                                    position: 'absolute', bottom: '100%', left: '0', marginBottom: '12px',
                                                    background: 'rgba(18, 18, 22, 0.95)', border: '1px solid rgba(255,255,255,0.1)',
                                                    borderRadius: '12px', padding: '8px', display: 'flex', flexDirection: 'column', gap: '4px',
                                                    boxShadow: '0 10px 25px -5px rgba(0,0,0,0.5)', backdropFilter: 'blur(12px)',
                                                    minWidth: '220px', zIndex: 1000
                                                }}>
                                                    <div style={{ padding: '8px 12px', fontSize: '10px', color: '#6b7280', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Capabilities</div>

                                                    <button
                                                        onClick={() => { setInfEnableWebSearch(!infEnableWebSearch); setShowToolsDropup(false); }}
                                                        style={{ display: 'flex', alignItems: 'center', gap: '10px', padding: '8px 12px', background: infEnableWebSearch ? 'rgba(59,130,246,0.1)' : 'transparent', border: 'none', borderRadius: '6px', color: infEnableWebSearch ? '#60a5fa' : '#9ca3af', cursor: 'pointer', textAlign: 'left' }}
                                                    >
                                                        <Search size={16} /> <span style={{ fontSize: '13px' }}>Web Search</span>
                                                    </button>

                                                    <button
                                                        onClick={() => { setInfEnableCodeExec(!infEnableCodeExec); setShowToolsDropup(false); }}
                                                        style={{ display: 'flex', alignItems: 'center', gap: '10px', padding: '8px 12px', background: infEnableCodeExec ? 'rgba(34,197,94,0.1)' : 'transparent', border: 'none', borderRadius: '6px', color: infEnableCodeExec ? '#4ade80' : '#9ca3af', cursor: 'pointer', textAlign: 'left' }}
                                                    >
                                                        <Code size={16} /> <span style={{ fontSize: '13px' }}>Code Execution</span>
                                                    </button>

                                                    <button
                                                        onClick={() => { setInfEnableCanvas(!infEnableCanvas); setShowToolsDropup(false); }}
                                                        style={{ display: 'flex', alignItems: 'center', gap: '10px', padding: '8px 12px', background: infEnableCanvas ? 'rgba(167,139,250,0.1)' : 'transparent', border: 'none', borderRadius: '6px', color: infEnableCanvas ? '#a78bfa' : '#9ca3af', cursor: 'pointer', textAlign: 'left' }}
                                                    >
                                                        <PenTool size={16} /> <span style={{ fontSize: '13px' }}>Artifacts Canvas</span>
                                                    </button>

                                                    <div style={{ height: '1px', background: 'rgba(255,255,255,0.05)', margin: '4px 0' }} />

                                                    <button
                                                        onClick={() => { fileInputRef.current?.click(); setShowToolsDropup(false); }}
                                                        style={{ display: 'flex', alignItems: 'center', gap: '10px', padding: '8px 12px', background: 'transparent', border: 'none', borderRadius: '6px', color: '#9ca3af', cursor: 'pointer', textAlign: 'left' }}
                                                    >
                                                        <Paperclip size={16} /> <span style={{ fontSize: '13px' }}>Attach Vision</span>
                                                    </button>
                                                </div>
                                            )}
                                        </div>
                                        <input type="file" ref={fileInputRef} style={{ display: 'none' }} accept="image/*" onChange={handleFileSelect} />

                                        {/* Clear Chat */}
                                        {chatMessages.length > 0 && (
                                            <button
                                                onClick={handleClearConversation}
                                                style={{ background: 'transparent', border: 'none', color: '#9ca3af', cursor: 'pointer', padding: '8px', borderRadius: '50%', transition: 'background 0.2s' }}
                                                className="icon-btn"
                                            >
                                                <RotateCcw size={20} />
                                            </button>
                                        )}

                                        {/* Image Preview */}
                                        {pendingImage && (
                                            <div style={{ position: 'relative', width: '32px', height: '32px', borderRadius: '4px', overflow: 'hidden', marginLeft: '4px' }}>
                                                <img src={pendingImage} alt="Preview" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                                                <div
                                                    onClick={handleRemoveImage}
                                                    style={{ position: 'absolute', inset: 0, background: 'rgba(0,0,0,0.5)', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', opacity: 0 }}
                                                    className="img-hover"
                                                >
                                                    <X size={12} color="#fff" />
                                                </div>
                                            </div>
                                        )}
                                    </div>

                                    {/* Right: Send / Stop */}
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                        <div style={{ fontSize: '12px', color: '#666', marginRight: '8px' }}>
                                            {inputMessage.length} / {contextSize}
                                        </div>
                                        <button
                                            onClick={isSending ? handleStopGeneration : () => handleSendMessage()}
                                            disabled={!isSending && !inputMessage.trim() && !pendingImage}
                                            style={{
                                                width: '36px', height: '36px',
                                                borderRadius: '50%',
                                                background: isSending ? 'rgba(239, 68, 68, 0.2)' : (inputMessage.trim() || pendingImage ? '#fff' : '#4b5563'),
                                                border: isSending ? '1px solid rgba(239, 68, 68, 0.5)' : 'none',
                                                color: isSending ? '#ef4444' : (inputMessage.trim() || pendingImage ? '#000' : '#9ca3af'),
                                                cursor: (inputMessage.trim() || pendingImage || isSending) ? 'pointer' : 'default',
                                                display: 'flex', alignItems: 'center', justifyContent: 'center',
                                                transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)'
                                            }}
                                        >
                                            {isSending ? (
                                                <div style={{ width: '12px', height: '12px', borderRadius: '2px', background: '#ef4444' }} />
                                            ) : <ArrowUp size={20} strokeWidth={3} />}
                                        </button>
                                    </div>

                                </div>
                            </div>
                        </div>
                        <style>{`
                            .icon-btn:hover { background: rgba(255,255,255,0.1) !important; color: white !important; }
                        `}</style>
                    </div>

                    {/* 2. Canvas Column (Middle) */}
                    {infEnableCanvas && (
                        <>
                            {/* Resizer Handle */}
                            <div
                                onMouseDown={() => setIsResizingCanvas(true)}
                                style={{ width: '4px', background: 'rgba(255,255,255,0.05)', cursor: 'col-resize', transition: 'background 0.2s', zIndex: 10 }}
                                onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(167, 139, 250, 0.5)'}
                                onMouseLeave={(e) => e.currentTarget.style.background = 'rgba(255,255,255,0.05)'}
                            />
                            <div style={{ width: canvasWidth, minWidth: '300px', borderLeft: '1px solid var(--border-subtle)', background: 'var(--bg-surface)', display: 'flex', flexDirection: 'column', flexShrink: 0 }}>
                                <CanvasPanel
                                    artifacts={infCanvasArtifacts}
                                    onArtifactsChange={setInfCanvasArtifacts}
                                    isVisible={true}
                                    onToggleVisibility={() => setInfCanvasVisible(false)}
                                    // Remove Analyze button logic as requested
                                    onAnalyzeWithAI={undefined}
                                />
                            </div>
                        </>
                    )}

                    {/* 3. Settings Column (Right) */}
                    {showSidebar && (
                        <>
                            {/* Resizer Handle */}
                            <div
                                onMouseDown={() => setIsResizingSettings(true)}
                                style={{ width: '4px', background: 'rgba(255,255,255,0.05)', cursor: 'col-resize', transition: 'background 0.2s', zIndex: 10 }}
                                onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(167, 139, 250, 0.5)'}
                                onMouseLeave={(e) => e.currentTarget.style.background = 'rgba(255,255,255,0.05)'}
                            />
                            <div style={{
                                width: settingsWidth,
                                minWidth: '250px',
                                borderLeft: '1px solid var(--border-subtle)',
                                background: 'var(--bg-surface)',
                                display: 'flex',
                                flexDirection: 'column',
                                overflowY: 'auto',
                                flexShrink: 0,
                                boxShadow: 'inset 2px 0 8px rgba(0,0,0,0.15)'
                            }}>
                                {/* Parameters Section */}
                                <div style={{ padding: '20px', borderBottom: '1px solid var(--border-subtle)' }}>
                                    <h3 style={{ fontSize: '12px', fontWeight: 600, color: '#6b7280', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '16px' }}>Parameters</h3>

                                    <div style={{ marginBottom: '16px' }}>
                                        <Input
                                            label="Temperature"
                                            type="range"
                                            min="0"
                                            max="2"
                                            step="0.1"
                                            value={temperature}
                                            onChange={(e) => setTemperature(Number(e.target.value))}
                                            tooltip="Controls randomness. Lower values are more focused, higher values are more creative."
                                            displayValue={temperature.toString()}
                                        />
                                    </div>

                                    <div style={{ marginBottom: '16px' }}>
                                        <label style={{ fontSize: '13px', display: 'block', marginBottom: '6px' }}>System Prompt</label>
                                        <textarea
                                            value={systemPrompt}
                                            onChange={(e) => setSystemPrompt(e.target.value)}
                                            style={{ width: '100%', height: '80px', padding: '10px', background: 'var(--bg-input)', border: '1px solid var(--border-default)', borderRadius: '8px', color: 'var(--text-main)', fontSize: '13px', resize: 'vertical' }}
                                            placeholder="You are a helpful assistant..."
                                        />
                                    </div>

                                    {userMode === 'power' && (
                                        <>
                                            <div style={{ marginBottom: '16px' }}>
                                                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px', marginBottom: '6px' }}>
                                                    <span>Top P</span>
                                                    <span style={{ color: '#a78bfa' }}>{topP}</span>
                                                </div>
                                                <input type="range" min="0" max="1" step="0.05" value={topP} onChange={(e) => setTopP(Number(e.target.value))} style={{ width: '100%', accentColor: '#a78bfa' }} />
                                            </div>

                                            <div style={{ marginBottom: '16px' }}>
                                                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px', marginBottom: '6px' }}>
                                                    <span>Top K</span>
                                                    <span style={{ color: '#a78bfa' }}>{topK}</span>
                                                </div>
                                                <input type="range" min="0" max="100" step="1" value={topK} onChange={(e) => setTopK(Number(e.target.value))} style={{ width: '100%', accentColor: '#a78bfa' }} />
                                            </div>

                                            <Input
                                                label="Context Size"
                                                type="number"
                                                value={contextSize}
                                                onChange={(e) => setContextSize(Number(e.target.value))}
                                            />
                                        </>
                                    )}
                                </div>

                                {/* Advanced Section (Power User) */}
                                {userMode === 'power' && (
                                    <div style={{ padding: '20px', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
                                        <button
                                            onClick={() => setShowAdvanced(!showAdvanced)}
                                            style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%', background: 'none', border: 'none', color: '#9ca3af', cursor: 'pointer', padding: 0, marginBottom: showAdvanced ? '16px' : 0 }}
                                        >
                                            <span style={{ fontSize: '12px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', display: 'flex', alignItems: 'center', gap: '6px' }}><Settings size={14} /> Advanced Options</span>
                                            <ChevronDown size={16} style={{ transform: showAdvanced ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }} />
                                        </button>

                                        {showAdvanced && (
                                            <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-2)' }}>
                                                <div
                                                    className={`selection-card ${infFlashAttn ? 'selected' : ''}`}
                                                    onClick={() => setInfFlashAttn(!infFlashAttn)}
                                                >
                                                    <div className="selection-indicator">
                                                        <Zap size={14} />
                                                    </div>
                                                    <span style={{ fontSize: '13px' }}>Flash Attention</span>
                                                </div>

                                                <div
                                                    className={`selection-card ${infNoMmap ? 'selected' : ''}`}
                                                    onClick={() => setInfNoMmap(!infNoMmap)}
                                                >
                                                    <div className="selection-indicator">
                                                        <HardDrive size={14} />
                                                    </div>
                                                    <span style={{ fontSize: '13px' }}>No Memory Map</span>
                                                </div>

                                                <div>
                                                    <label style={{ fontSize: '13px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '6px' }}>
                                                        <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}><Layers size={14} /> GPU Layers</span>
                                                        {/*
                                                            // Send result back to LLM
                                                            const nextMessage = `Tool result for ${toolName}:\n${resultMsg}\n\nPlease parse this result and answer the user's original request.`;
                                                            // Note: We need to trigger this securely. For now, it might be safer to let the user click "continue" or
                                                            // implement a robust queue. The global listener in AppContext handles tool recursion now.
                                                            // But since we removed the listener here, this block is actually dead code (removed in previous step).
                                                            // Wait, the previous step removed the listener, but did we also remove the 'handleSendMessage' that sets isSending?
                                                            // Yes, we need to map isSendingGlobal to the specific UI elements.
                                                        */}
                                                        <span style={{ color: '#a78bfa' }}>{infGpuLayers >= maxGpuLayers ? 'Max' : infGpuLayers} / {maxGpuLayers}</span>
                                                    </label>
                                                    <input
                                                        type="range"
                                                        min="0"
                                                        max={maxGpuLayers} // Dynamic max
                                                        step="1"
                                                        value={infGpuLayers}
                                                        onChange={(e) => {
                                                            const val = parseInt(e.target.value);
                                                            setInfGpuLayers(val);
                                                        }}
                                                        style={{ width: '100%', accentColor: '#a78bfa', cursor: 'pointer' }}
                                                    />
                                                </div>

                                                <div>
                                                    <label style={{ fontSize: '13px', display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '6px' }}><Cpu size={14} /> Batch Size</label>
                                                    <input type="number" value={infBatchSize} onChange={(e) => setInfBatchSize(Number(e.target.value))} style={{ width: '100%', padding: '8px 12px', background: 'var(--bg-input)', border: '1px solid var(--border-default)', borderRadius: '8px', color: 'var(--text-main)', fontSize: '13px' }} />
                                                </div>

                                                <div>
                                                    <label style={{ fontSize: '13px', marginBottom: '6px', display: 'block' }}>Micro-Batch Size</label>
                                                    <input type="number" value={infUbatchSize} onChange={(e) => setInfUbatchSize(Number(e.target.value))} style={{ width: '100%', padding: '8px 12px', background: 'var(--bg-input)', border: '1px solid var(--border-default)', borderRadius: '8px', color: 'var(--text-main)', fontSize: '13px' }} />
                                                </div>

                                                <div>
                                                    <label style={{ fontSize: '13px', marginBottom: '6px', display: 'block' }}>Threads (0 = auto)</label>
                                                    <input type="number" value={infThreads} onChange={(e) => setInfThreads(Number(e.target.value))} style={{ width: '100%', padding: '8px 12px', background: 'var(--bg-input)', border: '1px solid var(--border-default)', borderRadius: '8px', color: 'var(--text-main)', fontSize: '13px' }} />
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                )}

                                {/* Tools Section */}
                                <div style={{ padding: '20px', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
                                    <button
                                        onClick={() => setShowTools(!showTools)}
                                        style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%', background: 'none', border: 'none', color: '#9ca3af', cursor: 'pointer', padding: 0, marginBottom: showTools ? '16px' : 0 }}
                                    >
                                        <span style={{ fontSize: '12px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', display: 'flex', alignItems: 'center', gap: '6px' }}><Wrench size={14} /> Tools & Display</span>
                                        <ChevronDown size={16} style={{ transform: showTools ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }} />
                                    </button>

                                    {showTools && (
                                        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-2)' }}>
                                            <p style={{ fontSize: '11px', color: '#6b7280', marginBottom: '4px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Comparison & Testing</p>

                                            <div
                                                className={`selection-card ${evaluationMode === 'compare' ? 'selected' : ''}`}
                                                onClick={() => setEvaluationMode(evaluationMode === 'compare' ? 'off' : 'compare')}
                                            >
                                                <div className="selection-indicator">
                                                    <BarChart2 size={14} />
                                                </div>
                                                <span style={{ fontSize: '13px' }}>Benchmarking Mode</span>
                                            </div>

                                            <div
                                                className={`selection-card ${evaluationMode === 'arena' ? 'selected' : ''} ${evaluationMode === 'off' ? 'disabled' : ''}`}
                                                onClick={() => evaluationMode !== 'off' && setEvaluationMode(evaluationMode === 'arena' ? 'compare' : 'arena')}
                                                style={{ opacity: evaluationMode !== 'off' ? 1 : 0.5, cursor: evaluationMode !== 'off' ? 'pointer' : 'not-allowed' }}
                                            >
                                                <div className="selection-indicator">
                                                    <Search size={14} />
                                                </div>
                                                <span style={{ fontSize: '13px' }}>Blind Testing</span>
                                            </div>

                                            {isBenchmarking && (
                                                <div style={{ padding: '4px 8px 8px' }}>
                                                    <Select
                                                        label="Benchmark Against"
                                                        options={[
                                                            { value: '', label: 'None' },
                                                            ...modelOptions.map(opt => ({ value: opt.value, label: opt.label }))
                                                        ]}
                                                        value={selectedBenchmarkModel}
                                                        onChange={(val) => {
                                                            setSelectedBenchmarkModel(val);
                                                            // Auto-start if benchmarking is active?
                                                            // Maybe logic elsewhere handles it, or user must restart server. 
                                                            // For now, let's keep it manual start via huge "START" button or restart.
                                                        }}
                                                    />
                                                    <div style={{ marginTop: '8px', fontSize: '11px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                                                        <span style={{
                                                            width: '8px', height: '8px', borderRadius: '50%',
                                                            background: secondaryBenchStatus === 'ready' ? '#4ade80' : secondaryBenchStatus === 'loading' ? '#fbbf24' : '#ef4444'
                                                        }} />
                                                        <span style={{ color: '#9ca3af' }}>
                                                            {secondaryBenchStatus === 'idle' ? 'Secondary Idle' :
                                                                secondaryBenchStatus === 'loading' ? 'Starting...' :
                                                                    secondaryBenchStatus === 'ready' ? 'Ready (Port 8081)' : 'Error'}
                                                        </span>
                                                    </div>
                                                </div>
                                            )}

                                            <div style={{ borderTop: '1px solid var(--border-subtle)', margin: 'var(--space-2) 0' }} />

                                            <div
                                                className={`selection-card ${infShowMetrics ? 'selected' : ''}`}
                                                onClick={() => setInfShowMetrics(!infShowMetrics)}
                                            >
                                                <div className="selection-indicator" style={{ color: 'var(--accent-green)' }}>
                                                    <Activity size={14} />
                                                </div>
                                                <span style={{ fontSize: '13px' }}>Show Performance Metrics</span>
                                            </div>

                                            <div style={{ borderTop: '1px solid var(--border-subtle)', margin: 'var(--space-2) 0' }} />
                                            <p style={{ fontSize: '11px', color: '#6b7280', marginBottom: '4px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Agentic Tools</p>

                                            <div
                                                className={`selection-card ${infEnableWebSearch ? 'selected' : ''}`}
                                                onClick={() => setInfEnableWebSearch(!infEnableWebSearch)}
                                            >
                                                <div className="selection-indicator" style={{ color: 'var(--accent-blue)' }}>
                                                    <Search size={14} />
                                                </div>
                                                <span style={{ fontSize: '13px' }}>Enhanced Web Search</span>
                                            </div>

                                            <div
                                                className={`selection-card ${infEnableCodeExec ? 'selected' : ''}`}
                                                onClick={() => setInfEnableCodeExec(!infEnableCodeExec)}
                                            >
                                                <div className="selection-indicator" style={{ color: 'var(--accent-yellow)' }}>
                                                    <Code size={14} />
                                                </div>
                                                <span style={{ fontSize: '13px' }}>Local Code Execution</span>
                                            </div>

                                            <div
                                                className={`selection-card ${infEnableCanvas ? 'selected' : ''}`}
                                                onClick={() => setInfEnableCanvas(!infEnableCanvas)}
                                            >
                                                <div className="selection-indicator" style={{ color: 'var(--accent-primary)' }}>
                                                    <PenTool size={14} />
                                                </div>
                                                <span style={{ fontSize: '13px' }}>DeepMind Canvas Mode</span>
                                            </div>

                                            <button
                                                onClick={() => setShowToolCreator(true)}
                                                className="btn btn-secondary"
                                                style={{ marginTop: 'var(--space-2)', width: '100%' }}
                                            >
                                                <Settings2 size={12} /> Create Custom Tool
                                            </button>
                                        </div>
                                    )}
                                </div>

                                {/* Command Preview */}
                                {userMode === 'power' && selectedBaseModel && (
                                    <div style={{ padding: '20px' }}>
                                        <button
                                            onClick={() => setShowCommandPreview(!showCommandPreview)}
                                            style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%', background: 'none', border: 'none', color: '#9ca3af', cursor: 'pointer', padding: 0, marginBottom: showCommandPreview ? '12px' : 0 }}
                                        >
                                            <span style={{ fontSize: '12px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', display: 'flex', alignItems: 'center', gap: '6px' }}><Terminal size={14} /> Command Preview</span>
                                            <ChevronDown size={16} style={{ transform: showCommandPreview ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }} />
                                        </button>
                                        <button
                                            onClick={() => setEvaluationMode(evaluationMode === 'off' ? 'compare' : 'off')}
                                            style={{
                                                padding: '8px 14px',
                                                background: evaluationMode !== 'off' ? 'rgba(167, 139, 250, 0.1)' : 'rgba(255,255,255,0.05)',
                                                border: evaluationMode !== 'off' ? '1px solid rgba(167, 139, 250, 0.3)' : '1px solid rgba(255,255,255,0.1)',
                                                borderRadius: '8px',
                                                color: evaluationMode !== 'off' ? '#a78bfa' : '#9ca3af',
                                                fontSize: '12px',
                                                fontWeight: 600,
                                                cursor: 'pointer',
                                                display: 'flex',
                                                alignItems: 'center',
                                                gap: '8px',
                                                transition: 'all 0.2s'
                                            }}
                                        >
                                            <BarChart2 size={14} />
                                            {evaluationMode === 'off' ? 'Evaluation: Off' : evaluationMode === 'compare' ? 'Evaluation: Compare' : 'Evaluation: Arena'}
                                        </button>

                                        {showCommandPreview && (
                                            <pre style={{
                                                padding: '12px',
                                                background: 'rgba(0,0,0,0.4)',
                                                borderRadius: '8px',
                                                fontSize: '11px',
                                                color: '#a3e635',
                                                overflowX: 'auto',
                                                whiteSpace: 'pre-wrap',
                                                wordBreak: 'break-all',
                                                fontFamily: 'monospace',
                                                margin: 0
                                            }}>
                                                {commandPreview}
                                            </pre>
                                        )}
                                    </div>
                                )}
                            </div>
                        </>
                    )}
                </div>

                <ToolCreatorModal
                    isOpen={showToolCreator}
                    onClose={() => setShowToolCreator(false)}
                    onSaveSuccess={() => addLogMessage('Custom tool created successfully')}
                />
            </div>
        </div >
    );
};

export default InferencePage;
