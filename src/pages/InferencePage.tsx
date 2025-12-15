import React, { useState, useEffect, useRef, useMemo } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { useApp } from '../context/AppContext';
import { Button } from '../components/Button';
import { Input } from '../components/Input';
import { MarkdownRenderer } from '../components/MarkdownRenderer';
import { Canvas } from '../components/Canvas';
import ToolCreatorModal from '../components/ToolCreatorModal';
import { Select } from '../components/Select';
import {
    Send, Settings2, Bot, User, Trash2, Edit2, RotateCcw,
    ChevronDown, Cpu, Zap, HardDrive, Layers, Terminal,
    Loader2, CheckCircle2, XCircle, Power, Paperclip, X,
    Settings, Wrench, BarChart2, Clock, PenTool, Search, Code, Brain, Activity
} from 'lucide-react';

// ThinkingBlock component for COT models
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

// Parse message content for <think> tags
const parseMessageContent = (text: string): { thinking: string | null; response: string } => {
    const thinkMatch = text.match(/<think>([\s\S]*?)<\/think>/i);
    if (thinkMatch) {
        const thinking = thinkMatch[1].trim();
        const response = text.replace(/<think>[\s\S]*?<\/think>/gi, '').trim();
        return { thinking, response };
    }
    return { thinking: null, response: text };
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

const InferencePage: React.FC<InferencePageProps> = ({ modelConfig, addLogMessage }) => {
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
    } = useApp();

    const [showTools, setShowTools] = useState(false);

    const messagesEndRef = useRef<HTMLDivElement>(null);

    // Resources
    const [availableGGUFModels, setAvailableGGUFModels] = useState<string[]>([]);
    const [projectLoras, setProjectLoras] = useState<ProjectLoraInfo[]>([]);
    const [selectedProject, setSelectedProject] = useState('');
    const [selectedCheckpoint, setSelectedCheckpoint] = useState('');

    // UI State
    const [isSending, setIsSending] = useState(false);
    const [showSidebar, setShowSidebar] = useState(true);
    const [showAdvanced, setShowAdvanced] = useState(false);
    const [showCommandPreview, setShowCommandPreview] = useState(false);
    const [editingIndex, setEditingIndex] = useState<number | null>(null);
    const [editText, setEditText] = useState('');
    const [streamMetrics, setStreamMetrics] = useState<{
        tokens_per_second: number;
        prompt_eval_time_ms: number;
        eval_time_ms: number;
        total_tokens: number;
    } | null>(null);
    const [canvasContent, setCanvasContent] = useState('');
    const [canvasMode, setCanvasMode] = useState<'code' | 'markdown' | 'mermaid'>('code');
    const [canvasStreamingEdit, setCanvasStreamingEdit] = useState<{
        startLine: number;
        endLine: number;
        newContent: string;
        isActive: boolean;
        buffer: string; // accumulated content for current tag
        tagName: string | null;
    }>({ startLine: 0, endLine: 0, newContent: '', isActive: false, buffer: '', tagName: null });

    // Vision
    const [pendingImage, setPendingImage] = useState<string | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    // Backend Logs / Progress
    const [promptProgress, setPromptProgress] = useState<number | null>(null); // 0-100 or null
    const [isPromptProcessing, setIsPromptProcessing] = useState(false);

    const [showToolCreator, setShowToolCreator] = useState(false);

    // Check and download inference engine if missing
    useEffect(() => {
        const checkBinary = async () => {
            try {
                const exists: boolean = await invoke('check_llama_binary_command');
                if (!exists) {
                    addLogMessage('Inference engine not found. Downloading...');
                    await invoke('download_llama_binary_command');
                    addLogMessage('Inference engine installed successfully.');
                }
            } catch (error) {
                addLogMessage(`Error checking/downloading inference engine: ${error}`);
            }
        };
        checkBinary();
        loadResources();
    }, []);

    // Max GPU Layers (Dynamic)
    const [maxGpuLayers, setMaxGpuLayers] = useState(200);

    // Listen to logs for dynamic GPU layer detection
    useEffect(() => {
        const unlisten = listen('log', (event: any) => {
            const logMsg = event.payload as string;
            // Example: "offloading 28 repeating layers to GPU" or "offloaded 29/29 layers to GPU"
            // We want to find the total layers if possible.
            // "llama_model_load: n_layer = 32" is standard llama.cpp log but might be hidden in stderr.
            // "offloading X repeating layers" implies X is the count being offloaded, but total might be X+1 (output).

            const match = /offloaded (\d+)\/(\d+) layers/.exec(logMsg);
            if (match) {
                const total = parseInt(match[2]);
                setMaxGpuLayers(total);
                if (infGpuLayers > total) setInfGpuLayers(total); // Clamp
            }
        });
        return () => {
            unlisten.then(f => f());
        };
    }, [infGpuLayers]);

    // Stream listeners
    useEffect(() => {
        let unlistenChunk: () => void;
        let unlistenDone: () => void;

        const setupListeners = async () => {
            unlistenChunk = await listen('chat-stream-chunk', (event: any) => {
                const chunk = event.payload as { content: string };

                // Received first token => prompt processing done
                setIsPromptProcessing(false);
                setPromptProgress(null);

                setChatMessages(prev => {
                    const lastMsg = prev[prev.length - 1];
                    let fullText = chunk.content;

                    if (lastMsg && lastMsg.sender === 'bot' && lastMsg.isStreaming) {
                        fullText = lastMsg.text + chunk.content;

                        // --- Canvas Parsing Logic ---
                        if (infEnableCanvas) {
                            // Check for content creation
                            const contentMatch = /<canvas_content.*?type="(.*?)".*?>(.*)/s.exec(fullText);
                            if (contentMatch) {
                                // If we have a closing tag, use content up to it, otherwise use all content
                                const closingMatch = /(.*?)<\/canvas_content>/s.exec(contentMatch[2]);
                                const content = closingMatch ? closingMatch[1] : contentMatch[2];
                                setCanvasContent(content);
                                const type = contentMatch[1] as any;
                                if (['code', 'markdown', 'mermaid'].includes(type)) {
                                    setCanvasMode(type);
                                }
                            }

                            // Check for edits
                            const editMatch = /<canvas_edit>(.*)/s.exec(fullText);
                            if (editMatch) {
                                const editBody = editMatch[1];
                                const hasClose = /<\/canvas_edit>/.test(editBody);

                                const startLineM = /<start_line>(\d+)<\/start_line>/.exec(editBody);
                                const endLineM = /<end_line>(\d+)<\/end_line>/.exec(editBody);
                                const newContentM = /<new_content>(.*)/s.exec(editBody);

                                if (startLineM && endLineM && newContentM) {
                                    // If we have closing new_content tag
                                    const contentClose = /(.*?)<\/new_content>/s.exec(newContentM[1]);
                                    const rawNewContent = contentClose ? contentClose[1] : newContentM[1];

                                    setCanvasStreamingEdit({
                                        isActive: true, // Always valid if tags exist
                                        startLine: parseInt(startLineM[1]),
                                        endLine: parseInt(endLineM[1]),
                                        newContent: rawNewContent.trim(), // Trim strictly? Maybe not for code indentation
                                        buffer: '',
                                        tagName: null
                                    });
                                }

                                if (hasClose) {
                                    // Reset active edit state slightly after completion or keep it? 
                                    // Keeping it handles the 'final' applying.
                                }
                            }
                        }
                        // ----------------------------

                        return [...prev.slice(0, -1), { ...lastMsg, text: fullText }];
                    } else {
                        return [...prev, { text: chunk.content, sender: 'bot', timestamp: Date.now(), isStreaming: true }];
                    }
                });
            });

            unlistenDone = await listen('chat-stream-done', async (event: any) => {
                const metrics = event.payload;
                setStreamMetrics(metrics);

                // Get the full message content
                let fullMessage = '';
                setChatMessages(prev => {
                    const lastMsg = prev[prev.length - 1];
                    if (lastMsg && lastMsg.sender === 'bot') {
                        fullMessage = lastMsg.text;
                        return [...prev.slice(0, -1), { ...lastMsg, isStreaming: false }];
                    }
                    return prev;
                });

                // Check for tool calls
                const toolRegex = /<tool_call>\s*<name>(.*?)<\/name>\s*<arguments>(.*?)<\/arguments>\s*<\/tool_call>/s;
                const match = toolRegex.exec(fullMessage);

                if (match) {
                    const toolName = match[1].trim();
                    const argsStr = match[2].trim();
                    let args = {};
                    try {
                        args = JSON.parse(argsStr);
                    } catch (e) {
                        addLogMessage(`Failed to parse tool args: ${e}`);
                    }

                    addLogMessage(`Using tool: ${toolName}`);
                    setIsSending(true); // Keep sending state active

                    try {
                        // Execute tool
                        const result = await invoke('execute_tool_command', { tool_name: toolName, args });

                        // Add tool result to chat
                        const resultMsg = JSON.stringify(result);
                        setChatMessages(prev => [...prev, {
                            text: `Tool Result (${toolName}):\n${resultMsg}`,
                            sender: 'system',
                            timestamp: Date.now()
                        }]);

                        // Send result back to LLM
                        const nextMessage = `Tool result for ${toolName}:\n${resultMsg}\n\nPlease parse this result and answer the user's original request.`;

                        await invoke('send_chat_message_streaming_command', {
                            host: '127.0.0.1',
                            port: 8080, // Using default port as accessing modelConfig here might be tricky in closure, but state is safer
                            message: nextMessage,
                            systemPrompt: buildEnhancedSystemPrompt(), // Maintain system prompt
                            temperature,
                            topP,
                            topK,
                            ctxSize: contextSize,
                        });

                    } catch (error) {
                        addLogMessage(`Tool execution failed: ${error}`);
                        setIsSending(false);
                    }
                } else {
                    setIsSending(false);
                }
            });
        };

        setupListeners();
        return () => {
            if (unlistenChunk) unlistenChunk();
            if (unlistenDone) unlistenDone();
        };
    }, []);

    // Auto-scroll
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [chatMessages]);

    // AUTO-START: When model changes, start server automatically
    useEffect(() => {
        const restart = async () => {
            if (selectedBaseModel) {
                // If running or loading (or 'ready' but switch happened), stop first
                // Actually if 'idle', just start. If 'ready'/'loading'/'error', restart.
                // But serverStatus might lag.
                // Safest: always stop if status isn't idle, then start.
                if (infServerStatus !== 'idle') {
                    await handleStopServer();
                    // Tiny delay to ensure port release
                    await new Promise(r => setTimeout(r, 500));
                }
                handleStartServer(); // Start new
            }
        };
        restart();
    }, [selectedBaseModel]);

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
            const ggufs = resources.filter(r => r.type === 'gguf');
            setAvailableGGUFModels(ggufs.filter(r => !r.is_mmproj).map(r => r.path.replace('data/models/', '')));
            // mmproj models filtered out for now

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

    const handleStartServer = async () => {
        if (!selectedBaseModel) return;
        setInfServerStatus('loading');

        try {
            const port = modelConfig?.serverPort || 8080;
            // invoke start command (this spawns the process and returns immediately)
            await invoke('start_llama_server_command', {
                modelPath: selectedBaseModel,
                mmprojPath: '',
                loraPath: selectedLoraAdapter || null,
                port: port,
                gpuLayers: infGpuLayers,
                ctxSize: contextSize,
                batchSize: infBatchSize,
                ubatchSize: infUbatchSize,
                threads: infThreads > 0 ? infThreads : null,
                flashAttn: infFlashAttn,
                noMmap: infNoMmap,
            });

            addLogMessage('Server process launched. Waiting for readiness...');

            // Poll for health
            let retries = 0;
            const maxRetries = 60; // 30 seconds (500ms interval)
            const pollInterval = setInterval(async () => {
                const isHealthy = await checkServerHealth(port);
                if (isHealthy) {
                    clearInterval(pollInterval);
                    setIsServerRunning(true);
                    setInfServerStatus('ready');
                    addLogMessage('Server is ready and accepting requests.');
                } else {
                    retries++;
                    if (retries > maxRetries) {
                        clearInterval(pollInterval);
                        setInfServerStatus('error');
                        setIsServerRunning(false);
                        addLogMessage('Error: Server startup timed out.');
                        // Try to kill it?
                        await invoke('stop_llama_server_command');
                    }
                }
            }, 500);

        } catch (error) {
            addLogMessage(`Error starting server: ${error}`);
            setInfServerStatus('error');
        }
    };

    const handleStopServer = async () => {
        try {
            await invoke('stop_llama_server_command');
            // Don't clear history automatically on stop unless desired? 
            // User requested "loading/unloading logic can sometimes just leave no model loaded without obvious way to load".
            // Keeping chat history intact is usually better UX.
            setIsServerRunning(false);
            setInfServerStatus('idle');
            addLogMessage('Server stopped');
        } catch (error) {
            addLogMessage(`Error stopping server: ${error}`);
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
            setChatMessages(prev => [...prev, {
                text: msg,
                sender: 'user',
                timestamp: Date.now(),
                image: pendingImage // Optimistic update (need to add image prop to type if displaying)
            }]);
        }

        setIsSending(true);
        setIsPromptProcessing(true); // Start processing animation
        setPromptProgress(0); // Optional: if we can't get realprogress, we might animate this dummy
        setStreamMetrics(null);

        // Construct payload
        // If image exists, send array format
        let payload: any = msg;
        if (pendingImage) {
            payload = [
                { type: "text", text: msg },
                { type: "image_url", image_url: { url: pendingImage } }
            ];
        }

        try {
            await invoke('send_chat_message_streaming_command', {
                host: '127.0.0.1',
                port: modelConfig?.serverPort || 8080,
                message: payload,
                systemPrompt: buildEnhancedSystemPrompt(),
                temperature,
                topP,
                topK,
                ctxSize: contextSize,
            });
        } catch (error) {
            addLogMessage(`Error sending message: ${error}`);
            setChatMessages(prev => [...prev, { text: 'Error: Failed to send message.', sender: 'system', timestamp: Date.now() }]);
            setIsSending(false);
        }
    };

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
    };

    // Generate command preview
    const commandPreview = useMemo(() => {
        if (!selectedBaseModel) return '';
        const parts = ['llama-server'];
        parts.push(`--model "${selectedBaseModel}"`);
        parts.push(`--port ${modelConfig?.serverPort || 8080}`);
        parts.push(`--ctx-size ${contextSize}`);
        parts.push(`--n-gpu-layers ${infGpuLayers}`);
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

    const isCanvasVisible = infEnableCanvas && (!!canvasContent || canvasStreamingEdit.isActive);

    return (
        <div style={{ display: 'flex', height: 'calc(100vh - 100px)', overflow: 'hidden' }}>
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
                    background: 'linear-gradient(135deg, rgba(30,30,40,0.95) 0%, rgba(25,25,35,0.95) 100%)',
                    borderBottom: '1px solid rgba(255,255,255,0.08)',
                    flexShrink: 0
                }}>
                    {/* Model Selector */}
                    <div style={{ flex: 1, maxWidth: '300px' }}>
                        <Select
                            value={selectedBaseModel}
                            onChange={(val) => {
                                if (isServerRunning) handleStopServer();
                                setSelectedBaseModel(val);
                            }}
                            options={[
                                { value: '', label: 'Select a model...' },
                                ...availableGGUFModels.map(m => ({ value: m, label: m }))
                            ]}
                            placeholder="Select a model..."
                            style={{ width: '100%' }}
                        />
                    </div>


                    {/* LoRA Selector */}
                    <div style={{ flex: 1, maxWidth: '250px' }}>
                        <select
                            value={selectedProject}
                            onChange={(e) => {
                                setSelectedProject(e.target.value);
                                if (!e.target.value) {
                                    setSelectedCheckpoint('');
                                    setSelectedLoraAdapter('');
                                } else {
                                    const project = projectLoras.find(p => p.project_name === e.target.value);
                                    if (project && project.checkpoints.length > 0) {
                                        setSelectedCheckpoint(project.checkpoints[0].name);
                                        setSelectedLoraAdapter(project.checkpoints[0].path);
                                    }
                                }
                            }}
                            disabled={infServerStatus === 'loading'}
                            style={{
                                width: '100%',
                                padding: '10px 14px',
                                background: 'rgba(0,0,0,0.3)',
                                border: '1px solid rgba(167,139,250,0.3)',
                                borderRadius: '10px',
                                color: selectedProject ? '#a78bfa' : '#9ca3af',
                                fontSize: '14px',
                                cursor: 'pointer',
                                outline: 'none'
                            }}
                        >
                            <option value="">No LoRA</option>
                            {projectLoras.map(p => (
                                <option key={p.project_name} value={p.project_name}>🔗 {p.project_name}</option>
                            ))}
                        </select>
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
                            onClick={handleStopServer}
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

                {/* Main Content */}
                <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
                    {/* Chat Area */}
                    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', background: 'rgba(15,15,20,0.5)' }}>
                        {/* Messages */}
                        <div style={{ flex: 1, overflowY: 'auto', padding: '24px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
                            {chatMessages.length === 0 && (
                                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', opacity: 0.4 }}>
                                    <Bot size={56} style={{ marginBottom: '16px' }} />
                                    <p style={{ fontSize: '16px', color: '#9ca3af' }}>
                                        {infServerStatus === 'idle' ? 'Select a model to start chatting' :
                                            infServerStatus === 'loading' ? 'Loading model...' :
                                                'Start a conversation...'}
                                    </p>
                                </div>
                            )}

                            {chatMessages.map((msg, idx) => (
                                <div key={idx} style={{ display: 'flex', gap: '12px', flexDirection: msg.sender === 'user' ? 'row-reverse' : 'row', alignItems: 'flex-start' }}>
                                    <div style={{ width: '36px', height: '36px', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0, background: msg.sender === 'user' ? 'var(--accent-primary)' : 'rgba(255,255,255,0.1)' }}>
                                        {msg.sender === 'user' ? <User size={18} /> : <Bot size={18} />}
                                    </div>
                                    <div style={{ flex: 1, maxWidth: '75%' }}>
                                        {editingIndex === idx ? (
                                            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
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
                                                <div style={{
                                                    padding: '14px 18px',
                                                    borderRadius: '18px',
                                                    borderTopRightRadius: msg.sender === 'user' ? '4px' : '18px',
                                                    borderTopLeftRadius: msg.sender === 'user' ? '18px' : '4px',
                                                    background: msg.sender === 'user' ? 'var(--accent-primary)' : 'rgba(255,255,255,0.08)',
                                                    color: msg.sender === 'user' ? 'white' : '#e5e7eb'
                                                }}>
                                                    {msg.sender === 'bot' ? (() => {
                                                        const { thinking, response } = parseMessageContent(msg.text);
                                                        return (
                                                            <>
                                                                {thinking && <ThinkingBlock content={thinking} />}
                                                                <MarkdownRenderer content={response} />
                                                            </>
                                                        );
                                                    })() : (
                                                        <p style={{ whiteSpace: 'pre-wrap', lineHeight: 1.6, margin: 0 }}>{msg.text}</p>
                                                    )}
                                                    {msg.sender === 'bot' && !msg.isStreaming && streamMetrics && idx === chatMessages.length - 1 && infShowMetrics && (
                                                        <div style={{ fontSize: '11px', color: '#6b7280', marginTop: '10px', paddingTop: '8px', borderTop: '1px solid rgba(255,255,255,0.1)', display: 'flex', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
                                                            <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}><Zap size={12} /> {streamMetrics.tokens_per_second.toFixed(1)} t/s</span>
                                                            <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}><BarChart2 size={12} /> {streamMetrics.total_tokens} tokens</span>
                                                            <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}><Clock size={12} /> Prompt: {streamMetrics.prompt_eval_time_ms.toFixed(0)}ms</span>
                                                            <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}><Clock size={12} /> Eval: {streamMetrics.eval_time_ms.toFixed(0)}ms</span>
                                                        </div>
                                                    )}
                                                </div>
                                                <div style={{ display: 'flex', gap: '6px', marginTop: '6px', justifyContent: msg.sender === 'user' ? 'flex-end' : 'flex-start' }}>
                                                    <button onClick={() => handleEditStart(idx)} style={{ padding: '4px 8px', background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '6px', color: 'rgba(255,255,255,0.5)', cursor: 'pointer', fontSize: '11px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                                                        <Edit2 size={10} /> Edit
                                                    </button>
                                                    <button onClick={() => handleDeleteMessage(idx)} style={{ padding: '4px 8px', background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.2)', borderRadius: '6px', color: 'rgba(239,68,68,0.7)', cursor: 'pointer', fontSize: '11px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                                                        <Trash2 size={10} /> Delete
                                                    </button>
                                                </div>
                                            </>
                                        )}
                                    </div>
                                </div>
                            ))}
                            <div ref={messagesEndRef} />
                        </div>

                        {/* Input */}
                        <div style={{ padding: '16px 24px', borderTop: '1px solid rgba(255,255,255,0.06)', background: 'rgba(20,20,25,0.8)' }}>
                            {/* Image Preview */}
                            {pendingImage && (
                                <div style={{ marginBottom: '12px', display: 'flex' }}>
                                    <div style={{ position: 'relative', borderRadius: '8px', overflow: 'hidden', border: '1px solid rgba(255,255,255,0.1)' }}>
                                        <img src={pendingImage} alt="preview" style={{ height: '80px', display: 'block' }} />
                                        <button
                                            onClick={handleRemoveImage}
                                            style={{ position: 'absolute', top: 2, right: 2, background: 'rgba(0,0,0,0.6)', borderRadius: '50%', padding: '2px', border: 'none', color: 'white', cursor: 'pointer' }}
                                        >
                                            <X size={12} />
                                        </button>
                                    </div>
                                </div>
                            )}

                            <div style={{ display: 'flex', gap: '12px', alignItems: 'center', position: 'relative' }}>
                                <input
                                    type="file"
                                    ref={fileInputRef}
                                    onChange={handleFileSelect}
                                    style={{ display: 'none' }}
                                    accept="image/*"
                                />
                                <button
                                    onClick={() => fileInputRef.current?.click()}
                                    style={{ padding: '10px', background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '10px', color: availableGGUFModels.length > 0 ? '#e4e4e7' : '#52525b', cursor: 'pointer' }}
                                    title="Attach Image"
                                >
                                    <Paperclip size={18} />
                                </button>

                                {chatMessages.length > 0 && (
                                    <button onClick={handleClearConversation} style={{ padding: '10px', background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '10px', color: '#9ca3af', cursor: 'pointer' }} title="Clear Chat">
                                        <RotateCcw size={18} />
                                    </button>
                                )}
                                <input
                                    style={{ flex: 1, padding: '14px 18px', background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '12px', color: 'white', fontSize: '15px', outline: 'none' }}
                                    placeholder={
                                        infServerStatus === 'ready'
                                            ? (isPromptProcessing ? 'Processing prompt...' : 'Type a message...')
                                            : 'Load a model to start chatting'
                                    }
                                    value={inputMessage}
                                    onChange={(e) => setInputMessage(e.target.value)}
                                    onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleSendMessage()}
                                    disabled={infServerStatus !== 'ready' || isSending}
                                />
                                {isPromptProcessing && (
                                    <div style={{ position: 'absolute', right: '90px', top: '50%', transform: 'translateY(-50%)', display: 'flex', alignItems: 'center', gap: '6px', pointerEvents: 'none', zIndex: 10 }}>
                                        <div className="processing-wiggle" style={{
                                            fontSize: '12px', color: '#a78bfa', fontWeight: 600,
                                            animation: 'wiggle 1s ease-in-out infinite',
                                            padding: '4px 8px', borderRadius: '4px',
                                            background: 'rgba(0,0,0,0.4)', backdropFilter: 'blur(4px)'
                                        }}>
                                            Processing {promptProgress !== null ? `${promptProgress}%` : '...'}
                                        </div>
                                    </div>
                                )}
                                <Button
                                    variant="primary"
                                    onClick={() => handleSendMessage()}
                                    disabled={infServerStatus !== 'ready' || isSending || !inputMessage.trim()}
                                    isLoading={isSending}
                                    style={{ padding: '14px 20px', borderRadius: '12px' }}
                                >
                                    <Send size={18} />
                                </Button>
                            </div>
                        </div>
                    </div>

                    {/* Sidebar */}
                    {showSidebar && (
                        <div style={{
                            width: '320px',
                            borderLeft: '1px solid rgba(255,255,255,0.08)',
                            background: 'rgba(20,20,28,0.6)',
                            display: 'flex',
                            flexDirection: 'column',
                            overflowY: 'auto'
                        }}>
                            {/* Parameters Section */}
                            <div style={{ padding: '20px', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
                                <h3 style={{ fontSize: '12px', fontWeight: 600, color: '#6b7280', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '16px' }}>Parameters</h3>

                                <div style={{ marginBottom: '16px' }}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px', marginBottom: '6px' }}>
                                        <span>Temperature</span>
                                        <span style={{ color: '#a78bfa' }}>{temperature}</span>
                                    </div>
                                    <input type="range" min="0" max="2" step="0.1" value={temperature} onChange={(e) => setTemperature(Number(e.target.value))} style={{ width: '100%', accentColor: '#a78bfa' }} />
                                </div>

                                <div style={{ marginBottom: '16px' }}>
                                    <label style={{ fontSize: '13px', display: 'block', marginBottom: '6px' }}>System Prompt</label>
                                    <textarea
                                        value={systemPrompt}
                                        onChange={(e) => setSystemPrompt(e.target.value)}
                                        style={{ width: '100%', height: '80px', padding: '10px', background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', color: 'white', fontSize: '13px', resize: 'vertical' }}
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

                                        <Input label="Context Size" type="number" value={contextSize} onChange={(e) => setContextSize(Number(e.target.value))} />
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
                                        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                                            <label style={{ display: 'flex', alignItems: 'center', gap: '10px', cursor: 'pointer' }}>
                                                <input type="checkbox" checked={infFlashAttn} onChange={(e) => setInfFlashAttn(e.target.checked)} style={{ accentColor: '#a78bfa' }} />
                                                <Zap size={14} /> <span style={{ fontSize: '13px' }}>Flash Attention</span>
                                            </label>

                                            <label style={{ display: 'flex', alignItems: 'center', gap: '10px', cursor: 'pointer' }}>
                                                <input type="checkbox" checked={infNoMmap} onChange={(e) => setInfNoMmap(e.target.checked)} style={{ accentColor: '#a78bfa' }} />
                                                <HardDrive size={14} /> <span style={{ fontSize: '13px' }}>No Memory Map</span>
                                            </label>

                                            <div>
                                                <label style={{ fontSize: '13px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '6px' }}>
                                                    <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}><Layers size={14} /> GPU Layers</span>
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
                                                <input type="number" value={infBatchSize} onChange={(e) => setInfBatchSize(Number(e.target.value))} style={{ width: '100%', padding: '8px 12px', background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', color: 'white', fontSize: '13px' }} />
                                            </div>

                                            <div>
                                                <label style={{ fontSize: '13px', marginBottom: '6px', display: 'block' }}>Micro-Batch Size</label>
                                                <input type="number" value={infUbatchSize} onChange={(e) => setInfUbatchSize(Number(e.target.value))} style={{ width: '100%', padding: '8px 12px', background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', color: 'white', fontSize: '13px' }} />
                                            </div>

                                            <div>
                                                <label style={{ fontSize: '13px', marginBottom: '6px', display: 'block' }}>Threads (0 = auto)</label>
                                                <input type="number" value={infThreads} onChange={(e) => setInfThreads(Number(e.target.value))} style={{ width: '100%', padding: '8px 12px', background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', color: 'white', fontSize: '13px' }} />
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
                                    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                                        <label style={{ display: 'flex', alignItems: 'center', gap: '10px', cursor: 'pointer' }}>
                                            <input type="checkbox" checked={infShowMetrics} onChange={(e) => setInfShowMetrics(e.target.checked)} style={{ accentColor: '#10b981' }} />
                                            <Activity size={14} style={{ color: '#10b981' }} /> <span style={{ fontSize: '13px' }}>Show Metrics</span>
                                        </label>

                                        <div style={{ borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '12px', marginTop: '4px' }}>
                                            <p style={{ fontSize: '11px', color: '#6b7280', marginBottom: '10px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Agentic Tools</p>

                                            <label style={{ display: 'flex', alignItems: 'center', gap: '10px', cursor: 'pointer', marginBottom: '10px' }}>
                                                <input type="checkbox" checked={infEnableWebSearch} onChange={(e) => setInfEnableWebSearch(e.target.checked)} style={{ accentColor: '#3b82f6' }} />
                                                <Search size={14} style={{ color: '#3b82f6' }} /> <span style={{ fontSize: '13px' }}>Web Search</span>
                                            </label>

                                            <label style={{ display: 'flex', alignItems: 'center', gap: '10px', cursor: 'pointer' }}>
                                                <input type="checkbox" checked={infEnableCodeExec} onChange={(e) => setInfEnableCodeExec(e.target.checked)} style={{ accentColor: '#f59e0b' }} />
                                                <Code size={14} style={{ color: '#f59e0b' }} /> <span style={{ fontSize: '13px' }}>Code Execution</span>
                                            </label>

                                            <button
                                                onClick={() => setShowToolCreator(true)}
                                                style={{
                                                    marginTop: '8px', padding: '8px',
                                                    background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)',
                                                    borderRadius: '8px', color: '#9ca3af', fontSize: '12px',
                                                    cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '8px', justifyContent: 'center',
                                                    width: '100%'
                                                }}
                                            >
                                                <Settings2 size={12} /> Create Custom Tool
                                            </button>
                                        </div>

                                        <div style={{ borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '12px', marginTop: '4px' }}>
                                            <p style={{ fontSize: '11px', color: '#6b7280', marginBottom: '10px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Display</p>

                                            <label style={{ display: 'flex', alignItems: 'center', gap: '10px', cursor: 'pointer' }}>
                                                <input type="checkbox" checked={infEnableCanvas} onChange={(e) => setInfEnableCanvas(e.target.checked)} style={{ accentColor: '#8b5cf6' }} />
                                                <PenTool size={14} style={{ color: '#8b5cf6' }} /> <span style={{ fontSize: '13px' }}>Canvas Mode</span>
                                            </label>
                                        </div>
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
                    )}
                </div>
            </div>

            {/* Canvas Side Panel */}
            {
                isCanvasVisible && (
                    <div style={{
                        width: '45%',
                        minWidth: '400px',
                        background: '#0f0f14',
                        borderLeft: '1px solid rgba(255,255,255,0.08)',
                        display: 'flex',
                        flexDirection: 'column',
                        padding: '20px',
                        gap: '16px'
                    }}>
                        <Canvas
                            content={canvasContent}
                            mode={canvasMode}
                            onContentChange={(newContent) => setCanvasContent(newContent)}
                            isEditable={!isSending}
                            streamingEdit={canvasStreamingEdit.isActive ? canvasStreamingEdit : null}
                        />
                    </div>
                )
            }

            <ToolCreatorModal
                isOpen={showToolCreator}
                onClose={() => setShowToolCreator(false)}
                onSaveSuccess={() => addLogMessage('Custom tool created successfully')}
            />
        </div>
    );
};

export default InferencePage;
