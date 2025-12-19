import React, { useState, useEffect, useRef, useMemo } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { useApp } from '../context/AppContext';
import { Button } from '../components/Button';
import { Input } from '../components/Input';
import { MarkdownRenderer } from '../components/MarkdownRenderer';
import { CanvasPanel } from '../components/CanvasPanel';
import ToolCreatorModal from '../components/ToolCreatorModal';
import { Select } from '../components/Select';
import {
    Send, Settings2, Bot, User, Trash2, Edit2, RotateCcw,
    ChevronDown, Cpu, Zap, HardDrive, Layers, Terminal,
    Loader2, CheckCircle2, XCircle, Power, Paperclip, X,
    Settings, Wrench, BarChart2, Clock, PenTool, Search, Code, Brain, Activity
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
        infCanvasVisible, setInfCanvasVisible,
        infCanvasArtifacts, setInfCanvasArtifacts,
        infAutoFit, setInfAutoFit,
        infInferenceEngine, setInfInferenceEngine,
    } = useApp();

    const [showTools, setShowTools] = useState(false);

    const messagesEndRef = useRef<HTMLDivElement>(null);

    // Resources
    const [modelOptions, setModelOptions] = useState<Option[]>([]);
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
    // Streaming artifact tracking
    const streamingArtifactId = useRef<string | null>(null);

    // Vision
    const [pendingImage, setPendingImage] = useState<string | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    // Backend Logs / Progress
    const [promptProgress, setPromptProgress] = useState<number | null>(null); // 0-100 or null
    const [isPromptProcessing, setIsPromptProcessing] = useState(false);

    const [showToolCreator, setShowToolCreator] = useState(false);

    // Layout Resizing
    const [lastLoadedModel, setLastLoadedModel] = useState<string>(''); // Track what's actually running
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
        // Do not reset detected model here, let Context persist it.
        // If the model is invalid for the new engine, the user can select a new one.
    }, [infInferenceEngine]);

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
                            // Check for canvas content
                            const contentMatch = /<canvas_content type="(code|markdown|mermaid)"(?: language="(.*?)")?>(.*)/s.exec(fullText);
                            if (contentMatch) {
                                const type = contentMatch[1] as any;
                                const language = contentMatch[2];
                                // Check if we have closing tag yet
                                const closingMatch = /(.*?)<\/canvas_content>/s.exec(contentMatch[3]);
                                const content = closingMatch ? closingMatch[1] : contentMatch[3];

                                setInfCanvasArtifacts(prev => {
                                    // If streamingArtifactId is null, create new
                                    if (!streamingArtifactId.current) {
                                        const newId = `artifact-${Date.now()}`;
                                        streamingArtifactId.current = newId;
                                        setInfCanvasVisible(true);
                                        return [...prev, {
                                            id: newId,
                                            title: 'Generated Content',
                                            mode: type,
                                            language: language,
                                            content: content,
                                            createdAt: Date.now()
                                        }];
                                    }

                                    // Update existing
                                    return prev.map(a => a.id === streamingArtifactId.current ? {
                                        ...a,
                                        content: content,
                                        mode: type,
                                        language: language
                                    } : a);
                                });
                            }

                            // Edits not yet supported in streaming for new canvas, 
                            // as we need to match against existing content which is hard in streaming.
                            // We rely on full replacement for now or handle edits in 'chat-stream-done' if needed.
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

                // Reset streaming artifact mapping
                streamingArtifactId.current = null;

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
    // Also auto-switch engine based on model type
    useEffect(() => {
        const restart = async () => {
            if (selectedBaseModel) {
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

                // Restart if model changed or server is not running
                if (infServerStatus === 'idle' || infServerStatus === 'error' || lastLoadedModel !== selectedBaseModel) {
                    addLogMessage(`[AUTO] Starting/Restarting server for model: ${selectedBaseModel}`);
                    handleStartServer();
                } else {
                    addLogMessage(`[DEBUG] Server is already ${infServerStatus} with correct model, skipping auto-start.`);
                }
            }
        };
        restart();
    }, [selectedBaseModel, modelOptions, lastLoadedModel]);

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
                    const name = isGGUF ? r.path.replace('data/models/', '') : r.name;

                    // Check if this model has a matching mmproj (vision)
                    const hasVision = resources.some(res => res.is_mmproj && res.path.includes(name.split('-')[0]));

                    return {
                        value: isGGUF ? r.path : name,
                        label: isGGUF ? name : `${name} (Experimental)`,
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

    const handleStartServer = async () => {
        if (!selectedBaseModel) {
            addLogMessage('[DEBUG] handleStartServer called but no model selected.');
            return;
        }

        addLogMessage(`[DEBUG] Starting server with engine=${infInferenceEngine}, model=${selectedBaseModel}`);
        setInfServerStatus('loading');

        try {
            const port = modelConfig?.serverPort || 8080;
            addLogMessage(`[DEBUG] Using port ${port}`);

            if (infInferenceEngine === 'transformers') {
                addLogMessage('[DEBUG] Invoking start_transformers_server_command...');
                await invoke('start_transformers_server_command', {
                    modelPath: selectedBaseModel,
                    port: port
                });
            } else {
                addLogMessage('[DEBUG] Invoking start_llama_server_command...');
                await invoke('start_llama_server_command', {
                    modelPath: selectedBaseModel,
                    mmprojPath: '',
                    loraPath: selectedLoraAdapter || null,
                    port: port,
                    gpuLayers: infAutoFit ? null : infGpuLayers,
                    ctxSize: contextSize,
                    batchSize: infBatchSize,
                    ubatchSize: infUbatchSize,
                    threads: infThreads > 0 ? infThreads : null,
                    flashAttn: infFlashAttn,
                    noMmap: infNoMmap,
                    autoFit: infAutoFit,
                });
            }

            addLogMessage(`Server (${infInferenceEngine}) process launched. Waiting for readiness...`);

            // Poll for health
            let retries = 0;
            const maxRetries = 60; // 30 seconds (500ms interval)
            const pollInterval = setInterval(async () => {
                const isHealthy = await checkServerHealth(port);
                if (isHealthy) {
                    clearInterval(pollInterval);
                    setIsServerRunning(true);
                    setInfServerStatus('ready');
                    setLastLoadedModel(selectedBaseModel); // Track successfully loaded model
                    addLogMessage('Server is ready and accepting requests.');
                } else {
                    retries++;
                    if (retries % 10 === 0) {
                        addLogMessage(`[DEBUG] Health check attempt ${retries}/${maxRetries}...`);
                    }
                    if (retries > maxRetries) {
                        clearInterval(pollInterval);
                        setInfServerStatus('error');
                        setIsServerRunning(false);
                        addLogMessage('Error: Server startup timed out. Check logs above for backend errors.');
                        await handleStopServer();
                    }
                }
            }, 500);

        } catch (error) {
            addLogMessage(`[ERROR] Starting server failed: ${error}`);
            setInfServerStatus('error');
        }
    };

    const handleStopServer = async () => {
        try {
            // Robust handling: stop BOTH possible backend processes regardless of current setting
            // to ensure the port is truly cleared during switches.
            await Promise.allSettled([
                invoke('stop_transformers_server_command'),
                invoke('stop_llama_server_command')
            ]);

            setIsServerRunning(false);
            setInfServerStatus('idle');
            addLogMessage('Inference engine stopped');
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
        if (infAutoFit) {
            parts.push('--fit on');
            parts.push('--fit-margin 1024');
        } else {
            parts.push(`--n-gpu-layers ${infGpuLayers}`);
        }
        parts.push(`--batch-size ${infBatchSize}`);
        parts.push(`--ubatch-size ${infUbatchSize}`);
        if (infFlashAttn) parts.push('--flash-attn on');
        if (infNoMmap) parts.push('--no-mmap');
        if (infThreads > 0) parts.push(`--threads ${infThreads}`);
        if (selectedLoraAdapter) parts.push(`--lora "${selectedLoraAdapter}"`);
        return parts.join(' \\\n  ');
    }, [selectedBaseModel, contextSize, infGpuLayers, infBatchSize, infUbatchSize, infFlashAttn, infNoMmap, infThreads, selectedLoraAdapter, modelConfig, infAutoFit]);

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
                    <div style={{ flex: 2, maxWidth: '500px' }}>
                        <Select
                            value={selectedBaseModel}
                            onChange={(val) => {
                                // Detect engine from selected option if possible, or just use GGUF as default
                                const opt = modelOptions.find(o => o.value === val);
                                if (opt) {
                                    setInfInferenceEngine(opt.engine === 'GGUF' ? 'llamacpp' : 'transformers');
                                }
                                if (isServerRunning) handleStopServer();
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


                {/* Main Content Area - Flex Row */}
                <div style={{ display: 'flex', flex: 1, overflow: 'hidden', position: 'relative' }}>

                    {/* 1. Chat Column */}
                    <div style={{
                        flex: 1,
                        display: 'flex',
                        flexDirection: 'column',
                        background: 'rgba(15,15,20,0.5)',
                        minWidth: '300px',
                        overflow: 'hidden'
                    }}>
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
                                    <div style={{ flex: 1, maxWidth: '85%' }}>
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

                        {/* Input Area */}
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
                                    style={{ padding: '10px', background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '10px', color: modelOptions.length > 0 ? '#e4e4e7' : '#52525b', cursor: 'pointer' }}
                                    title="Attach Image"
                                >
                                    <Paperclip size={18} />
                                </button>

                                {infEnableCanvas && (
                                    <button
                                        onClick={() => setInfCanvasVisible(!infCanvasVisible)}
                                        style={{ padding: '10px', background: infCanvasVisible ? 'rgba(139,92,246,0.15)' : 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '10px', color: infCanvasVisible ? '#a78bfa' : '#9ca3af', cursor: 'pointer' }}
                                        title="Toggle Canvas"
                                    >
                                        <PenTool size={18} />
                                    </button>
                                )}

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
                            <div style={{ width: canvasWidth, minWidth: '300px', borderLeft: '1px solid rgba(255,255,255,0.08)', background: '#121216', display: 'flex', flexDirection: 'column', flexShrink: 0 }}>
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
                                borderLeft: '1px solid rgba(255,255,255,0.08)',
                                background: 'rgba(20,20,28,0.6)',
                                display: 'flex',
                                flexDirection: 'column',
                                overflowY: 'auto',
                                flexShrink: 0
                            }}>
                                {/* Parameters Section */}
                                <div style={{ padding: '20px', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
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

                                            <Input
                                                label="Context Size"
                                                type="number"
                                                value={contextSize}
                                                onChange={(e) => setContextSize(Number(e.target.value))}
                                                tooltip="The maximum number of tokens the model can remember (Context Window)."
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
                                            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                                                <label style={{ display: 'flex', alignItems: 'center', gap: '10px', cursor: 'pointer' }}>
                                                    <input type="checkbox" checked={infAutoFit} onChange={(e) => setInfAutoFit(e.target.checked)} style={{ accentColor: '#a78bfa' }} />
                                                    <Cpu size={14} /> <span style={{ fontSize: '13px' }}>Auto-Fit Memory (--fit)</span>
                                                </label>

                                                <label style={{ display: 'flex', alignItems: 'center', gap: '10px', cursor: 'pointer' }}>
                                                    <input type="checkbox" checked={infFlashAttn} onChange={(e) => setInfFlashAttn(e.target.checked)} style={{ accentColor: '#a78bfa' }} />
                                                    <Zap size={14} /> <span style={{ fontSize: '13px' }}>Flash Attention</span>
                                                </label>

                                                <label style={{ display: 'flex', alignItems: 'center', gap: '10px', cursor: 'pointer' }}>
                                                    <input type="checkbox" checked={infNoMmap} onChange={(e) => setInfNoMmap(e.target.checked)} style={{ accentColor: '#a78bfa' }} />
                                                    <HardDrive size={14} /> <span style={{ fontSize: '13px' }}>No Memory Map</span>
                                                </label>

                                                <div style={{ opacity: infAutoFit ? 0.5 : 1, pointerEvents: infAutoFit ? 'none' : 'auto' }}>
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
                        </>
                    )}
                </div>

                <ToolCreatorModal
                    isOpen={showToolCreator}
                    onClose={() => setShowToolCreator(false)}
                    onSaveSuccess={() => addLogMessage('Custom tool created successfully')}
                />
            </div>
        </div>
    );
};

export default InferencePage;
