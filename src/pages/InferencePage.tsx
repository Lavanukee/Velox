import React, { useState, useEffect, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { useApp } from '../context/AppContext';
import { Button } from '../components/Button';
import { Input, Select } from '../components/Input';
import { Card } from '../components/Card';
import { Send, Play, Square, Settings2, Bot, User, Trash2, Edit2, RotateCcw } from 'lucide-react';

interface InferencePageProps {
    modelConfig: any;
    addLogMessage: (message: string) => void;
}

const InferencePage: React.FC<InferencePageProps> = ({ modelConfig, addLogMessage }) => {
    const {
        userMode,
        chatMessages, setChatMessages,
        inputMessage, setInputMessage,
        selectedBaseModel, setSelectedBaseModel,
        selectedMmproj, setSelectedMmproj,
        selectedLoraAdapter, setSelectedLoraAdapter,
        temperature, setTemperature,
        contextSize, setContextSize,
        topP, setTopP,
        topK, setTopK,
        systemPrompt, setSystemPrompt,
        isServerRunning, setIsServerRunning
    } = useApp();

    const messagesEndRef = useRef<HTMLDivElement>(null);

    // Resources
    const [availableGGUFModels, setAvailableGGUFModels] = useState<string[]>([]);
    const [availableMmprojModels, setAvailableMmprojModels] = useState<string[]>([]);

    // Project-based LoRAs
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

    const [projectLoras, setProjectLoras] = useState<ProjectLoraInfo[]>([]);
    const [selectedProject, setSelectedProject] = useState('');
    const [selectedCheckpoint, setSelectedCheckpoint] = useState('');

    // State
    const [isStartingServer, setIsStartingServer] = useState(false);
    const [isSending, setIsSending] = useState(false);
    const [showConfig, setShowConfig] = useState(true);
    const [editingIndex, setEditingIndex] = useState<number | null>(null);
    const [editText, setEditText] = useState('');
    const [streamMetrics, setStreamMetrics] = useState<{
        tokens_per_second: number;
        prompt_eval_time_ms: number;
        eval_time_ms: number;
        total_tokens: number;
    } | null>(null);

    // Initialize from config if global state is empty
    useEffect(() => {
        if (modelConfig) {
            if (!selectedBaseModel && modelConfig.llamaModelPath) setSelectedBaseModel(modelConfig.llamaModelPath);
            if (!selectedMmproj && modelConfig.mmprojPath) setSelectedMmproj(modelConfig.mmprojPath);
            if (!selectedLoraAdapter && modelConfig.loraPath) setSelectedLoraAdapter(modelConfig.loraPath);
        }
    }, [modelConfig]);

    // Check and download inference engine if missing
    useEffect(() => {
        const checkBinary = async () => {
            try {
                const exists: boolean = await invoke('check_llama_binary_command');
                if (!exists) {
                    addLogMessage('Inference engine not found. Downloading...');
                    // This might take a while, so we log progress
                    await invoke('download_llama_binary_command');
                    addLogMessage('Inference engine installed successfully.');
                }
            } catch (error) {
                addLogMessage(`Error checking/downloading inference engine: ${error}`);
            }
        };
        checkBinary();
    }, []);

    useEffect(() => {
        loadResources();
        checkServerStatus();
        const interval = setInterval(checkServerStatus, 5000);
        return () => clearInterval(interval);
    }, []);

    useEffect(() => {
        let unlistenChunk: () => void;
        let unlistenDone: () => void;

        const setupListeners = async () => {
            unlistenChunk = await listen('chat-stream-chunk', (event: any) => {
                const chunk = event.payload as { content: string };
                setChatMessages(prev => {
                    const lastMsg = prev[prev.length - 1];
                    if (lastMsg && lastMsg.sender === 'bot' && lastMsg.isStreaming) {
                        return [...prev.slice(0, -1), { ...lastMsg, text: lastMsg.text + chunk.content }];
                    } else {
                        return [...prev, { text: chunk.content, sender: 'bot', timestamp: Date.now(), isStreaming: true }];
                    }
                });
            });

            unlistenDone = await listen('chat-stream-done', (event: any) => {
                const metrics = event.payload;
                setStreamMetrics(metrics);
                setIsSending(false);
                setChatMessages(prev => {
                    const lastMsg = prev[prev.length - 1];
                    if (lastMsg && lastMsg.sender === 'bot') {
                        return [...prev.slice(0, -1), { ...lastMsg, isStreaming: false }];
                    }
                    return prev;
                });
            });
        };

        setupListeners();

        return () => {
            if (unlistenChunk) unlistenChunk();
            if (unlistenDone) unlistenDone();
        };
    }, []);

    useEffect(() => {
        scrollToBottom();
    }, [chatMessages]);

    // Auto-select best checkpoint when project changes (for regular users)
    useEffect(() => {
        if (selectedProject && userMode === 'user') {
            const project = projectLoras.find(p => p.project_name === selectedProject);
            if (project && project.checkpoints.length > 0) {
                const bestCheckpoint = project.checkpoints[0];
                setSelectedCheckpoint(bestCheckpoint.name);
                setSelectedLoraAdapter(bestCheckpoint.path);
            }
        }
    }, [selectedProject, projectLoras, userMode]);

    // Update LoRA adapter when checkpoint changes (for power users)
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

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    const loadResources = async () => {
        try {
            const resources: any[] = await invoke('list_all_resources_command');
            const ggufs = resources.filter(r => r.type === 'gguf');
            setAvailableGGUFModels(ggufs.filter(r => !r.is_mmproj).map(r => r.path.replace('data/models/', '')));
            setAvailableMmprojModels(ggufs.filter(r => r.is_mmproj).map(r => r.path.replace('data/models/', '')));

            const projects: ProjectLoraInfo[] = await invoke('list_loras_by_project_command');
            setProjectLoras(projects);
        } catch (error) {
            addLogMessage(`Error loading resources: ${error}`);
        }
    };

    const checkServerStatus = async () => {
        try {
            const running: boolean = await invoke('check_llama_server_status_command');
            setIsServerRunning(running);
        } catch (error) {
            console.error(error);
        }
    };

    const handleStartServer = async () => {
        if (!selectedBaseModel) return;
        setIsStartingServer(true);
        try {
            await invoke('start_llama_server_command', {
                modelPath: selectedBaseModel,
                mmprojPath: selectedMmproj || '',
                loraPath: selectedLoraAdapter || null,
                host: modelConfig?.serverHost || '127.0.0.1',
                port: modelConfig?.serverPort || 8080,
                nGpuLayers: modelConfig?.nGpuLayers || 0,
                ctxSize: contextSize,
                batchSize: modelConfig?.batchSizeInference || 512,
                ubatchSize: modelConfig?.ubatchSize || 512,
                temp: temperature,
                noMmap: modelConfig?.noMmap || false,
                flashAttn: true,
            });
            setIsServerRunning(true);
            addLogMessage('Server started successfully');
        } catch (error) {
            addLogMessage(`Error starting server: ${error}`);
        } finally {
            setIsStartingServer(false);
        }
    };

    const handleStopServer = async () => {
        try {
            await invoke('stop_llama_server_command');
            // Clear chat history when stopping server so new session starts fresh
            await invoke('clear_chat_history_command');
            setIsServerRunning(false);
            addLogMessage('Server stopped');
        } catch (error) {
            addLogMessage(`Error stopping server: ${error}`);
        }
    };

    const handleSendMessage = async (messageToSend?: string) => {
        const msg = messageToSend || inputMessage;
        if (!msg.trim() || !isServerRunning) return;

        if (!messageToSend) {
            setInputMessage('');
        }

        // Only add user message if not resending
        if (!messageToSend) {
            setChatMessages(prev => [...prev, { text: msg, sender: 'user', timestamp: Date.now() }]);
        }

        setIsSending(true);
        setStreamMetrics(null);

        try {
            await invoke('send_chat_message_streaming_command', {
                host: modelConfig?.serverHost || '127.0.0.1',
                port: modelConfig?.serverPort || 8080,
                message: msg,
                systemPrompt,
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

    const handleDeleteMessage = (index: number) => {
        setChatMessages(prev => prev.filter((_, i) => i !== index));
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

    const handleClearConversation = async () => {
        if (confirm('Clear the entire conversation?')) {
            try {
                // Clear backend chat history so model forgets the conversation
                await invoke('clear_chat_history_command');
                // Clear frontend messages
                setChatMessages([]);
                addLogMessage('Chat history cleared');
            } catch (error) {
                addLogMessage(`Error clearing chat history: ${error}`);
            }
        }
    };

    // Check if the last message is a user message with no response
    const canResend = chatMessages.length > 0 && chatMessages[chatMessages.length - 1].sender === 'user';

    return (
        <div className="flex h-[calc(100vh-140px)] gap-6" style={{ display: 'flex', height: 'calc(100vh - 140px)', gap: '24px' }}>
            {/* Chat Area */}
            <div className="flex-1 flex flex-col bg-panel rounded-xl border border-white/10 overflow-hidden glass" style={{ flex: 1, display: 'flex', flexDirection: 'column', borderRadius: '16px', border: '1px solid rgba(255,255,255,0.1)', overflow: 'hidden', background: 'rgba(20,20,25,0.4)' }}>
                {/* Chat Header */}
                <div className="p-4 border-b border-white/5 flex justify-between items-center bg-white/5" style={{ padding: '16px', borderBottom: '1px solid rgba(255,255,255,0.05)', background: 'rgba(255,255,255,0.02)' }}>
                    <div className="flex items-center gap-3">
                        <div className={`w-2 h-2 rounded-full ${isServerRunning ? 'bg-green-500 shadow-[0_0_10px_#10b981]' : 'bg-red-500'}`} style={{ width: '8px', height: '8px', borderRadius: '50%', background: isServerRunning ? '#10b981' : '#ef4444', boxShadow: isServerRunning ? '0 0 10px #10b981' : 'none' }} />
                        <span className="text-sm font-medium text-gray-300">
                            {isServerRunning ? 'Model Online' : 'Model Offline'}
                        </span>
                    </div>
                    <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                        {chatMessages.length > 0 && (
                            <Button variant="ghost" size="sm" onClick={handleClearConversation} leftIcon={<RotateCcw size={16} />}>
                                Clear
                            </Button>
                        )}
                        <Button variant="ghost" size="sm" onClick={() => setShowConfig(!showConfig)} leftIcon={<Settings2 size={16} />}>
                            {showConfig ? 'Hide Config' : 'Show Config'}
                        </Button>
                    </div>
                </div>

                {/* Messages */}
                <div className="flex-1 overflow-y-auto p-6 space-y-6" style={{ flex: 1, overflowY: 'auto', padding: '24px', display: 'flex', flexDirection: 'column', gap: '24px' }}>
                    {chatMessages.length === 0 && (
                        <div className="flex flex-col items-center justify-center h-full text-gray-500 opacity-50">
                            <Bot size={48} className="mb-4" />
                            <p>Start a conversation...</p>
                        </div>
                    )}

                    {chatMessages.map((msg, idx) => (
                        <div key={idx} style={{ display: 'flex', gap: '16px', flexDirection: msg.sender === 'user' ? 'row-reverse' : 'row', alignItems: 'flex-start' }}>
                            <div style={{ width: '32px', height: '32px', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0, background: msg.sender === 'user' ? 'var(--accent-primary)' : 'rgba(255,255,255,0.1)' }}>
                                {msg.sender === 'user' ? <User size={16} /> : <Bot size={16} />}
                            </div>
                            <div style={{ flex: 1, maxWidth: '80%' }}>
                                {editingIndex === idx ? (
                                    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                                        <textarea
                                            value={editText}
                                            onChange={(e) => setEditText(e.target.value)}
                                            style={{
                                                background: 'rgba(0,0,0,0.3)',
                                                border: '1px solid rgba(255,255,255,0.2)',
                                                borderRadius: '8px',
                                                padding: '12px',
                                                color: 'white',
                                                minHeight: '80px',
                                                resize: 'vertical',
                                                width: '100%'
                                            }}
                                        />
                                        <div style={{ display: 'flex', gap: '8px' }}>
                                            <Button size="sm" onClick={() => handleEditSave(idx)}>Save</Button>
                                            <Button size="sm" variant="ghost" onClick={() => setEditingIndex(null)}>Cancel</Button>
                                        </div>
                                    </div>
                                ) : (
                                    <>
                                        <div className={`p-4 rounded-2xl ${msg.sender === 'user' ? 'bg-accent-primary text-white rounded-tr-none' : 'bg-white/10 text-gray-200 rounded-tl-none'}`} style={{ padding: '16px', borderRadius: '16px', borderTopRightRadius: msg.sender === 'user' ? '0' : '16px', borderTopLeftRadius: msg.sender === 'user' ? '16px' : '0', background: msg.sender === 'user' ? 'var(--accent-primary)' : 'rgba(255,255,255,0.1)', color: msg.sender === 'user' ? 'white' : '#e5e7eb' }}>
                                            <p className="whitespace-pre-wrap leading-relaxed">{msg.text}</p>
                                            {msg.sender === 'bot' && !msg.isStreaming && streamMetrics && idx === chatMessages.length - 1 && (
                                                <div className="text-xs text-gray-500 mt-2 border-t border-white/10 pt-1">
                                                    {streamMetrics.tokens_per_second.toFixed(2)} t/s · {streamMetrics.prompt_eval_time_ms.toFixed(0)}ms prompt · {streamMetrics.eval_time_ms.toFixed(0)}ms eval
                                                </div>
                                            )}
                                        </div>
                                        <div style={{ display: 'flex', gap: '6px', marginTop: '6px', justifyContent: msg.sender === 'user' ? 'flex-end' : 'flex-start' }}>
                                            <button
                                                onClick={() => handleEditStart(idx)}
                                                style={{
                                                    background: 'rgba(255,255,255,0.05)',
                                                    border: '1px solid rgba(255,255,255,0.1)',
                                                    borderRadius: '6px',
                                                    padding: '4px 8px',
                                                    color: 'rgba(255,255,255,0.6)',
                                                    cursor: 'pointer',
                                                    display: 'flex',
                                                    alignItems: 'center',
                                                    gap: '4px',
                                                    fontSize: '12px',
                                                    transition: 'all 0.2s'
                                                }}
                                                onMouseEnter={(e) => {
                                                    e.currentTarget.style.background = 'rgba(255,255,255,0.1)';
                                                    e.currentTarget.style.color = 'white';
                                                }}
                                                onMouseLeave={(e) => {
                                                    e.currentTarget.style.background = 'rgba(255,255,255,0.05)';
                                                    e.currentTarget.style.color = 'rgba(255,255,255,0.6)';
                                                }}
                                            >
                                                <Edit2 size={12} />
                                                Edit
                                            </button>
                                            <button
                                                onClick={() => handleDeleteMessage(idx)}
                                                style={{
                                                    background: 'rgba(239,68,68,0.1)',
                                                    border: '1px solid rgba(239,68,68,0.2)',
                                                    borderRadius: '6px',
                                                    padding: '4px 8px',
                                                    color: 'rgba(239,68,68,0.8)',
                                                    cursor: 'pointer',
                                                    display: 'flex',
                                                    alignItems: 'center',
                                                    gap: '4px',
                                                    fontSize: '12px',
                                                    transition: 'all 0.2s'
                                                }}
                                                onMouseEnter={(e) => {
                                                    e.currentTarget.style.background = 'rgba(239,68,68,0.2)';
                                                    e.currentTarget.style.color = '#ef4444';
                                                }}
                                                onMouseLeave={(e) => {
                                                    e.currentTarget.style.background = 'rgba(239,68,68,0.1)';
                                                    e.currentTarget.style.color = 'rgba(239,68,68,0.8)';
                                                }}
                                            >
                                                <Trash2 size={12} />
                                                Delete
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
                <div className="p-4 border-t border-white/5 bg-white/5" style={{ padding: '16px', borderTop: '1px solid rgba(255,255,255,0.05)', background: 'rgba(255,255,255,0.02)' }}>
                    <div className="flex gap-3" style={{ display: 'flex', gap: '12px' }}>
                        <input
                            className="flex-1 bg-black/20 border border-white/10 rounded-lg px-4 py-3 focus:outline-none focus:border-accent-primary transition-colors"
                            style={{ flex: 1, background: 'rgba(0,0,0,0.2)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', padding: '12px 16px', color: 'white', outline: 'none' }}
                            placeholder={isServerRunning ? (canResend ? "Resend or type new message..." : "Type a message...") : "Start the server to chat"}
                            value={inputMessage}
                            onChange={(e) => setInputMessage(e.target.value)}
                            onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleSendMessage()}
                            disabled={!isServerRunning || isSending}
                        />
                        {canResend && !inputMessage && (
                            <Button
                                variant="secondary"
                                onClick={() => handleSendMessage(chatMessages[chatMessages.length - 1].text)}
                                disabled={!isServerRunning || isSending}
                                isLoading={isSending}
                            >
                                Resend
                            </Button>
                        )}
                        <Button
                            variant="primary"
                            onClick={() => handleSendMessage()}
                            disabled={!isServerRunning || isSending || !inputMessage.trim()}
                            isLoading={isSending}
                            size="icon"
                        >
                            <Send size={20} />
                        </Button>
                    </div>
                </div>
            </div >

            {/* Configuration Sidebar - Made scrollable with max-height */}
            {
                showConfig && (
                    <div
                        style={{
                            width: '320px',
                            display: 'flex',
                            flexDirection: 'column',
                            gap: '16px',
                            height: '100%',
                            maxHeight: 'calc(100vh - 140px)',
                        }}
                    >
                        {/* Server Control - Always at top */}
                        {!isServerRunning ? (
                            <Button
                                variant="primary"
                                className="w-full"
                                onClick={handleStartServer}
                                isLoading={isStartingServer}
                                leftIcon={<Play size={16} />}
                                style={{ width: '100%', flexShrink: 0 }}
                            >
                                Start Server
                            </Button>
                        ) : (
                            <Button
                                variant="danger"
                                className="w-full"
                                onClick={handleStopServer}
                                leftIcon={<Square size={16} />}
                                style={{ width: '100%', flexShrink: 0 }}
                            >
                                Stop Server
                            </Button>
                        )}

                        {/* Scrollable Options Container */}
                        <div style={{
                            flex: 1,
                            overflowY: 'auto',
                            border: '1px solid rgba(255,255,255,0.1)',
                            borderRadius: '12px',
                            padding: '12px',
                            paddingBottom: '40px',
                            background: 'rgba(0,0,0,0.2)',
                            display: 'flex',
                            flexDirection: 'column',
                            gap: '16px'
                        }}>
                            <Card>
                                <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">Model Control</h3>

                                <div className="space-y-4" style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                                    <Select
                                        label="Base Model"
                                        options={[
                                            { value: '', label: 'Select Model' },
                                            ...availableGGUFModels.map(m => ({ value: m, label: m }))
                                        ]}
                                        value={selectedBaseModel}
                                        onChange={(e) => setSelectedBaseModel(e.target.value)}
                                        disabled={isServerRunning}
                                    />

                                    {userMode === 'power' && (
                                        <>
                                            <Select
                                                label="Training Project"
                                                options={[
                                                    { value: '', label: 'No LoRA' },
                                                    ...projectLoras.map(p => ({ value: p.project_name, label: p.project_name }))
                                                ]}
                                                value={selectedProject}
                                                onChange={(e) => {
                                                    setSelectedProject(e.target.value);
                                                    if (!e.target.value) {
                                                        setSelectedCheckpoint('');
                                                        setSelectedLoraAdapter('');
                                                    }
                                                }}
                                                disabled={isServerRunning}
                                            />

                                            {selectedProject && (
                                                <Select
                                                    label="Checkpoint"
                                                    options={
                                                        projectLoras
                                                            .find(p => p.project_name === selectedProject)
                                                            ?.checkpoints.map(c => ({
                                                                value: c.name,
                                                                label: c.is_final ? `${c.name} (Final)` : c.name
                                                            })) || []
                                                    }
                                                    value={selectedCheckpoint}
                                                    onChange={(e) => setSelectedCheckpoint(e.target.value)}
                                                    disabled={isServerRunning}
                                                />
                                            )}
                                            {(availableMmprojModels.length > 0 || selectedMmproj) && (
                                                <Select
                                                    label="MM Projector"
                                                    options={[
                                                        { value: '', label: 'None' },
                                                        ...availableMmprojModels.map(m => ({ value: m, label: m }))
                                                    ]}
                                                    value={selectedMmproj}
                                                    onChange={(e) => setSelectedMmproj(e.target.value)}
                                                    disabled={isServerRunning}
                                                />
                                            )}
                                        </>
                                    )}

                                    {userMode === 'user' && projectLoras.length > 0 && (
                                        <Select
                                            label="Fine-Tuned Model"
                                            options={[
                                                { value: '', label: 'None (Base Model Only)' },
                                                ...projectLoras.map(p => ({
                                                    value: p.project_name,
                                                    label: `${p.project_name} (auto-best)`
                                                }))
                                            ]}
                                            value={selectedProject}
                                            onChange={(e) => {
                                                setSelectedProject(e.target.value);
                                                if (!e.target.value) {
                                                    setSelectedLoraAdapter('');
                                                }
                                            }}
                                            disabled={isServerRunning}
                                        />
                                    )}
                                </div>
                            </Card>

                            <Card>
                                <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">Parameters</h3>
                                <div className="space-y-4" style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                                    <div>
                                        <div className="flex justify-between text-sm mb-1">
                                            <label>Temperature</label>
                                            <span className="text-gray-400">{temperature}</span>
                                        </div>
                                        <input
                                            type="range"
                                            min="0" max="2" step="0.1"
                                            value={temperature}
                                            onChange={(e) => setTemperature(Number(e.target.value))}
                                            className="w-full accent-accent-primary"
                                            style={{ width: '100%' }}
                                        />
                                    </div>

                                    {/* System Prompt - Now visible for all users */}
                                    <div className="space-y-2">
                                        <label className="text-sm font-medium">System Prompt</label>
                                        <textarea
                                            className="input-field w-full h-24 resize-none"
                                            value={systemPrompt}
                                            onChange={(e) => setSystemPrompt(e.target.value)}
                                            style={{ width: '100%', height: '100px', resize: 'vertical', background: 'rgba(0,0,0,0.2)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', padding: '12px', color: 'white' }}
                                            placeholder="You are a helpful assistant..."
                                        />
                                    </div>

                                    {userMode === 'power' && (
                                        <>
                                            <div>
                                                <div className="flex justify-between text-sm mb-1">
                                                    <label>Top P</label>
                                                    <span className="text-gray-400">{topP}</span>
                                                </div>
                                                <input
                                                    type="range"
                                                    min="0" max="1" step="0.05"
                                                    value={topP}
                                                    onChange={(e) => setTopP(Number(e.target.value))}
                                                    className="w-full accent-accent-primary"
                                                    style={{ width: '100%' }}
                                                />
                                            </div>

                                            <div>
                                                <div className="flex justify-between text-sm mb-1">
                                                    <label>Top K</label>
                                                    <span className="text-gray-400">{topK}</span>
                                                </div>
                                                <input
                                                    type="range"
                                                    min="0" max="100" step="1"
                                                    value={topK}
                                                    onChange={(e) => setTopK(Number(e.target.value))}
                                                    className="w-full accent-accent-primary"
                                                    style={{ width: '100%' }}
                                                />
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
                            </Card>
                        </div>
                    </div>
                )
            }
        </div >
    );
};

export default InferencePage;

