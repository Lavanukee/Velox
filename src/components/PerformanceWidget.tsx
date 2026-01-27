import React, { useState, useMemo, useCallback, useEffect } from 'react';

import { HardwareInfo } from '../types';
import {
    Activity,
    Zap,
    Cpu,
    Shield,
    Sparkles,
    AlertTriangle,
    HardDrive,
    Clock,
    Layers,
    Server,
    ChevronDown,
    Settings
} from 'lucide-react';
import {
    estimateResources,
    autoOptimize,
    ModelMetadata,
    TrainingConfig,
    HardwareBudget,
    ResourceEstimate,
    extractModelSizeFromPath,
    parseModelConfig
} from '../lib/ResourceBudgetEngine';
import { invoke } from '@tauri-apps/api/core';

interface PerformanceWidgetProps {
    // Estimation Data
    modelPath: string;
    numEpochs: number;
    batchSize: number;
    loraR: number;
    loraAlpha: number;
    maxSeqLength: number;
    totalSamples: number;
    gradientAccumulation: number;

    // State Controls
    useCpuOffload: boolean;
    setUseCpuOffload: (val: boolean) => void;
    usePagedOptimizer: boolean;
    setUsePagedOptimizer: (val: boolean) => void;
    useGradientCheckpointing: boolean;
    setUseGradientCheckpointing: (val: boolean) => void;
    hybridTraining: boolean;
    setHybridTraining: (val: boolean) => void;
    gpuLayers: number | 'auto';
    setGpuLayers: (val: number | 'auto') => void;
    offloadOptimizer: boolean;
    setOffloadOptimizer: (val: boolean) => void;
    useDeepspeed: boolean;
    setUseDeepspeed: (val: boolean) => void;

    // Batch size setter for auto-optimize
    setBatchSize?: (val: number) => void;
    setLoraR?: (val: number) => void;
    setLoraAlpha?: (val: number) => void;

    // Preset
    onPresetChange?: (preset: string) => void;

    hardwareInfo?: HardwareInfo;
}

export const PerformanceWidget: React.FC<PerformanceWidgetProps> = ({
    modelPath,
    numEpochs,
    batchSize,
    loraR,
    loraAlpha,
    maxSeqLength,
    totalSamples,
    gradientAccumulation,
    useCpuOffload,
    setUseCpuOffload,
    usePagedOptimizer,
    setUsePagedOptimizer,
    useGradientCheckpointing,
    setUseGradientCheckpointing,
    hybridTraining,
    setHybridTraining,
    gpuLayers,
    setGpuLayers,
    offloadOptimizer,
    setOffloadOptimizer,
    useDeepspeed,
    setUseDeepspeed,
    setBatchSize,
    setLoraR,
    setLoraAlpha,
    onPresetChange = () => { },
    hardwareInfo
}) => {
    const [selectedPreset, setSelectedPreset] = useState<string>('custom');
    const [showAdvanced, setShowAdvanced] = useState(false);

    // --- Budget State ---
    const detectedVram = hardwareInfo?.gpus?.[0]?.vram_total
        ? Math.round(hardwareInfo.gpus[0].vram_total / (1024 * 1024 * 1024))
        : 24;
    const detectedRam = hardwareInfo?.ram_total
        ? Math.round(hardwareInfo.ram_total / (1024 * 1024 * 1024))
        : 96;

    const [vramBudget, setVramBudget] = useState(detectedVram);
    const [ramBudget, setRamBudget] = useState(Math.round(detectedRam * 0.85));

    // --- Model Metadata State ---
    const [modelMeta, setModelMeta] = useState<ModelMetadata>({
        sizeB: 7,
        numLayers: 32,
        dModel: 4096,
        isVision: false
    });

    // Fetch model config when modelPath changes
    useEffect(() => {
        if (!modelPath) return;

        const fetchConfig = async () => {
            console.log(`[PerformanceWidget] Fetching config for: ${modelPath}`);
            try {
                const configJson = await invoke<Record<string, unknown>>('get_model_config_command', { modelPath });
                console.log(`[PerformanceWidget] Config received:`, configJson);
                const meta = parseModelConfig(configJson);
                console.log(`[PerformanceWidget] Parsed metadata:`, meta);
                setModelMeta(meta);
            } catch (err) {
                console.warn(`[PerformanceWidget] Config fetch failed, using path-based fallback:`, err);
                // Fallback to path-based extraction with better heuristics
                const sizeB = extractModelSizeFromPath(modelPath);
                // Better numLayers and dModel estimates based on common model architectures
                let numLayers = 32;
                let dModel = 4096;
                if (sizeB >= 70) { numLayers = 80; dModel = 8192; }
                else if (sizeB >= 30) { numLayers = 64; dModel = 6656; }
                else if (sizeB >= 13) { numLayers = 40; dModel = 5120; }
                else if (sizeB >= 7) { numLayers = 32; dModel = 4096; }
                else if (sizeB >= 3) { numLayers = 28; dModel = 3072; }
                else { numLayers = 24; dModel = 2048; }

                console.log(`[PerformanceWidget] Fallback estimate: ${sizeB}B, ${numLayers} layers`);
                setModelMeta({
                    sizeB,
                    numLayers,
                    dModel,
                    isVision: modelPath.toLowerCase().includes('vl') || modelPath.toLowerCase().includes('vision')
                });
            }
        };
        fetchConfig();
    }, [modelPath]);

    // --- Build TrainingConfig ---
    const trainingConfig: TrainingConfig = useMemo(() => ({
        numEpochs,
        batchSize,
        gradientAccumulationSteps: gradientAccumulation,
        maxSeqLength,
        loraR,
        loraAlpha,
        use4Bit: true,
        useGradientCheckpointing,
        usePagedOptimizer,
        useCpuOffload,
        hybridTraining,
        gpuLayers,
        offloadOptimizer,
        useDeepspeed,
        totalSamples
    }), [numEpochs, batchSize, gradientAccumulation, maxSeqLength, loraR, loraAlpha, useGradientCheckpointing, usePagedOptimizer, useCpuOffload, hybridTraining, gpuLayers, offloadOptimizer, useDeepspeed, totalSamples]);

    const budget: HardwareBudget = useMemo(() => ({
        vramGb: vramBudget,
        ramGb: ramBudget
    }), [vramBudget, ramBudget]);

    // --- Compute Estimate ---
    const estimate: ResourceEstimate = useMemo(() => {
        return estimateResources(modelMeta, trainingConfig, budget);
    }, [modelMeta, trainingConfig, budget]);

    const isOverVram = estimate.vramGb > vramBudget;
    const isOverRam = estimate.ramGb > ramBudget;

    // --- Auto-Optimize Handler ---
    const handleAutoOptimize = useCallback(() => {
        const result = autoOptimize(modelMeta, trainingConfig, budget);

        if (!result.success) {
            console.warn('Auto-optimize could not fit within budget. Changes applied:', result.changesApplied);
        }

        const optConfig = result.optimizedConfig;

        // Apply changes
        if (optConfig.useGradientCheckpointing !== useGradientCheckpointing) {
            setUseGradientCheckpointing(optConfig.useGradientCheckpointing);
        }
        if (optConfig.usePagedOptimizer !== usePagedOptimizer) {
            setUsePagedOptimizer(optConfig.usePagedOptimizer);
        }
        if (optConfig.batchSize !== batchSize && setBatchSize) {
            setBatchSize(optConfig.batchSize);
        }
        if (optConfig.hybridTraining !== hybridTraining) {
            setHybridTraining(optConfig.hybridTraining);
        }
        if (optConfig.gpuLayers !== gpuLayers) {
            setGpuLayers(optConfig.gpuLayers);
        }
        if (optConfig.offloadOptimizer !== offloadOptimizer) {
            setOffloadOptimizer(optConfig.offloadOptimizer);
        }
        if (optConfig.loraR !== loraR && setLoraR) {
            setLoraR(optConfig.loraR);
        }
        if (optConfig.loraAlpha !== loraAlpha && setLoraAlpha) {
            setLoraAlpha(optConfig.loraAlpha);
        }

        setSelectedPreset('custom');
    }, [modelMeta, trainingConfig, budget, useGradientCheckpointing, usePagedOptimizer, batchSize, hybridTraining, gpuLayers, offloadOptimizer, loraR, loraAlpha, setUseGradientCheckpointing, setUsePagedOptimizer, setBatchSize, setHybridTraining, setGpuLayers, setOffloadOptimizer, setLoraR, setLoraAlpha]);

    const handlePresetSelect = (preset: string) => {
        setSelectedPreset(preset);
        onPresetChange(preset);
    };

    const formatTime = (hours: number): string => {
        if (hours < 1) return `${Math.round(hours * 60)} min`;
        if (hours < 24) return `${hours.toFixed(1)} hrs`;
        const days = Math.floor(hours / 24);
        const rem = Math.round(hours % 24);
        return `${days}d ${rem}h`;
    };

    return (
        <div
            className="w-full relative overflow-hidden"
            style={{
                backgroundColor: '#0d0d12',
                color: 'white',
                isolation: 'isolate',
                borderRadius: '12px',
                border: '1px solid rgba(255, 255, 255, 0.05)',
                boxShadow: '0 4px 12px rgba(0, 0, 0, 0.2)'
            }}
        >
            {/* Background gradient removed to match window */}
            <div
                className="absolute inset-0"
                style={{ background: 'radial-gradient(ellipse at top right, rgba(30, 58, 138, 0.1), transparent, transparent)' }}
            />

            <div className="relative z-10 p-6">
                {/* Header */}
                <div className="flex items-center justify-between mb-12">
                    <div className="flex items-center gap-4">
                        <div
                            className="p-3 rounded-2xl border"
                            style={{
                                background: 'linear-gradient(to bottom right, rgba(59, 130, 246, 0.2), rgba(168, 85, 247, 0.2))',
                                borderColor: 'rgba(59, 130, 246, 0.3)',
                                boxShadow: '0 0 15px rgba(59, 130, 246, 0.1)'
                            }}
                        >
                            <Activity color="#60a5fa" size={24} />
                        </div>
                        <div>
                            <h3 className="text-xl font-bold tracking-tight" style={{ color: 'white' }}>Performance Center</h3>
                            <p className="text-sm" style={{ color: '#94a3b8' }}>Live resource monitoring & optimization</p>
                        </div>
                    </div>
                    <div
                        className="flex items-center gap-3 px-4 py-2 rounded-xl border"
                        style={{
                            background: 'rgba(30, 41, 59, 0.5)', /* slate-800/50 */
                            borderColor: 'rgba(51, 65, 85, 0.5)' /* slate-700/50 */
                        }}
                    >
                        <Server size={16} color="#60a5fa" />
                        <span className="text-sm font-semibold" style={{ color: '#cbd5e1' }}>
                            {modelMeta.sizeB}B
                        </span>
                        <span style={{ color: '#475569' }}>•</span>
                        <span className="text-sm" style={{ color: '#94a3b8' }}>
                            {modelMeta.numLayers} Layers
                        </span>
                    </div>
                </div>

                {/* Warnings */}
                {estimate.warnings.length > 0 && (
                    <div
                        className="mb-8 p-5 rounded-2xl border backdrop-blur-sm"
                        style={{
                            background: 'linear-gradient(to right, rgba(239, 68, 68, 0.1), rgba(249, 115, 22, 0.1))',
                            borderColor: 'rgba(239, 68, 68, 0.3)'
                        }}
                    >
                        {estimate.warnings.map((w, i) => (
                            <div key={i} className="flex items-start gap-3 text-sm">
                                <AlertTriangle size={18} color="#ff8400ff" className="mt-0.5 flex-shrink-0" />
                                <span className="font-medium" style={{ color: '#fca5a5' }}>{w}</span>
                            </div>
                        ))}
                    </div>
                )}

                {/* Resource Estimates - Cards */}
                <div
                    style={{
                        display: 'grid',
                        gridTemplateColumns: 'repeat(4, 1fr)',
                        gap: '12px',
                        marginBottom: '40px'
                    }}
                >
                    {/* VRAM */}
                    <div
                        style={{
                            background: isOverVram
                                ? 'linear-gradient(to bottom right, rgba(239, 68, 68, 0.15), rgba(220, 38, 38, 0.05))'
                                : 'rgba(255, 255, 255, 0.03)',
                            border: isOverVram ? '1px solid rgba(239, 68, 68, 0.3)' : '1px solid rgba(255, 255, 255, 0.05)',
                            borderRadius: '12px',
                            padding: '16px',
                            position: 'relative',
                            overflow: 'hidden'
                        }}
                    >
                        <div className="relative z-10">
                            <div className="flex items-center gap-2 mb-3">
                                <Cpu size={18} color={isOverVram ? '#f87171' : '#60a5fa'} />
                                <span className="text-xs font-bold uppercase tracking-wider" style={{ color: '#cbd5e1' }}>VRAM</span>
                            </div>
                            <div className="flex items-baseline gap-2">
                                <span className="text-3xl font-bold" style={{ color: isOverVram ? '#f87171' : 'white' }}>
                                    {estimate.vramGb}
                                </span>
                                <span className="text-sm font-medium" style={{ color: '#64748b' }}>GB</span>
                            </div>
                            <div className="mt-2 h-1.5 w-full rounded-full bg-slate-700/50 overflow-hidden">
                                <div
                                    className="h-full rounded-full transition-all duration-500"
                                    style={{
                                        width: `${Math.min(100, (estimate.vramGb / vramBudget) * 100)}%`,
                                        background: isOverVram ? '#f87171' : '#3b82f6'
                                    }}
                                />
                            </div>
                            <div className="mt-2 text-right text-xs" style={{ color: '#94a3b8' }}>
                                Limit: {vramBudget} GB
                            </div>
                        </div>
                    </div>

                    {/* RAM */}
                    <div
                        style={{
                            background: isOverRam
                                ? 'linear-gradient(to bottom right, rgba(239, 68, 68, 0.15), rgba(220, 38, 38, 0.05))'
                                : 'rgba(255, 255, 255, 0.03)',
                            border: isOverRam ? '1px solid rgba(239, 68, 68, 0.3)' : '1px solid rgba(255, 255, 255, 0.05)',
                            borderRadius: '12px',
                            padding: '16px',
                            position: 'relative',
                            overflow: 'hidden'
                        }}
                    >
                        <div className="relative z-10">
                            <div className="flex items-center gap-2 mb-3">
                                <HardDrive size={18} color={isOverRam ? '#f87171' : '#4ade80'} />
                                <span className="text-xs font-bold uppercase tracking-wider" style={{ color: '#cbd5e1' }}>RAM</span>
                            </div>
                            <div className="flex items-baseline gap-2">
                                <span className="text-3xl font-bold" style={{ color: isOverRam ? '#f87171' : 'white' }}>
                                    {estimate.ramGb}
                                </span>
                                <span className="text-sm font-medium" style={{ color: '#64748b' }}>GB</span>
                            </div>
                            <div className="mt-2 h-1.5 w-full rounded-full bg-slate-700/50 overflow-hidden">
                                <div
                                    className="h-full rounded-full transition-all duration-500"
                                    style={{
                                        width: `${Math.min(100, (estimate.ramGb / ramBudget) * 100)}%`,
                                        background: isOverRam ? '#f87171' : '#22c55e'
                                    }}
                                />
                            </div>
                            <div className="mt-2 text-right text-xs" style={{ color: '#94a3b8' }}>
                                Limit: {ramBudget} GB
                            </div>
                        </div>
                    </div>

                    {/* Time */}
                    <div
                        style={{
                            background: 'rgba(255, 255, 255, 0.03)',
                            border: '1px solid rgba(255, 255, 255, 0.05)',
                            borderRadius: '12px',
                            padding: '16px',
                            position: 'relative',
                            overflow: 'hidden'
                        }}
                    >
                        <div className="relative">
                            <div className="flex items-center gap-2 mb-3">
                                <Clock size={18} color="#fbbf24" />
                                <span className="text-xs font-bold uppercase tracking-wider" style={{ color: '#cbd5e1' }}>Duration</span>
                            </div>
                            <div className="text-3xl font-bold text-white mb-2" style={{ color: 'white' }}>
                                {formatTime(estimate.timeHours)}
                            </div>
                            <div className="text-xs" style={{ color: '#64748b' }}>Estimated training time</div>
                        </div>
                    </div>

                    {/* Speed */}
                    <div
                        style={{
                            background: 'rgba(255, 255, 255, 0.03)',
                            border: '1px solid rgba(255, 255, 255, 0.05)',
                            borderRadius: '12px',
                            padding: '16px',
                            position: 'relative',
                            overflow: 'hidden'
                        }}
                    >
                        <div className="relative">
                            <div className="flex items-center gap-2 mb-3">
                                <Zap size={18} color="#c084fc" />
                                <span className="text-xs font-bold uppercase tracking-wider" style={{ color: '#cbd5e1' }}>Speed</span>
                            </div>
                            <div className="flex items-baseline gap-2 mb-2">
                                <span className="text-3xl font-bold text-white" style={{ color: 'white' }}>
                                    {estimate.tokensPerSecond}
                                </span>
                                <span className="text-sm font-medium" style={{ color: '#64748b' }}>tk/s</span>
                            </div>
                            <div className="text-xs" style={{ color: '#64748b' }}>Throughput</div>
                        </div>
                    </div>
                </div>

                <div
                    className="mb-8 p-6 border"
                    style={{
                        background: 'rgba(255, 255, 255, 0.02)',
                        borderColor: 'rgba(255, 255, 255, 0.05)',
                        borderRadius: '12px'
                    }}
                >
                    <div className="flex items-center justify-between mb-10">
                        <div className="flex items-center gap-3">
                            <div className="p-2 rounded-xl" style={{ background: 'rgba(59, 130, 246, 0.1)' }}>
                                <Sparkles size={18} color="#60a5fa" />
                            </div>
                            <div>
                                <h4 className="font-bold text-white">Resource Budget</h4>
                                <p className="text-xs text-slate-400">Set your hardware limits</p>
                            </div>
                        </div>
                        <button
                            onClick={handleAutoOptimize}
                            style={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: '8px',
                                padding: '10px 20px',
                                borderRadius: '12px',
                                fontSize: '14px',
                                fontWeight: 'bold',
                                background: 'linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%)',
                                color: '#ffffff',
                                border: '1px solid rgba(255, 255, 255, 0.1)',
                                boxShadow: '0 4px 10px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.1)',
                                cursor: 'pointer',
                                transition: 'all 0.2s',
                                outline: 'none'
                            }}
                            onMouseDown={(e) => e.currentTarget.style.transform = 'scale(0.95)'}
                            onMouseUp={(e) => e.currentTarget.style.transform = 'scale(1)'}
                            onMouseLeave={(e) => e.currentTarget.style.transform = 'scale(1)'}
                        >
                            <Sparkles size={16} className="animate-pulse" />
                            <span>Auto-Optimize</span>
                        </button>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                        {/* VRAM Slider */}
                        <div className="p-4" style={{ background: 'rgba(255, 255, 255, 0.02)', border: '1px solid rgba(255, 255, 255, 0.05)', borderRadius: '12px' }}>
                            <div className="flex justify-between items-center mb-4">
                                <div className="flex items-center gap-2">
                                    <Cpu size={16} color="#60a5fa" />
                                    <span className="text-sm font-semibold" style={{ color: '#e2e8f0' }}>VRAM Limit</span>
                                </div>
                                <span className="text-lg font-bold" style={{ color: isOverVram ? '#f87171' : '#60a5fa' }}>
                                    {vramBudget} <span className="text-sm" style={{ opacity: 0.7 }}>GB</span>
                                </span>
                            </div>
                            <div className="relative px-2">
                                <input
                                    type="range"
                                    min={4}
                                    max={detectedVram}
                                    value={vramBudget}
                                    onChange={(e) => setVramBudget(parseInt(e.target.value))}
                                    className="w-full h-3 rounded-full appearance-none cursor-pointer outline-none block"
                                    style={{
                                        background: `linear-gradient(to right, rgb(59, 130, 246) 0%, rgb(59, 130, 246) ${((vramBudget - 4) / (detectedVram - 4)) * 100}%, rgb(30, 41, 59) ${((vramBudget - 4) / (detectedVram - 4)) * 100}%, rgb(30, 41, 59) 100%)`
                                    }}
                                />
                                <style>{`
                                    input[type="range"]::-webkit-slider-thumb {
                                        appearance: none;
                                        width: 24px;
                                        height: 24px;
                                        border-radius: 50%;
                                        background: white;
                                        cursor: pointer;
                                        border: 4px solid #3b82f6;
                                        box-shadow: 0 0 15px rgba(59, 130, 246, 0.4);
                                        margin-top: -2px; /* Center thumb */
                                    }
                                    input[type="range"]::-moz-range-thumb {
                                        width: 24px;
                                        height: 24px;
                                        border-radius: 50%;
                                        background: white;
                                        cursor: pointer;
                                        border: 4px solid #3b82f6;
                                        box-shadow: 0 0 15px rgba(59, 130, 246, 0.4);
                                    }
                                `}</style>
                            </div>
                            <div className="flex justify-between mt-3 text-xs font-mono font-medium" style={{ color: '#64748b' }}>
                                <span>4 GB</span>
                                <span>{detectedVram} GB</span>
                            </div>
                        </div>

                        {/* RAM Slider */}
                        <div className="p-4" style={{ background: 'rgba(255, 255, 255, 0.02)', border: '1px solid rgba(255, 255, 255, 0.05)', borderRadius: '12px' }}>
                            <div className="flex justify-between items-center mb-4">
                                <div className="flex items-center gap-2">
                                    <HardDrive size={16} color="#4ade80" />
                                    <span className="text-sm font-semibold" style={{ color: '#e2e8f0' }}>RAM Limit</span>
                                </div>
                                <span className="text-lg font-bold" style={{ color: isOverRam ? '#f87171' : '#4ade80' }}>
                                    {ramBudget} <span className="text-sm" style={{ opacity: 0.7 }}>GB</span>
                                </span>
                            </div>
                            <div className="relative px-2">
                                <input
                                    type="range"
                                    min={8}
                                    max={detectedRam}
                                    value={ramBudget}
                                    onChange={(e) => setRamBudget(parseInt(e.target.value))}
                                    className="w-full h-3 rounded-full appearance-none cursor-pointer outline-none block"
                                    style={{
                                        background: `linear-gradient(to right, rgb(34, 197, 94) 0%, rgb(34, 197, 94) ${((ramBudget - 8) / (detectedRam - 8)) * 100}%, rgb(30, 41, 59) ${((ramBudget - 8) / (detectedRam - 8)) * 100}%, rgb(30, 41, 59) 100%)`
                                    }}
                                />
                                <style>{`
                                    input[type="range"]::-webkit-slider-thumb {
                                        appearance: none;
                                        width: 24px;
                                        height: 24px;
                                        border-radius: 50%;
                                        background: white;
                                        cursor: pointer;
                                        border: 4px solid #22c55e;
                                        box-shadow: 0 0 15px rgba(34, 197, 94, 0.4);
                                        margin-top: -2px;
                                    }
                                    input[type="range"]::-moz-range-thumb {
                                        width: 24px;
                                        height: 24px;
                                        border-radius: 50%;
                                        background: white;
                                        cursor: pointer;
                                        border: 4px solid #22c55e;
                                        box-shadow: 0 0 15px rgba(34, 197, 94, 0.4);
                                    }
                                `}</style>
                            </div>
                            <div className="flex justify-between mt-3 text-xs font-mono font-medium" style={{ color: '#64748b' }}>
                                <span>8 GB</span>
                                <span>{detectedRam} GB</span>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Optimization Strategy */}
                <div className="mb-6">
                    <div className="flex items-center gap-3 mb-10">
                        <div className="p-2 rounded-xl" style={{ background: 'rgba(168, 85, 247, 0.1)' }}>
                            <Shield size={18} color="#c084fc" />
                        </div>
                        <div>
                            <h4 className="font-bold text-white">Optimization Strategy</h4>
                            <p className="text-xs text-slate-400">Choose your tuning profile</p>
                        </div>
                    </div>
                    {/* Fixed Grid Layout: Force 4 columns with inline styles */}
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px' }}>
                        {[
                            { id: 'standard', label: 'Standard', icon: Shield, color: 'blue', colorHex: '#3b82f6', desc: 'Balanced' },
                            { id: 'hybrid', label: 'Hybrid', icon: Cpu, color: 'purple', colorHex: '#a855f7', desc: 'GPU + CPU' },
                            { id: 'max_speed', label: 'Speed', icon: Zap, color: 'amber', colorHex: '#f59e0b', desc: 'Fastest' },
                            { id: 'max_memory', label: 'Eco', icon: Layers, color: 'green', colorHex: '#22c55e', desc: 'Low VRAM' }
                        ].map((preset) => {
                            const Icon = preset.icon;
                            // @ts-ignore
                            const active = selectedPreset === preset.id;
                            // @ts-ignore
                            const color = preset.color;
                            // @ts-ignore
                            const colorHex = preset.colorHex;

                            // Determine active colors
                            const activeBg = `rgba(${color === 'blue' ? '59, 130, 246' : color === 'purple' ? '168, 85, 247' : color === 'amber' ? '245, 158, 11' : '34, 197, 94'}, 0.15)`;
                            const activeBorder = `rgba(${color === 'blue' ? '59, 130, 246' : color === 'purple' ? '168, 85, 247' : color === 'amber' ? '245, 158, 11' : '34, 197, 94'}, 0.5)`;
                            const glow = `0 0 20px -5px rgba(${color === 'blue' ? '59, 130, 246' : color === 'purple' ? '168, 85, 247' : color === 'amber' ? '245, 158, 11' : '34, 197, 94'}, 0.3)`;

                            return (
                                <button
                                    key={preset.id}
                                    onClick={() => handlePresetSelect(preset.id)}
                                    className="group relative p-4 border transition-all duration-300 flex flex-col items-center justify-center gap-3 text-center h-full"
                                    style={{
                                        background: active ? activeBg : 'rgba(30, 41, 59, 0.3)',
                                        borderColor: active ? activeBorder : 'rgba(51, 65, 85, 0.4)',
                                        boxShadow: active ? glow : 'none',
                                        minHeight: '120px',
                                        borderRadius: '12px'
                                    }}
                                >
                                    {active && (
                                        <div className="absolute top-3 right-3 w-2.5 h-2.5 rounded-full bg-white shadow-sm animate-pulse"
                                            style={{ backgroundColor: colorHex, boxShadow: `0 0 8px ${colorHex}` }}
                                        />
                                    )}

                                    <div
                                        className="p-3 transition-colors"
                                        style={{ background: 'transparent' }}
                                    >
                                        <Icon size={24} color={active ? colorHex : '#94a3b8'} />
                                    </div>

                                    <div>
                                        <div className="text-sm font-bold mb-1" style={{ color: active ? 'white' : '#94a3b8' }}>
                                            {preset.label}
                                        </div>
                                        {/* @ts-ignore */}
                                        <div className="text-xs font-medium" style={{ color: active ? '#cbd5e1' : '#64748b' }}>{preset.desc}</div>
                                    </div>
                                </button>
                            );
                        })}
                    </div>
                </div>

                {/* Advanced Settings */}
                <div>
                    <button
                        onClick={() => setShowAdvanced(!showAdvanced)}
                        className="flex items-center justify-between w-full p-4 transition-all duration-200 mb-3"
                        style={{
                            background: 'rgba(30, 41, 59, 0.3)',
                            border: '1px solid rgba(51, 65, 85, 0.5)',
                            borderRadius: '12px'
                        }}
                    >
                        <div className="flex items-center gap-3">
                            <Settings size={18} color="#94a3b8" />
                            <span className="text-sm font-bold" style={{ color: 'white' }}>Advanced Settings</span>
                        </div>
                        <ChevronDown size={18} className={`transition-transform duration-200 ${showAdvanced ? 'rotate-180' : ''}`} color="#94a3b8" />
                    </button>

                    {showAdvanced && (
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 animate-in fade-in slide-in-from-top-2 duration-300">
                            {[
                                {
                                    label: 'Gradient Checkpointing',
                                    desc: 'Save VRAM (slower)',
                                    active: useGradientCheckpointing,
                                    onClick: () => setUseGradientCheckpointing(!useGradientCheckpointing),
                                    icon: Layers
                                },
                                {
                                    label: 'Paged Optimizer',
                                    desc: 'Offload to RAM',
                                    active: usePagedOptimizer,
                                    onClick: () => setUsePagedOptimizer(!usePagedOptimizer),
                                    icon: HardDrive
                                },
                                {
                                    label: 'Hybrid Training',
                                    desc: 'Use CPU + GPU',
                                    active: hybridTraining,
                                    onClick: () => {
                                        setHybridTraining(!hybridTraining);
                                        if (!hybridTraining) setUseCpuOffload(false);
                                    },
                                    icon: Cpu
                                },
                                {
                                    label: 'DeepSpeed ZeRO-2',
                                    desc: 'Distributed Optim.',
                                    active: useDeepspeed,
                                    onClick: () => setUseDeepspeed(!useDeepspeed),
                                    icon: Zap,
                                    color: 'purple'
                                }
                            ].map((toggle, idx) => {
                                const Icon = toggle.icon;
                                const isPurple = toggle.color === 'purple';
                                const activeBg = isPurple
                                    ? 'linear-gradient(to right, rgba(168, 85, 247, 0.2), rgba(147, 51, 234, 0.1))'
                                    : 'linear-gradient(to right, rgba(59, 130, 246, 0.2), rgba(37, 99, 235, 0.1))';
                                const activeBorder = isPurple ? 'rgba(168, 85, 247, 0.5)' : 'rgba(59, 130, 246, 0.5)';

                                return (
                                    <div
                                        key={idx}
                                        onClick={toggle.onClick}
                                        className="group relative p-4 border transition-all duration-200 cursor-pointer flex items-center justify-between"
                                        style={{
                                            background: toggle.active ? activeBg : 'rgba(15, 23, 42, 0.4)', // Darker inactive bg
                                            borderColor: toggle.active ? activeBorder : 'rgba(51, 65, 85, 0.4)',
                                            borderRadius: '12px'
                                        }}
                                    >
                                        <div className="flex items-center gap-4">
                                            <div
                                                className="p-2.5 rounded-lg"
                                                style={{ background: toggle.active ? (isPurple ? 'rgba(168, 85, 247, 0.2)' : 'rgba(59, 130, 246, 0.2)') : 'rgba(51, 65, 85, 0.3)' }}
                                            >
                                                <Icon size={18} color={toggle.active ? (isPurple ? '#c084fc' : '#60a5fa') : '#94a3b8'} />
                                            </div>
                                            <div>
                                                <div className="text-sm font-bold" style={{ color: toggle.active ? 'white' : '#cbd5e1' }}>
                                                    {toggle.label}
                                                </div>
                                                <p className="text-xs" style={{ color: '#64748b' }}>{toggle.desc}</p>
                                            </div>
                                        </div>

                                        <div
                                            className="relative w-12 h-7 rounded-full transition-all duration-200 border"
                                            style={{
                                                background: toggle.active ? (isPurple ? '#a855f7' : '#3b82f6') : '#1e293b',
                                                borderColor: toggle.active ? (isPurple ? '#a855f7' : '#3b82f6') : '#475569'
                                            }}
                                        >
                                            <div
                                                className={`absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full transition-all duration-200 shadow-sm ${toggle.active ? 'translate-x-5' : 'translate-x-0'}`}
                                            />
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};