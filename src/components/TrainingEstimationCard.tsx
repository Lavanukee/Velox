import React, { useMemo } from 'react';
import { Card } from './Card';
import { Activity, Zap, Clock, Cpu, HardDrive, CheckCircle2 } from 'lucide-react';
import { HardwareInfo } from '../types';

interface TrainingEstimationCardProps {
    modelPath: string;
    numEpochs: number;
    batchSize: number;
    loraR: number;
    loraAlpha: number;
    maxSeqLength: number;
    totalSamples: number;
    use4Bit?: boolean; // QLoRA 4-bit quantization
    gradientAccumulation?: number;
    hardwareInfo?: HardwareInfo;
    useCpuOffload?: boolean;
    usePagedOptimizer?: boolean;
    useGradientCheckpointing?: boolean;
    hybridTraining?: boolean;
    gpuLayers?: number | 'auto';
    offloadOptimizer?: boolean;
    useDeepspeed?: boolean;
}

interface ResourceEstimate {
    vramGb: number;
    ramGb: number;
    timeHours: number;
    powerKwh: number;
}

// Extract model size from path (e.g., "Qwen3-VL-30B" -> 30)
function extractModelSizeB(modelPath: string): number {
    const name = modelPath.toLowerCase();

    // Match patterns like "30b", "3.7b", "72b", etc.
    const match = name.match(/(\d+\.?\d*)b/);
    if (match) {
        return parseFloat(match[1]);
    }

    // Common model size keywords
    if (name.includes('tiny') || name.includes('mini')) return 0.5;
    if (name.includes('small')) return 1;
    if (name.includes('base')) return 3;
    if (name.includes('large')) return 7;
    if (name.includes('xl')) return 13;
    if (name.includes('xxl')) return 70;

    return 7; // Default to 7B
}

export const TrainingEstimationCard: React.FC<TrainingEstimationCardProps> = ({
    modelPath,
    numEpochs,
    batchSize,
    loraR,
    maxSeqLength,
    totalSamples,
    use4Bit = true, // Default to QLoRA
    gradientAccumulation = 4,
    hardwareInfo,
    useCpuOffload = false,
    usePagedOptimizer = true,
    useGradientCheckpointing = true,
    hybridTraining = false,
    gpuLayers = 'auto',
    offloadOptimizer = false,
    useDeepspeed = false
}) => {
    const modelSizeB = extractModelSizeB(modelPath);

    const estimate = useMemo((): ResourceEstimate => {
        const modelSizeB = extractModelSizeB(modelPath);

        // ============================================================
        // VRAM ESTIMATION (More realistic formulas)
        // ============================================================

        // 1. Model weights VRAM
        const bitsPerParam = use4Bit ? 4 : 16;
        const totalWeightsGb = (modelSizeB * 1e9 * bitsPerParam) / (8 * 1024 * 1024 * 1024);
        let modelWeightsGb = totalWeightsGb;
        let weightsInRamGb = 0;

        if (hybridTraining) {
            let totalLayers = 32;
            if (modelSizeB > 60) totalLayers = 80;
            else if (modelSizeB > 25) totalLayers = 60;
            else if (modelSizeB > 10) totalLayers = 40;

            const actualGpuLayers = gpuLayers === 'auto'
                ? Math.min(totalLayers, modelSizeB > 15 ? 20 : 32)
                : gpuLayers;

            const gpuRatio = Math.min(1.0, Math.max(0, actualGpuLayers / totalLayers));
            modelWeightsGb = totalWeightsGb * gpuRatio;
            weightsInRamGb = totalWeightsGb * (1 - gpuRatio);
        } else if (useCpuOffload) {
            modelWeightsGb = totalWeightsGb * 0.2;
            weightsInRamGb = totalWeightsGb * 0.8;
        }

        // 2. LoRA adapters VRAM
        // Heuristic: r/d_model ratio. d_model approx 4096 for 7B.
        // LoRA size ~= modelSize * (r / 2048) * (target_modules_factor around 0.5)
        const loraVram = modelSizeB * (loraR / 4096) * 1.5; // Safety margin included

        // 3. Optimizer states (AdamW 8-bit)
        let optimizerVram = 0;
        let optimizerRam = 0;
        const trainableParamRatio = (loraR / 4096) * 2; // Rough approx of active params
        const totalOptimizerGb = modelSizeB * trainableParamRatio * 0.5; // Optimizer states are small for LoRA

        if (offloadOptimizer || usePagedOptimizer) {
            optimizerRam = totalOptimizerGb;
            optimizerVram = 0;
        } else {
            optimizerVram = totalOptimizerGb;
        }

        // 4. Activation memory
        let activationPerSample = (maxSeqLength * (modelSizeB / 7)) / 8000; // tuned constant
        if (useGradientCheckpointing) {
            activationPerSample = activationPerSample * 0.15;
        }
        const activationVram = batchSize * activationPerSample;

        // 5. Gradient memory
        const gradientVram = totalOptimizerGb; // Gradients similar size to optimizer states

        // Total VRAM with safety margin
        const totalVram = modelWeightsGb + loraVram + optimizerVram + activationVram + gradientVram + 0.5; // +0.5GB context overhead
        const vramWithMargin = totalVram * 1.05;

        // ============================================================
        // RAM ESTIMATION
        // ============================================================
        // Dataset RAM: Don't load entire dataset. Just buffer.
        // 4 workers * batchSize * Context * 4 bytes
        const datasetRamBuffer = (4 * batchSize * maxSeqLength * 4) / (1024 * 1024 * 1024);
        let systemRam = Math.max(4, datasetRamBuffer + 2); // Base OS + Python overhead

        if (weightsInRamGb > 0) systemRam += weightsInRamGb;
        if (optimizerRam > 0) systemRam += optimizerRam;

        // Base model loading overhead in RAM before moving to GPU
        // Usually models load to RAM first then GPU. But with 'device_map=auto' it might stream.
        // We assume 1x model size spike during loading if not careful, but steady state is lower.
        // Let's model steady state for now.
        if (useCpuOffload || hybridTraining) {
            systemRam += 2; // Extra overhead for offloading management
        }

        if (useDeepspeed) systemRam += 4; // Extra overhead

        // ============================================================
        // TIME ESTIMATION
        // ============================================================
        let tokensPerSecond: number;
        if (modelSizeB <= 3) tokensPerSecond = 1200;
        else if (modelSizeB <= 8) tokensPerSecond = 800;
        else if (modelSizeB <= 15) tokensPerSecond = 400;
        else if (modelSizeB <= 35) tokensPerSecond = 150;
        else tokensPerSecond = 50;

        // Penalties
        if (hybridTraining) tokensPerSecond *= 0.6; // Moderate slowdown
        else if (useCpuOffload) tokensPerSecond *= 0.15; // Major slowdown

        if (useGradientCheckpointing) tokensPerSecond *= 0.8;
        if (useDeepspeed) tokensPerSecond *= 0.9;

        const totalTokens = totalSamples * numEpochs * maxSeqLength;
        const timeSeconds = totalTokens / tokensPerSecond;
        const timeHours = timeSeconds / 3600;

        // ============================================================
        // POWER ESTIMATION
        // ============================================================
        const gpuTdp = 450;
        const utilization = (useCpuOffload || hybridTraining) ? 0.7 : 0.9;
        const powerKwh = (gpuTdp * utilization * timeHours) / 1000;

        return {
            vramGb: Math.round(vramWithMargin * 10) / 10,
            ramGb: Math.round(systemRam),
            timeHours: Math.round(timeHours * 100) / 100,
            powerKwh: Math.round(powerKwh * 10) / 10
        };
    }, [modelPath, numEpochs, batchSize, loraR, maxSeqLength, totalSamples, use4Bit, gradientAccumulation, useCpuOffload, usePagedOptimizer, useGradientCheckpointing, hybridTraining, gpuLayers, offloadOptimizer, useDeepspeed]);

    const formatTime = (hours: number): string => {
        if (hours < 1) {
            const minutes = Math.round(hours * 60);
            return `${minutes} min`;
        } else if (hours < 24) {
            return `${hours.toFixed(1)} hrs`;
        } else {
            const days = Math.floor(hours / 24);
            const remainingHours = Math.round(hours % 24);
            return `${days}d ${remainingHours}h`;
        }
    };

    // Determine detected VRAM (prefer primary/highest VRAM GPU)
    const detectedVram = hardwareInfo?.gpus?.[0]?.vram_total
        ? hardwareInfo.gpus[0].vram_total / (1024 * 1024 * 1024)
        : 24; // Default to 24 if unknown

    const detectedGpuName = hardwareInfo?.gpus?.[0]?.name || "Generic Nvidia GPU";

    const isOverVram = estimate.vramGb > detectedVram;

    return (
        <Card style={{
            background: 'linear-gradient(135deg, rgba(20, 20, 26, 0.9) 0%, rgba(25, 25, 32, 0.9) 100%)',
            border: '1px solid rgba(99, 102, 241, 0.2)',
            boxShadow: '0 4px 20px rgba(99, 102, 241, 0.1)'
        }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <Activity size={20} style={{ color: '#818cf8' }} />
                    <h3 style={{ fontSize: '16px', fontWeight: 600, margin: 0 }}>Training Estimates</h3>
                </div>
                <div style={{
                    fontSize: '11px',
                    color: '#60a5fa',
                    background: 'rgba(59, 130, 246, 0.15)',
                    padding: '4px 8px',
                    borderRadius: '4px'
                }}>
                    {modelSizeB}B model • {use4Bit ? '4-bit' : 'FP16'}
                </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                {/* VRAM */}
                <div style={{
                    padding: '12px',
                    borderRadius: '8px',
                    background: isOverVram ? 'rgba(239, 68, 68, 0.15)' : 'rgba(99, 102, 241, 0.1)',
                    border: `1px solid ${isOverVram ? 'rgba(239, 68, 68, 0.3)' : 'rgba(99, 102, 241, 0.2)'}`
                }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                        <Cpu size={14} style={{ color: isOverVram ? '#f87171' : '#818cf8' }} />
                        <span style={{ fontSize: '11px', color: '#9ca3af' }}>VRAM</span>
                    </div>
                    <div style={{ fontSize: '22px', fontWeight: 700, color: isOverVram ? '#f87171' : '#e5e7eb' }}>
                        {estimate.vramGb} <span style={{ fontSize: '12px', color: '#9ca3af' }}>GB / {detectedVram.toFixed(0)} GB</span>
                    </div>
                    {isOverVram && (
                        <div style={{ fontSize: '9px', color: '#f87171', marginTop: '4px' }}>
                            ⚠️ Exceeds available VRAM
                        </div>
                    )}
                </div>

                {/* RAM */}
                <div style={{
                    padding: '12px',
                    borderRadius: '8px',
                    background: 'rgba(52, 211, 153, 0.1)',
                    border: '1px solid rgba(52, 211, 153, 0.2)'
                }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                        <HardDrive size={14} style={{ color: '#34d399' }} />
                        <span style={{ fontSize: '11px', color: '#9ca3af' }}>RAM</span>
                    </div>
                    <div style={{ fontSize: '22px', fontWeight: 700, color: '#e5e7eb' }}>
                        {estimate.ramGb} <span style={{ fontSize: '12px', color: '#9ca3af' }}>GB</span>
                    </div>
                </div>

                {/* Time */}
                <div style={{
                    padding: '12px',
                    borderRadius: '8px',
                    background: 'rgba(251, 191, 36, 0.1)',
                    border: '1px solid rgba(251, 191, 36, 0.2)'
                }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                        <Clock size={14} style={{ color: '#fbbf24' }} />
                        <span style={{ fontSize: '11px', color: '#9ca3af' }}>Time</span>
                    </div>
                    <div style={{ fontSize: '22px', fontWeight: 700, color: '#e5e7eb' }}>
                        {formatTime(estimate.timeHours)}
                    </div>
                </div>

                {/* Power */}
                <div style={{
                    padding: '12px',
                    borderRadius: '8px',
                    background: 'rgba(236, 72, 153, 0.1)',
                    border: '1px solid rgba(236, 72, 153, 0.2)'
                }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                        <Zap size={14} style={{ color: '#f472b6' }} />
                        <span style={{ fontSize: '11px', color: '#9ca3af' }}>Power</span>
                    </div>
                    <div style={{ fontSize: '22px', fontWeight: 700, color: '#e5e7eb' }}>
                        {estimate.powerKwh} <span style={{ fontSize: '12px', color: '#9ca3af' }}>kWh</span>
                    </div>
                </div>
            </div>

            <div style={{
                marginTop: '10px',
                padding: '6px 10px',
                background: 'rgba(255, 255, 255, 0.03)',
                borderRadius: '4px',
                fontSize: '10px',
                color: '#6b7280',
                display: 'flex',
                alignItems: 'center',
                gap: '6px'
            }}>
                <CheckCircle2 size={12} />
                <span>Estimates based on {detectedGpuName}</span>
            </div>
        </Card>
    );
};


