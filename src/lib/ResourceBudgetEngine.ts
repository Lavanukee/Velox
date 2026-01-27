/**
 * ResourceBudgetEngine.ts
 *
 * Core utility for estimating training resource requirements (VRAM, RAM, Time)
 * and auto-optimizing configurations to fit within a user-defined hardware budget.
 */

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

export interface HardwareBudget {
    vramGb: number;       // Available VRAM in GB (e.g., 24)
    ramGb: number;        // Available RAM in GB (e.g., 85)
    gpuModel?: string;    // For baseline training speed lookup
}

export interface ModelMetadata {
    sizeB: number;          // Estimated model size in billions of parameters
    numLayers: number;      // Number of hidden layers
    dModel: number;         // Hidden dimension (e.g., 4096 for 7B)
    isVision?: boolean;     // VLM models have vision encoder overhead
}

export interface TrainingConfig {
    numEpochs: number;
    batchSize: number;
    gradientAccumulationSteps: number;
    maxSeqLength: number;
    loraR: number;
    loraAlpha: number;
    use4Bit: boolean;
    useGradientCheckpointing: boolean;
    usePagedOptimizer: boolean;
    useCpuOffload: boolean;       // Full CPU offload (very slow)
    hybridTraining: boolean;      // Split layers between GPU and RAM
    gpuLayers: number | 'auto';   // Number of layers on GPU (for hybrid)
    offloadOptimizer: boolean;    // Offload AdamW states to RAM
    useDeepspeed: boolean;
    totalSamples: number;
}

export interface MemoryBreakdown {
    modelWeightsVram: number;
    loraVram: number;
    activationsVram: number;
    gradientVram: number;
    optimizerVram: number;
    // RAM
    modelWeightsRam: number;
    optimizerRam: number;
    systemOverhead: number;
}

export interface ResourceEstimate {
    vramGb: number;
    ramGb: number;
    timeHours: number;
    tokensPerSecond: number;
    warnings: string[];
    breakdown: MemoryBreakdown;
}

export interface OptimizationResult {
    optimizedConfig: TrainingConfig;
    estimate: ResourceEstimate;
    success: boolean;
    changesApplied: string[];
}

// ============================================================================
// CONSTANTS & BASELINES
// ============================================================================

// Baseline tokens/second for different model sizes on RTX 4090 (24GB, QLoRA, GC on)
// These are rough estimates and should be calibrated with real data.
const BASELINE_TOKENS_PER_SECOND: { [key: string]: number } = {
    '1b': 2500,
    '3b': 1500,
    '7b': 800,
    '8b': 700,
    '13b': 350,
    '14b': 300,
    '30b': 120,
    '32b': 100,
    '70b': 40,
    '72b': 35,
};

// Number of target modules for LoRA (q, k, v, o, gate, up, down)
const LORA_TARGET_MODULES_COUNT = 7;

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

function getBaselineTokensPerSecond(modelSizeB: number): number {
    // Find the closest match in our baseline table
    if (modelSizeB <= 1.5) return BASELINE_TOKENS_PER_SECOND['1b'];
    if (modelSizeB <= 4) return BASELINE_TOKENS_PER_SECOND['3b'];
    if (modelSizeB <= 7.5) return BASELINE_TOKENS_PER_SECOND['7b'];
    if (modelSizeB <= 10) return BASELINE_TOKENS_PER_SECOND['8b'];
    if (modelSizeB <= 13.5) return BASELINE_TOKENS_PER_SECOND['13b'];
    if (modelSizeB <= 20) return BASELINE_TOKENS_PER_SECOND['14b'];
    if (modelSizeB <= 32) return BASELINE_TOKENS_PER_SECOND['30b'];
    if (modelSizeB <= 40) return BASELINE_TOKENS_PER_SECOND['32b'];
    if (modelSizeB <= 72) return BASELINE_TOKENS_PER_SECOND['70b'];
    return BASELINE_TOKENS_PER_SECOND['72b'];
}

/**
 * Estimates the number of trainable LoRA parameters.
 * LoRA adds A (d_model x r) and B (r x d_model) matrices to each target module.
 * For each module: 2 * d_model * r parameters.
 * Total: num_layers * target_modules_count * 2 * d_model * r
 */
function estimateLoraParams(model: ModelMetadata, loraR: number): number {
    return model.numLayers * LORA_TARGET_MODULES_COUNT * 2 * model.dModel * loraR;
}

// ============================================================================
// CORE ESTIMATION ENGINE
// ============================================================================

export function estimateResources(
    model: ModelMetadata,
    config: TrainingConfig,
    budget: HardwareBudget
): ResourceEstimate {
    const warnings: string[] = [];
    const breakdown: MemoryBreakdown = {
        modelWeightsVram: 0,
        loraVram: 0,
        activationsVram: 0,
        gradientVram: 0,
        optimizerVram: 0,
        modelWeightsRam: 0,
        optimizerRam: 0,
        systemOverhead: 0,
    };

    // --- 1. Model Weights ---
    // 4-bit: modelSizeB * 1e9 * 4 bits / 8 bits/byte / 1024^3 bytes/GB
    // 16-bit: modelSizeB * 1e9 * 16 bits / 8 bits/byte / 1024^3 bytes/GB
    const bitsPerParam = config.use4Bit ? 4 : 16;
    const totalWeightsGb = (model.sizeB * 1e9 * bitsPerParam) / (8 * 1024 ** 3);

    let modelWeightsOnGpu = totalWeightsGb;
    let modelWeightsOnRam = 0;

    if (config.hybridTraining) {
        // Determine how many layers are on GPU vs CPU
        const actualGpuLayers = config.gpuLayers === 'auto'
            ? Math.min(model.numLayers, Math.floor(model.numLayers * 0.5)) // Default to 50%
            : config.gpuLayers;
        const gpuRatio = Math.max(0, Math.min(1, actualGpuLayers / model.numLayers));
        modelWeightsOnGpu = totalWeightsGb * gpuRatio;
        modelWeightsOnRam = totalWeightsGb * (1 - gpuRatio);
    } else if (config.useCpuOffload) {
        // Full offload: Only ~20% on GPU at any time (rough estimate)
        modelWeightsOnGpu = totalWeightsGb * 0.25;
        modelWeightsOnRam = totalWeightsGb * 0.75;
    }

    breakdown.modelWeightsVram = modelWeightsOnGpu;
    breakdown.modelWeightsRam = modelWeightsOnRam;

    // --- 2. LoRA Adapter Weights ---
    // LoRA adapters are always on GPU (small size)
    const loraParams = estimateLoraParams(model, config.loraR);
    // FP16 for LoRA weights: loraParams * 2 bytes / 1024^3 GB
    breakdown.loraVram = (loraParams * 2) / (1024 ** 3);

    // --- 3. Optimizer States (AdamW) ---
    // AdamW stores m (momentum) and v (variance) for each trainable param.
    // 8-bit optimizer: 1 byte each, so 2 bytes total per param.
    // FP16 optimizer: 2 bytes each, so 4 bytes total per param.
    // We assume 8-bit by default.
    const optimizerBytesPerParam = config.usePagedOptimizer ? 2 : 4;
    const totalOptimizerGb = (loraParams * optimizerBytesPerParam) / (1024 ** 3);

    if (config.offloadOptimizer || config.usePagedOptimizer) {
        breakdown.optimizerRam = totalOptimizerGb;
        breakdown.optimizerVram = 0;
    } else {
        breakdown.optimizerVram = totalOptimizerGb;
        breakdown.optimizerRam = 0;
    }

    // --- 4. Activation Memory ---
    // Activations are hard to estimate precisely. A common formula:
    // activations_per_sample = 2 * num_layers * seq_len * d_model * (bytes_per_element) / 1024^3
    // The factor of 2 accounts for forward and backward passes.
    // Gradient checkpointing reduces this significantly (to sqrt(num_layers) factor).
    const bytesPerElement = 2; // FP16
    let activationFactor = 2.0;
    if (config.useGradientCheckpointing) {
        // GC stores only a subset of activations, recomputing on backward pass.
        // Reduces memory by a factor of ~sqrt(num_layers), but let's use a simpler 0.15x multiplier.
        activationFactor = 0.3;
    }

    const activationPerSampleGb =
        (activationFactor * model.numLayers * config.maxSeqLength * model.dModel * bytesPerElement) / (1024 ** 3);
    breakdown.activationsVram = config.batchSize * activationPerSampleGb;

    // --- 5. Gradient Memory ---
    // Gradients for trainable params (LoRA only). FP16 = 2 bytes per param.
    breakdown.gradientVram = (loraParams * 2) / (1024 ** 3);

    // --- 6. System Overhead ---
    // CUDA context, PyTorch overhead, small buffers. ~1.5 GB for large models.
    const vramOverhead = 1.5;
    // Python, dataset workers, etc. ~6-8 GB.
    breakdown.systemOverhead = 8;

    // Add VLM vision encoder overhead if applicable
    let visionOverhead = 0;
    if (model.isVision) {
        // Vision encoder adds ~1-3 GB VRAM depending on resolution
        visionOverhead = 2.0;
    }

    // --- Totals ---
    const totalVram = breakdown.modelWeightsVram +
        breakdown.loraVram +
        breakdown.activationsVram +
        breakdown.gradientVram +
        breakdown.optimizerVram +
        vramOverhead +
        visionOverhead;

    const totalRam = breakdown.modelWeightsRam +
        breakdown.optimizerRam +
        breakdown.systemOverhead;

    // --- Warnings ---
    if (totalVram > budget.vramGb) {
        warnings.push(`Estimated VRAM (${totalVram.toFixed(1)} GB) exceeds budget (${budget.vramGb} GB).`);
    }
    if (totalRam > budget.ramGb) {
        warnings.push(`Estimated RAM (${totalRam.toFixed(1)} GB) exceeds budget (${budget.ramGb} GB).`);
    }

    // --- Time Estimation ---
    let tokensPerSecond = getBaselineTokensPerSecond(model.sizeB);

    // Apply penalties for slower configurations
    if (config.hybridTraining) {
        tokensPerSecond *= 0.5; // Significant slowdown from CPU-GPU sync
    } else if (config.useCpuOffload) {
        tokensPerSecond *= 0.1; // Major slowdown
    }
    if (config.useGradientCheckpointing) {
        tokensPerSecond *= 0.8; // ~20% slower due to recomputation
    }
    if (config.useDeepspeed) {
        tokensPerSecond *= 0.9; // Slight overhead
    }

    const totalTokens = config.totalSamples * config.numEpochs * config.maxSeqLength;
    // Time in seconds = total_tokens / tokens_per_second
    // This is a simplification. Real calculation: (total_samples / effective_batch_size) * time_per_step
    const timeSeconds = totalTokens / tokensPerSecond;
    const timeHours = timeSeconds / 3600;

    return {
        vramGb: Math.round(totalVram * 10) / 10,
        ramGb: Math.round(totalRam * 10) / 10,
        timeHours: Math.round(timeHours * 100) / 100,
        tokensPerSecond: Math.round(tokensPerSecond),
        warnings,
        breakdown,
    };
}

// ============================================================================
// AUTO-OPTIMIZER
// ============================================================================

export function autoOptimize(
    model: ModelMetadata,
    desiredConfig: TrainingConfig,
    budget: HardwareBudget
): OptimizationResult {
    const config = { ...desiredConfig };
    const changesApplied: string[] = [];

    // Step 1: Try with current config
    let estimate = estimateResources(model, config, budget);

    if (estimate.vramGb <= budget.vramGb && estimate.ramGb <= budget.ramGb) {
        return { optimizedConfig: config, estimate, success: true, changesApplied };
    }

    // Step 2: Enable Gradient Checkpointing
    if (!config.useGradientCheckpointing) {
        config.useGradientCheckpointing = true;
        changesApplied.push("Enabled Gradient Checkpointing");
        estimate = estimateResources(model, config, budget);
        if (estimate.vramGb <= budget.vramGb && estimate.ramGb <= budget.ramGb) {
            return { optimizedConfig: config, estimate, success: true, changesApplied };
        }
    }

    // Step 3: Enable Paged Optimizer
    if (!config.usePagedOptimizer) {
        config.usePagedOptimizer = true;
        changesApplied.push("Enabled Paged 8-bit Optimizer");
        estimate = estimateResources(model, config, budget);
        if (estimate.vramGb <= budget.vramGb && estimate.ramGb <= budget.ramGb) {
            return { optimizedConfig: config, estimate, success: true, changesApplied };
        }
    }

    // Step 4: Reduce Batch Size
    while (config.batchSize > 1 && estimate.vramGb > budget.vramGb) {
        config.batchSize = Math.max(1, config.batchSize - 1);
        changesApplied.push(`Reduced Batch Size to ${config.batchSize}`);
        estimate = estimateResources(model, config, budget);
        if (estimate.vramGb <= budget.vramGb && estimate.ramGb <= budget.ramGb) {
            return { optimizedConfig: config, estimate, success: true, changesApplied };
        }
    }

    // Step 5: Enable Hybrid Training
    if (!config.hybridTraining && budget.ramGb > 32) { // Only suggest if RAM is substantial
        config.hybridTraining = true;
        config.gpuLayers = Math.floor(model.numLayers * 0.6); // Start with 60% on GPU
        config.offloadOptimizer = true;
        changesApplied.push(`Enabled Hybrid Training (${config.gpuLayers}/${model.numLayers} layers on GPU)`);
        estimate = estimateResources(model, config, budget);
        if (estimate.vramGb <= budget.vramGb && estimate.ramGb <= budget.ramGb) {
            return { optimizedConfig: config, estimate, success: true, changesApplied };
        }
    }

    // Step 6: Reduce GPU Layers (if Hybrid)
    if (config.hybridTraining && typeof config.gpuLayers === 'number') {
        while (config.gpuLayers > 5 && estimate.vramGb > budget.vramGb) {
            config.gpuLayers = Math.max(5, config.gpuLayers - 5);
            changesApplied.push(`Reduced GPU Layers to ${config.gpuLayers}`);
            estimate = estimateResources(model, config, budget);
            if (estimate.vramGb <= budget.vramGb && estimate.ramGb <= budget.ramGb) {
                return { optimizedConfig: config, estimate, success: true, changesApplied };
            }
        }
    }

    // Step 7: Reduce LoRA Rank (Last Resort)
    const rankSteps = [128, 64, 32, 16, 8];
    for (const r of rankSteps) {
        if (config.loraR > r) {
            config.loraR = r;
            config.loraAlpha = r * 2; // Common heuristic
            changesApplied.push(`Reduced LoRA Rank to ${r}`);
            estimate = estimateResources(model, config, budget);
            if (estimate.vramGb <= budget.vramGb && estimate.ramGb <= budget.ramGb) {
                return { optimizedConfig: config, estimate, success: true, changesApplied };
            }
        }
    }

    // Failed to fit
    return { optimizedConfig: config, estimate, success: false, changesApplied };
}

// ============================================================================
// MODEL METADATA EXTRACTION HELPERS
// ============================================================================

/**
 * Parses a model config.json to extract metadata.
 */
export function parseModelConfig(configJson: Record<string, unknown>): ModelMetadata {
    // Try to find num_hidden_layers and hidden_size.
    // These can be at the top level or nested under text_config (for VLMs).

    let numLayers = 32; // Default
    let dModel = 4096; // Default
    let sizeB = 7; // Default
    let isVision = false;

    // =========================================================================
    // 1. Extract architecture parameters
    // =========================================================================

    // Top level
    if (typeof configJson.num_hidden_layers === 'number') {
        numLayers = configJson.num_hidden_layers;
    }
    if (typeof configJson.hidden_size === 'number') {
        dModel = configJson.hidden_size;
    }

    // Nested in text_config (common for VLMs like Qwen-VL)
    const textConfig = configJson.text_config as Record<string, unknown> | undefined;
    if (textConfig) {
        if (typeof textConfig.num_hidden_layers === 'number') {
            numLayers = textConfig.num_hidden_layers;
        }
        if (typeof textConfig.hidden_size === 'number') {
            dModel = textConfig.hidden_size;
        }
    }

    // Vision config indicates a VLM
    if (configJson.vision_config) {
        isVision = true;
    }

    // =========================================================================
    // 2. BEST: Use num_parameters if directly available (some configs have this)
    // =========================================================================
    const numParams = configJson.num_parameters ?? configJson.n_parameters ??
        (textConfig as Record<string, unknown> | undefined)?.num_parameters;

    if (typeof numParams === 'number' && numParams > 0) {
        sizeB = numParams / 1e9;
        console.log(`[ResourceBudgetEngine] Model size from num_parameters: ${sizeB.toFixed(1)}B`);
        return { sizeB: Math.round(sizeB * 10) / 10, numLayers, dModel, isVision };
    }

    // =========================================================================
    // 3. NEXT: Check for MoE (Mixture of Experts) indicators
    // MoE models have many more total params than indicated by hidden_size
    // =========================================================================
    const numExperts = configJson.num_local_experts ?? configJson.num_experts ??
        (textConfig as Record<string, unknown> | undefined)?.num_local_experts ??
        (textConfig as Record<string, unknown> | undefined)?.num_experts;

    const expertsDim = configJson.moe_intermediate_size ?? configJson.expert_intermediate_size ??
        (textConfig as Record<string, unknown> | undefined)?.moe_intermediate_size;

    const isMoE = typeof numExperts === 'number' && numExperts > 1;

    // =========================================================================
    // 4. Calculate from architecture (more accurate than just hidden_size)
    // =========================================================================
    // Transformer parameter formula (approximate):
    // - Embedding: vocab_size × hidden_size (usually ~2-4% of total)
    // - Each layer: ~12 × hidden_size^2 for standard, more for MoE
    // - LM Head: hidden_size × vocab_size (shared with embeddings usually)

    const vocabSize = (configJson.vocab_size ?? textConfig?.vocab_size ?? 32000) as number;
    const intermediateSize = (configJson.intermediate_size ?? textConfig?.intermediate_size ?? dModel * 4) as number;

    // Base calculation: attention + MLP per layer + embeddings
    // Attention: 4 × d_model² (Q, K, V, O projections)
    // MLP: 3 × d_model × intermediate_size (gate, up, down for SwiGLU)
    // Embeddings: vocab × d_model

    const attentionParams = 4 * dModel * dModel;
    let mlpParams = 3 * dModel * intermediateSize;

    if (isMoE && typeof numExperts === 'number') {
        // MoE: each expert has its own MLP, plus router
        const expertMlpSize = typeof expertsDim === 'number' ? expertsDim : intermediateSize;
        mlpParams = numExperts * 3 * dModel * expertMlpSize;
        // Add router parameters (small but counted)
        mlpParams += dModel * numExperts;
        console.log(`[ResourceBudgetEngine] MoE detected: ${numExperts} experts`);
    }

    const layerParams = attentionParams + mlpParams;
    const totalLayerParams = numLayers * layerParams;
    const embeddingParams = 2 * vocabSize * dModel; // embed + LM head (often shared but counted for safety)

    const calculatedParams = totalLayerParams + embeddingParams;
    sizeB = calculatedParams / 1e9;

    console.log(`[ResourceBudgetEngine] Calculated model size: ${sizeB.toFixed(1)}B (layers=${numLayers}, d_model=${dModel}, MoE=${isMoE})`);

    // =========================================================================
    // 5. Sanity check: clamp to reasonable range and round
    // =========================================================================
    if (sizeB < 0.1) sizeB = 0.5;
    if (sizeB > 500) sizeB = 500; // Cap at 500B for display sanity

    // Round to 1 decimal place for clean display
    sizeB = Math.round(sizeB * 10) / 10;

    return { sizeB, numLayers, dModel, isVision };
}

/**
 * Fallback: Extract model size from path name (e.g., "Qwen-7B" -> 7)
 * Handles MoE naming like "30B-A3B" by preferring the larger number.
 */
export function extractModelSizeFromPath(modelPath: string): number {
    const name = modelPath.toLowerCase();

    // Find all numbers followed by 'b' (like 7b, 30b, 72b)
    const matches = name.match(/(\d+\.?\d*)b/g);
    if (matches && matches.length > 0) {
        // Parse all matches and return the largest (to handle "30B-A3B" -> 30)
        const sizes = matches.map(m => {
            const numMatch = m.match(/(\d+\.?\d*)/);
            return numMatch ? parseFloat(numMatch[1]) : 0;
        });
        return Math.max(...sizes);
    }

    // Common keywords
    if (name.includes('tiny') || name.includes('mini')) return 0.5;
    if (name.includes('small')) return 1;
    if (name.includes('base')) return 3;
    if (name.includes('large') && !name.includes('xl')) return 7;
    if (name.includes('xl') && !name.includes('xxl')) return 13;
    if (name.includes('xxl')) return 70;
    return 7; // Default
}
