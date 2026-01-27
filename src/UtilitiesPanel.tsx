
import React, { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { open } from '@tauri-apps/plugin-dialog';
import { Select } from './components/Select';
import { useApp } from './context/AppContext';
import './styles/legacy.css';

interface UtilitiesPanelProps {
  addLogMessage: (message: string) => void;
  addNotification: (message: string, type?: 'success' | 'error' | 'info') => void;
}

const UtilitiesPanel: React.FC<UtilitiesPanelProps> = ({ addLogMessage, addNotification }) => {

  const {
    utIsConverting: isConverting, setUtIsConverting: setIsConverting,
    utConversionLabel: _conversionLabel, setUtConversionLabel: setConversionLabel,
    utSourcePath: sourcePath, setUtSourcePath: setSourcePath,
    utOutputPath: outputPath, setUtOutputPath: setOutputPath,
    utBasePath: basePath, setUtBasePath: setBasePath,
    utQuantizationType: quantizationType, setUtQuantizationType: setQuantizationType,
    utConversionType: conversionType, setUtConversionType: setConversionType,
    utConversionEngine: conversionEngine, setUtConversionEngine: setConversionEngine,
    resources: _resources, loadResources: globalLoadResources
  } = useApp();

  const [showAdvanced, setShowAdvanced] = useState(false);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [availableLoras, setAvailableLoras] = useState<string[]>([]);
  const [projectLoras, setProjectLoras] = useState<any[]>([]);

  // VRAM Auto-detection
  const [maxVram, setMaxVram] = useState<number>(0);

  // Quantization options based on engine
  const standardQuantOptions = [
    { value: 'q8_0', label: 'q8_0 (8-bit integer)' },
    { value: 'f16', label: 'f16 (16-bit float)' },
    { value: 'bf16', label: 'bf16 (bfloat16)' }
  ];

  const unslothQuantOptions = [
    { value: 'q4_k_m', label: 'Q4_K_M (4-bit, recommended)' },
    { value: 'q5_k_m', label: 'Q5_K_M (5-bit, balanced)' },
    { value: 'q8_0', label: 'Q8_0 (8-bit, high quality)' },
    { value: 'f16', label: 'F16 (16-bit, best quality)' },
    { value: 'not_quantized', label: 'None (Full precision)' }
  ];

  useEffect(() => {
    loadResources();
    detectHardware();
  }, []);

  const detectHardware = async () => {
    try {
      const info: any = await invoke('get_hardware_info_command');
      if (info && info.gpus && info.gpus.length > 0) {
        const bestGpu = info.gpus[0];
        setMaxVram(bestGpu.vram_total);
        addLogMessage(`Detected GPU: ${bestGpu.name} (${(bestGpu.vram_total / 1024 / 1024 / 1024).toFixed(1)} GB VRAM)`);
      }
    } catch (e) {
      console.error("Hardware detect failed", e);
    }
  };

  useEffect(() => {
    if (conversionEngine === 'unsloth') {
      const vramGb = maxVram / (1024 * 1024 * 1024);
      if (vramGb >= 20) {
        setQuantizationType('q8_0');
      } else if (vramGb >= 8) {
        setQuantizationType('q4_k_m');
      } else {
        setQuantizationType('q4_k_m');
      }
    } else {
      setQuantizationType('q8_0');
    }
  }, [conversionEngine, maxVram]);

  const loadResources = async () => {
    try {
      const models: string[] = await invoke('list_model_folders_command');
      const loras: string[] = await invoke('list_lora_adapters_command');
      const projects: any[] = await invoke('list_loras_by_project_command');
      setAvailableModels(models);
      setAvailableLoras(loras);
      setProjectLoras(projects);
    } catch (error) {
      addLogMessage(`ERROR loading resources: ${error}`);
    }
  };

  const handleBrowseSource = async () => {
    try {
      const selected = await open({
        multiple: false,
        directory: conversionType === 'hf_to_gguf',
      });
      if (selected) {
        const path = Array.isArray(selected) ? selected[0] : selected;
        setSourcePath(path);
      }
    } catch (error) {
      addLogMessage(`ERROR selecting source: ${error}`);
    }
  };

  const handleBrowseBase = async () => {
    try {
      const selected = await open({
        multiple: false,
        directory: true,
      });
      if (selected) {
        const path = Array.isArray(selected) ? selected[0] : selected;
        setBasePath(path);
      }
    } catch (error) {
      addLogMessage(`ERROR selecting base model: ${error}`);
    }
  };

  const handleQuickBrowse = async () => {
    try {
      const selected = await open({
        multiple: false,
        directory: true,
      });
      if (selected) {
        const path = Array.isArray(selected) ? selected[0] : selected;
        setSourcePath(path);
        addLogMessage(`Selected for quick convert: ${path}`);
        addNotification(`Selected for quick convert: ${path}`, 'info');
      }
    } catch (error) {
      addLogMessage(`ERROR selecting quick-convert source: ${error}`);
    }
  };

  const handleConvert = async () => {
    if (!sourcePath) {
      addNotification('Please select a source', 'error');
      return;
    }

    if (conversionType === 'lora_to_gguf' && !basePath) {
      addNotification('Please select a base model for LoRA conversion', 'error');
      return;
    }

    setIsConverting(true);
    const engineLabel = conversionEngine === 'unsloth' ? 'Unsloth' : 'Standard';
    const label = `${engineLabel} ${conversionType === 'hf_to_gguf' ? 'HF → GGUF' : 'LoRA → GGUF'}`;
    setConversionLabel(label);
    addLogMessage(`Starting ${label} conversion...`);

    try {
      if (conversionEngine === 'unsloth') {
        let finalOutput = outputPath;
        if (!finalOutput) {
          const rawName = sourcePath.split(/[\\/]/).pop() || 'model';
          const nameWithoutExt = rawName.replace(/\.[^/.]+$/, "");
          finalOutput = `${sourcePath.substring(0, sourcePath.lastIndexOf(rawName))}${nameWithoutExt}-unsloth-${quantizationType}.gguf`;
        }

        const unslothArgs = conversionType === 'lora_to_gguf'
          ? {
            sourcePath: basePath, // Base model is the primary model for Unsloth
            outputPath: finalOutput,
            quantizationType,
            loraPath: sourcePath // LoRA adapter is the additional one to merge
          }
          : {
            sourcePath,
            outputPath: finalOutput,
            quantizationType,
            loraPath: null
          };

        await invoke('convert_unsloth_gguf_command', unslothArgs);
        addLogMessage("Unsloth GGUF conversion complete!");
        addNotification('Unsloth GGUF conversion complete!', 'success');
      } else {
        if (conversionType === 'hf_to_gguf') {
          await invoke('convert_hf_to_gguf_command', {
            sourcePath,
            outputPath: outputPath || undefined,
            quantizationType
          });
          addNotification('HF model converted to GGUF!', 'success');
        } else {
          await invoke('convert_lora_to_gguf_command', {
            loraPath: sourcePath,
            basePath,
            outputPath: outputPath || undefined,
            quantizationType
          });
          addNotification('LoRA converted to GGUF-compatible!', 'success');
        }
      }

      setSourcePath('');
      setOutputPath('');
      setBasePath('');
      setConversionLabel('');
      globalLoadResources(); // Sync resources
    } catch (error) {
      addLogMessage(`ERROR during conversion: ${error}`);
      addNotification(`Conversion failed: ${error}`, 'error');
    } finally {
      setIsConverting(false);
    }
  };

  return (
    <div className="utilities-container">
      <div className="section-header">
        <h1>Utilities</h1>
        <p style={{ color: 'var(--text-muted)' }}>Convert and manage your model formats</p>
      </div>

      <div className="utilities-content">
        <div className="conversion-panel">
          <h2 style={{ marginBottom: '24px' }}>Model Conversion</h2>

          {/* Unsloth Hero Section (Default) */}
          <div style={{
            padding: '24px',
            background: 'rgba(34, 197, 94, 0.05)',
            border: '1px solid rgba(34, 197, 94, 0.15)',
            borderRadius: '12px',
            marginBottom: '24px',
            position: 'relative',
            overflow: 'hidden'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
              <h3 style={{ color: '#4ade80', fontWeight: 600, fontSize: '18px', margin: 0 }}>Unsloth Dynamic GGUF</h3>
              <span style={{
                fontSize: '11px',
                fontFamily: 'monospace',
                background: 'rgba(34, 197, 94, 0.2)',
                color: '#86efac',
                padding: '4px 8px',
                borderRadius: '4px'
              }}>FAST • 2.0</span>
            </div>
            <p style={{ fontSize: '14px', color: 'var(--text-muted)', marginBottom: '16px', lineHeight: '1.5' }}>
              High-performance conversion engine. Auto-detects optimal settings for your hardware.
            </p>

            <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
              <button
                onClick={() => {
                  setConversionEngine(conversionEngine === 'unsloth' ? 'standard' : 'unsloth');
                  setShowAdvanced(!showAdvanced);
                }}
                style={{
                  background: 'none',
                  border: 'none',
                  fontSize: '11px',
                  color: 'var(--text-muted)',
                  cursor: 'pointer',
                  textDecoration: 'underline',
                  padding: 0
                }}
              >
                {conversionEngine === 'unsloth' ? 'Switch to Legacy / Standard Engine' : 'Switch back to Unsloth'}
              </button>
            </div>
          </div>

          {conversionEngine === 'standard' && (
            <div style={{
              padding: '16px',
              background: 'rgba(59, 130, 246, 0.05)',
              border: '1px solid rgba(59, 130, 246, 0.15)',
              borderRadius: '12px',
              marginBottom: '24px'
            }}>
              <div style={{ fontSize: '13px', color: '#60a5fa', fontWeight: 500, marginBottom: '4px' }}>Standard Engine Active</div>
              <p style={{ fontSize: '12px', color: 'var(--text-muted)', margin: 0 }}>Using standard llama.cpp conversion scripts. Slower but widely compatible.</p>
            </div>
          )}

          {/* New Tabbed Conversion Selector */}
          <div style={{
            display: 'flex',
            gap: '4px',
            background: 'rgba(255,255,255,0.03)',
            padding: '4px',
            borderRadius: '8px',
            marginBottom: '24px'
          }}>
            <button
              onClick={() => setConversionType('hf_to_gguf')}
              style={{
                flex: 1,
                padding: '10px',
                fontSize: '13px',
                fontWeight: 500,
                border: 'none',
                background: conversionType === 'hf_to_gguf' ? 'rgba(59, 130, 246, 0.15)' : 'transparent',
                color: conversionType === 'hf_to_gguf' ? '#93c5fd' : '#9ca3af',
                borderRadius: '6px',
                cursor: 'pointer',
                transition: 'all 0.2s'
              }}
            >
              HuggingFace → GGUF
            </button>
            <button
              onClick={() => setConversionType('lora_to_gguf')}
              style={{
                flex: 1,
                padding: '10px',
                fontSize: '13px',
                fontWeight: 500,
                border: 'none',
                background: conversionType === 'lora_to_gguf' ? 'rgba(59, 130, 246, 0.15)' : 'transparent',
                color: conversionType === 'lora_to_gguf' ? '#93c5fd' : '#9ca3af',
                borderRadius: '6px',
                cursor: 'pointer',
                transition: 'all 0.2s'
              }}
            >
              LoRA → GGUF-Compatible
            </button>
          </div>

          <div className="conversion-form">
            {conversionType === 'hf_to_gguf' ? (
              <div className="input-group">
                <label>Source HF Model</label>
                <div className="file-input-group">
                  <Select
                    style={{ flex: 1 }}
                    value={sourcePath}
                    onChange={(val) => setSourcePath(val)}
                    placeholder="Select a model..."
                    options={[
                      { value: '', label: 'Select a model...' },
                      ...availableModels.map(model => ({ value: `data/models/${model}`, label: model }))
                    ]}
                  />
                  <button className="btn-secondary" onClick={handleBrowseSource}>
                    Browse
                  </button>
                </div>
              </div>
            ) : (
              <>
                <div className="input-group">
                  <label>Source LoRA</label>
                  <div className="file-input-group">
                    <Select
                      style={{ flex: 1 }}
                      value={sourcePath}
                      onChange={(val) => {
                        setSourcePath(val);
                        // Auto-select base model for projects
                        if (val.includes('/outputs/')) {
                          const parts = val.split('/');
                          const outputIdx = parts.indexOf('outputs');
                          if (outputIdx >= 0 && parts[outputIdx + 1]) {
                            const projectName = parts[outputIdx + 1];
                            const project = projectLoras.find(p => p.project_name === projectName);

                            if (project && project.base_model) {
                              // Try to fuzzy match with available models
                              // project.base_model could be full path C:\... or relative
                              const baseName = project.base_model.replace(/\\/g, '/').split('/').pop();

                              if (baseName) {
                                const match = availableModels.find(m =>
                                  m === baseName ||
                                  m.toLowerCase() === baseName.toLowerCase() ||
                                  m.includes(baseName)
                                );

                                if (match) {
                                  setBasePath(`data/models/${match}`);
                                  addNotification(`Auto-selected base model: ${match}`, 'info');
                                }
                              }
                            }
                          }
                        }
                      }}
                      placeholder="Select a LoRA..."
                      options={[
                        { value: '', label: 'Select a LoRA...' },
                        ...availableLoras.map(lora => {
                          const isProjectLoRA = lora.startsWith('outputs/');
                          const path = isProjectLoRA
                            ? `data/outputs/${lora.replace('outputs/', '')}/final_model`
                            : `data/loras/${lora}`;
                          const label = isProjectLoRA
                            ? `Project: ${lora.replace('outputs/', '')}`
                            : lora;
                          return { value: path, label };
                        })
                      ]}
                    />
                    <button className="btn-secondary" onClick={handleBrowseSource}>
                      Browse
                    </button>
                  </div>
                </div>

                <div className="input-group">
                  <label>Base Model (for LoRA conversion)</label>
                  <div className="file-input-group">
                    <Select
                      style={{ flex: 1 }}
                      value={basePath}
                      onChange={(val) => setBasePath(val)}
                      placeholder="Select base model..."
                      options={[
                        { value: '', label: 'Select base model...' },
                        ...availableModels.map(model => ({ value: `data/models/${model}`, label: model }))
                      ]}
                    />
                    <button className="btn-secondary" onClick={handleBrowseBase}>
                      Browse
                    </button>
                  </div>
                  <p style={{ fontSize: '11px', color: '#f87171', marginTop: '6px', opacity: 0.9 }}>
                    <span style={{ fontWeight: 600 }}>Note:</span> Base model must be a standard HuggingFace model (Safetensors/Pytorch), not a GGUF file.
                  </p>
                </div>
              </>
            )}

            <div className="input-group">
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <label style={{ margin: 0 }}>Quantization Type</label>
                {conversionEngine === 'unsloth' && maxVram > 0 && (
                  <span style={{ fontSize: '11px', color: '#4ade80', opacity: 0.8 }}>
                    (Auto-selected for {(maxVram / 1024 / 1024 / 1024).toFixed(0)}GB VRAM)
                  </span>
                )}
              </div>
              <Select
                value={quantizationType}
                onChange={(val) => setQuantizationType(val)}
                options={conversionEngine === 'unsloth' ? unslothQuantOptions : standardQuantOptions}
              />
            </div>

            <div className="input-group">
              <label>Output Path (optional)</label>
              <input
                className="input-field"
                value={outputPath}
                onChange={(e) => setOutputPath(e.target.value)}
                placeholder="Leave empty for automatic naming"
              />
              <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>
                Default: [Input Name]-unsloth-[Quant].gguf
              </p>
            </div>

            <button
              className="btn-primary"
              onClick={handleConvert}
              disabled={isConverting || !sourcePath || (conversionType === 'lora_to_gguf' && !basePath)}
              style={{
                width: '100%',
                padding: '14px',
                fontSize: '15px',
                marginTop: '12px',
                background: conversionEngine === 'unsloth'
                  ? 'linear-gradient(135deg, #22c55e 0%, #16a34a 100%)'
                  : 'var(--gradient-primary)'
              }}
            >
              {isConverting ? 'Converting...' : `Convert`}
            </button>
          </div>
        </div>

        <div className="drag-drop-zone">
          <div className="drag-drop-content">
            <h3>Quick Convert</h3>
            <p style={{ color: 'var(--text-muted)', marginBottom: '24px' }}>Drag and drop files here for quick conversion</p>

            <div style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: '16px'
            }}>
              <button
                className="btn-secondary"
                onClick={handleQuickBrowse}
                style={{ padding: '12px 24px', fontSize: '14px' }}
              >
                Browse Files
              </button>

              {sourcePath && (
                <div style={{
                  background: 'rgba(255,255,255,0.03)',
                  padding: '12px',
                  borderRadius: '8px',
                  width: '100%',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '12px',
                  border: '1px solid var(--border-color)'
                }}>
                  <code style={{
                    fontSize: '12px',
                    color: 'var(--text-muted)',
                    flex: 1,
                    textAlign: 'left',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap'
                  }}>
                    {sourcePath}
                  </code>
                  <button
                    className="btn-secondary"
                    onClick={() => setSourcePath('')}
                    style={{ padding: '4px 8px', fontSize: '11px' }}
                  >
                    Clear
                  </button>
                </div>
              )}
            </div>

            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '32px' }}>
              Supported: .safetensors, .bin, model directories
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default UtilitiesPanel;