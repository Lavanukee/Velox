
import React, { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { open } from '@tauri-apps/plugin-dialog';
import { Select } from './components/Select';
import './styles/legacy.css';

interface UtilitiesPanelProps {
  addLogMessage: (message: string) => void;
  addNotification: (message: string, type?: 'success' | 'error' | 'info') => void;
}

const UtilitiesPanel: React.FC<UtilitiesPanelProps> = ({ addLogMessage, addNotification }) => {

  const [conversionType, setConversionType] = useState<'hf_to_gguf' | 'lora_to_gguf'>('hf_to_gguf');
  const [sourcePath, setSourcePath] = useState('');
  const [outputPath, setOutputPath] = useState('');
  const [basePath, setBasePath] = useState(''); // For LoRA conversion
  const [quantizationType, setQuantizationType] = useState<'q8_0' | 'f16' | 'bf16'>('q8_0'); // New state for quantization
  const [isConverting, setIsConverting] = useState(false);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [availableLoras, setAvailableLoras] = useState<string[]>([]);

  useEffect(() => {
    loadResources();
  }, []);

  const loadResources = async () => {
    try {
      const models: string[] = await invoke('list_model_folders_command');
      const loras: string[] = await invoke('list_lora_adapters_command');
      setAvailableModels(models);
      setAvailableLoras(loras);
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
      // Allow directory selection so users can pick model dirs; this can be adjusted
      // to allow files instead by setting `directory: false` if desired.
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
    addLogMessage(`Starting ${conversionType} conversion...`);

    try {
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

      setSourcePath('');
      setOutputPath('');
      setBasePath('');
    } catch (error) {
      addLogMessage(`ERROR during conversion: ${error}`);
      addNotification('Conversion failed', 'error');
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
          <h2>Model Conversion</h2>

          <div className="conversion-selector">
            <label>
              <input
                type="radio"
                value="hf_to_gguf"
                checked={conversionType === 'hf_to_gguf'}
                onChange={(e) => setConversionType(e.target.value as any)}
              />
              <span>HuggingFace → GGUF</span>
            </label>
            <label>
              <input
                type="radio"
                value="lora_to_gguf"
                checked={conversionType === 'lora_to_gguf'}
                onChange={(e) => setConversionType(e.target.value as any)}
              />
              <span>LoRA → GGUF-Compatible LoRA</span>
            </label>
          </div>

          <div className="conversion-form">
            {conversionType === 'hf_to_gguf' ? (
              <>
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
              </>
            ) : (
              <>
                <div className="input-group">
                  <label>Source LoRA</label>
                  <div className="file-input-group">
                    <Select
                      style={{ flex: 1 }}
                      value={sourcePath}
                      onChange={(val) => setSourcePath(val)}
                      placeholder="Select a LoRA..."
                      options={[
                        { value: '', label: 'Select a LoRA...' },
                        ...availableLoras.map(lora => ({ value: `data/loras/${lora}`, label: lora }))
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
                </div>
              </>
            )}

            <div className="input-group">
              <label>Quantization Type</label>
              <Select
                value={quantizationType}
                onChange={(val) => setQuantizationType(val as 'q8_0' | 'f16' | 'bf16')}
                options={[
                  { value: 'q8_0', label: 'q8_0 (8-bit integer)' },
                  { value: 'f16', label: 'f16 (16-bit float)' },
                  { value: 'bf16', label: 'bf16 (bfloat16)' }
                ]}
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
            </div>

            <button
              className="btn-primary btn-large"
              onClick={handleConvert}
              disabled={isConverting || !sourcePath || (conversionType === 'lora_to_gguf' && !basePath)}
            >
              {isConverting ? 'Converting...' : 'Convert'}
            </button>
          </div>

          <div className="conversion-info">
            <h3>ℹ️ Conversion Info</h3>
            {conversionType === 'hf_to_gguf' ? (
              <p>
                Converts HuggingFace model formats to GGUF format compatible with llama.cpp.
                This process may take several minutes depending on model size.
              </p>
            ) : (
              <p>
                Converts LoRA adapters to GGUF-compatible format. Requires the base model
                that the LoRA was trained on for proper conversion.
              </p>
            )}
          </div>
        </div>

        <div className="drag-drop-zone">
          <div className="drag-drop-content">
            <h3>🎯 Quick Convert</h3>
            <p>Drag and drop files here for quick conversion</p>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: '1rem' }}>
              Supported: .safetensors, .bin, model directories
            </p>
            <div style={{ display: 'flex', justifyContent: 'center', marginTop: '1rem', flexDirection: 'column', alignItems: 'center' }}>
              <button className="btn-secondary" onClick={handleQuickBrowse}>
                Browse
              </button>

              {sourcePath && (
                <div style={{ marginTop: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <code style={{ fontSize: '0.85rem', color: 'var(--text-muted)', maxWidth: '420px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', display: 'block' }}>
                    {sourcePath}
                  </code>
                  <button
                    className="btn-secondary"
                    onClick={() => {
                      setSourcePath('');
                      addLogMessage('Cleared quick-convert selection');
                      addNotification('Cleared quick-convert selection', 'info');
                    }}
                  >
                    Clear
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default UtilitiesPanel;