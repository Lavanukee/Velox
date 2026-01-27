import React from 'react';
import { Card } from '../components/Card';
import { Button } from '../components/Button';
import { Toggle } from '../components/Toggle';
import { RotateCcw, Info, RefreshCw, Cpu, MemoryStick, HardDrive } from 'lucide-react';
import { useApp } from '../context/AppContext';
import { invoke } from '@tauri-apps/api/core';

interface GpuInfo {
    name: string;
    vram_total: number;
}

interface HardwareInfo {
    cpu: string;
    ram_total: number;
    gpus: GpuInfo[];
}

interface SettingsPageProps {
    onReinstallPython?: () => void;
    onReinstallDependencies?: () => void;
}

const SettingsPage: React.FC<SettingsPageProps> = ({ onReinstallPython, onReinstallDependencies }) => {
    const {
        autoUpdate, setAutoUpdate,
        showInfoTooltips, setShowInfoTooltips,
        userFeatures, setUserFeatures
    } = useApp();

    const [hwInfo, setHwInfo] = React.useState<HardwareInfo | null>(null);

    React.useEffect(() => {
        const fetchHw = async () => {
            try {
                const info = await invoke<HardwareInfo>('get_hardware_info_command');
                setHwInfo(info);
            } catch (err) {
                console.error("Failed to fetch hardware info:", err);
            }
        };
        fetchHw();
    }, []);

    const formatBytes = (bytes: number) => {
        if (bytes === 0) return '0 GB';
        const gb = bytes / (1024 * 1024 * 1024);
        return `${gb.toFixed(1)} GB`;
    };

    return (
        <div className="space-y-6 max-w-3xl mx-auto" style={{ maxWidth: '800px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '24px' }}>
            <Card>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '24px' }}>
                    <h2 className="text-xl font-bold">General Settings</h2>
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                    <div className="list-item" style={{ borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)', justifyContent: 'space-between' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                            <div style={{ padding: '10px', background: 'rgba(59,130,246,0.1)', borderRadius: '10px', color: 'var(--accent-blue)' }}>
                                <RefreshCw size={20} />
                            </div>
                            <div>
                                <h3 className="font-semibold text-main">Auto-Update</h3>
                                <p className="text-sm text-secondary">Keep your AI engines and environment up to date automatically.</p>
                            </div>
                        </div>
                        <Toggle checked={autoUpdate} onChange={setAutoUpdate} />
                    </div>

                    <div className="list-item" style={{ borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)', justifyContent: 'space-between' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                            <div style={{ padding: '10px', background: 'rgba(16,185,129,0.1)', borderRadius: '10px', color: 'var(--accent-green)' }}>
                                <Info size={20} />
                            </div>
                            <div>
                                <h3 className="font-semibold text-main">Show Helper Tooltips</h3>
                                <p className="text-sm text-secondary">Display (i) icons throughout the app for helpful information.</p>
                            </div>
                        </div>
                        <Toggle checked={showInfoTooltips} onChange={setShowInfoTooltips} />
                    </div>
                </div>
            </Card>

            <Card>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
                    <h2 className="text-xl font-bold">Feature Management</h2>
                </div>
                <p className="text-sm text-secondary mb-6">Granular control over specific Power User features and interface elements.</p>

                <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(2, 1fr)',
                    gap: '12px'
                }}>
                    <div className="selection-card" style={{ padding: '12px 16px' }} onClick={() => setUserFeatures({ showUserModeToggle: !userFeatures.showUserModeToggle })}>
                        <input type="checkbox" checked={userFeatures.showUserModeToggle} readOnly />
                        <div style={{ flex: 1 }}>
                            <div className="font-semibold text-sm">Mode Toggle</div>
                            <div style={{ fontSize: '11px', color: 'var(--text-dim)' }}>Show Power User switch in header</div>
                        </div>
                    </div>

                    <div className="selection-card" style={{ padding: '12px 16px' }} onClick={() => setUserFeatures({ showAdvancedFinetuning: !userFeatures.showAdvancedFinetuning })}>
                        <input type="checkbox" checked={userFeatures.showAdvancedFinetuning} readOnly />
                        <div style={{ flex: 1 }}>
                            <div className="font-semibold text-sm">Advanced Fine-Tuning</div>
                            <div style={{ fontSize: '11px', color: 'var(--text-dim)' }}>Unlock all hyperparameters</div>
                        </div>
                    </div>

                    <div className="selection-card" style={{ padding: '12px 16px' }} onClick={() => setUserFeatures({ showAdvancedInference: !userFeatures.showAdvancedInference })}>
                        <input type="checkbox" checked={userFeatures.showAdvancedInference} readOnly />
                        <div style={{ flex: 1 }}>
                            <div className="font-semibold text-sm">Advanced Inference</div>
                            <div style={{ fontSize: '11px', color: 'var(--text-dim)' }}>GPU layers & batch size controls</div>
                        </div>
                    </div>

                    <div className="selection-card" style={{ padding: '12px 16px' }} onClick={() => setUserFeatures({ showMetrics: !userFeatures.showMetrics })}>
                        <input type="checkbox" checked={userFeatures.showMetrics} readOnly />
                        <div style={{ flex: 1 }}>
                            <div className="font-semibold text-sm">Real-time Metrics</div>
                            <div style={{ fontSize: '11px', color: 'var(--text-dim)' }}>Tokens/sec & latency tracking</div>
                        </div>
                    </div>

                    <div className="selection-card" style={{ padding: '12px 16px' }} onClick={() => setUserFeatures({ enableWebSearch: !userFeatures.enableWebSearch })}>
                        <input type="checkbox" checked={userFeatures.enableWebSearch} readOnly />
                        <div style={{ flex: 1 }}>
                            <div className="font-semibold text-sm">Web Search</div>
                            <div style={{ fontSize: '11px', color: 'var(--text-dim)' }}>Allow model to browse internet</div>
                        </div>
                    </div>

                    <div className="selection-card" style={{ padding: '12px 16px' }} onClick={() => setUserFeatures({ enableCodeExec: !userFeatures.enableCodeExec })}>
                        <input type="checkbox" checked={userFeatures.enableCodeExec} readOnly />
                        <div style={{ flex: 1 }}>
                            <div className="font-semibold text-sm">Code Execution</div>
                            <div style={{ fontSize: '11px', color: 'var(--text-dim)' }}>Run generated python locally</div>
                        </div>
                    </div>

                    <div className="selection-card" style={{ padding: '12px 16px' }} onClick={() => setUserFeatures({ enableCanvas: !userFeatures.enableCanvas })}>
                        <input type="checkbox" checked={userFeatures.enableCanvas} readOnly />
                        <div style={{ flex: 1 }}>
                            <div className="font-semibold text-sm">Canvas UI</div>
                            <div style={{ fontSize: '11px', color: 'var(--text-dim)' }}>Side-by-side artifact rendering</div>
                        </div>
                    </div>

                    <div className="selection-card" style={{ padding: '12px 16px' }} onClick={() => setUserFeatures({ autoFitMemory: !userFeatures.autoFitMemory })}>
                        <input type="checkbox" checked={userFeatures.autoFitMemory} readOnly />
                        <div style={{ flex: 1 }}>
                            <div className="font-semibold text-sm">Memory Auto-Fit</div>
                            <div style={{ fontSize: '11px', color: 'var(--text-dim)' }}>Automatic kv-cache management</div>
                        </div>
                    </div>

                    <div className="selection-card" style={{ padding: '12px 16px' }} onClick={() => setUserFeatures({ autoProcessDatasets: !userFeatures.autoProcessDatasets })}>
                        <input type="checkbox" checked={userFeatures.autoProcessDatasets} readOnly />
                        <div style={{ flex: 1 }}>
                            <div className="font-semibold text-sm">Auto-Process Datasets</div>
                            <div style={{ fontSize: '11px', color: 'var(--text-dim)' }}>Background vision-tagging/conversion</div>
                        </div>
                    </div>
                </div>
            </Card>

            <Card>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '24px' }}>
                    <h2 className="text-xl font-bold">System Specifications</h2>
                </div>

                {!hwInfo ? (
                    <div className="animate-pulse" style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                        <div style={{ background: 'var(--bg-elevated)', borderRadius: 'var(--radius-md)', height: '48px' }} />
                        <div style={{ background: 'var(--bg-elevated)', borderRadius: 'var(--radius-md)', height: '48px' }} />
                    </div>
                ) : (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                        {/* CPU */}
                        <div className="list-item" style={{ borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)', gap: '16px' }}>
                            <div style={{ padding: '10px', background: 'rgba(239,68,68,0.1)', borderRadius: '10px', color: 'var(--accent-red)' }}>
                                <Cpu size={20} />
                            </div>
                            <div style={{ flex: 1 }}>
                                <div style={{ fontSize: '10px', color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 700, marginBottom: '2px' }}>Processor</div>
                                <div style={{ color: 'var(--text-main)', fontWeight: 500 }}>{hwInfo.cpu}</div>
                            </div>
                        </div>

                        {/* RAM */}
                        <div className="list-item" style={{ borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)', gap: '16px' }}>
                            <div style={{ padding: '10px', background: 'rgba(167,139,250,0.1)', borderRadius: '10px', color: 'var(--accent-primary)' }}>
                                <MemoryStick size={20} />
                            </div>
                            <div style={{ flex: 1 }}>
                                <div style={{ fontSize: '10px', color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 700, marginBottom: '2px' }}>System Memory</div>
                                <div style={{ color: 'var(--text-main)', fontWeight: 500 }}>{formatBytes(hwInfo.ram_total)} RAM</div>
                            </div>
                        </div>

                        {/* GPUs */}
                        {hwInfo.gpus.map((gpu, idx) => (
                            <div key={idx} className="list-item" style={{ borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)', gap: '16px' }}>
                                <div style={{ padding: '10px', background: 'rgba(59,130,246,0.1)', borderRadius: '10px', color: 'var(--accent-blue)' }}>
                                    <HardDrive size={20} />
                                </div>
                                <div style={{ flex: 1 }}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2px' }}>
                                        <div style={{ fontSize: '10px', color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 700 }}>Graphics Adapter {idx + 1}</div>
                                        {idx === 0 && <span className="badge badge-primary">PRIMARY</span>}
                                    </div>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                        <div style={{ color: 'var(--text-main)', fontWeight: 500 }}>{gpu.name}</div>
                                        <div style={{ color: 'var(--accent-blue)', fontFamily: 'var(--font-mono)', fontSize: '0.875rem' }}>{formatBytes(gpu.vram_total)} VRAM</div>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </Card>

            <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                <Button variant="ghost" style={{ color: 'var(--accent-red)' }} leftIcon={<RotateCcw size={16} />} onClick={() => {
                    if (confirm("Are you sure you want to reset all settings to defaults?")) {
                        setAutoUpdate(true);
                        setShowInfoTooltips(true);
                    }
                }}>
                    Reset All Settings
                </Button>
            </div>

            <Card>
                <div style={{ borderLeft: '4px solid var(--accent-red)', paddingLeft: '16px' }}>
                    <h2 className="text-xl font-bold mb-2" style={{ color: 'var(--accent-red)' }}>Advanced Troubleshoot</h2>
                    <p className="text-sm text-secondary mb-6">Only use these if the application environment is corrupted or failing to start.</p>
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                    <div style={{ padding: '20px', background: 'rgba(239, 68, 68, 0.05)', border: '1px solid rgba(239, 68, 68, 0.1)', borderRadius: 'var(--radius-lg)' }}>
                        <h3 style={{ fontWeight: 500, color: 'var(--accent-red)', marginBottom: '8px' }}>Repair Environment</h3>
                        <p className="text-sm text-secondary mb-6">
                            Forcing a reinstallation will clear the current Python environment and redownload all dependencies.
                        </p>
                        <div style={{ display: 'flex', gap: '12px' }}>
                            <Button variant="danger" onClick={() => {
                                if (confirm("Are you sure you want to reinstall the Python runtime? This will delete the current installation.")) {
                                    onReinstallPython?.();
                                }
                            }}>
                                Reinstall Runtime
                            </Button>
                            <Button variant="secondary" onClick={() => {
                                if (confirm("Are you sure you want to reinstall dependencies? This may take a while.")) {
                                    onReinstallDependencies?.();
                                }
                            }}>
                                Force Dependencies Update
                            </Button>
                        </div>
                    </div>
                </div>
            </Card>
        </div>
    );
};

export default SettingsPage;
