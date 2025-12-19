import React from 'react';
import { Card } from '../components/Card';
import { Button } from '../components/Button';
import { Toggle } from '../components/Toggle';
import { RotateCcw, Info, RefreshCw } from 'lucide-react';
import { useApp } from '../context/AppContext';

interface SettingsPageProps {
    onReinstallPython?: () => void;
    onReinstallDependencies?: () => void;
}

const SettingsPage: React.FC<SettingsPageProps> = ({ onReinstallPython, onReinstallDependencies }) => {
    const {
        autoUpdate, setAutoUpdate,
        showInfoTooltips, setShowInfoTooltips
    } = useApp();

    return (
        <div className="space-y-6 max-w-3xl mx-auto" style={{ maxWidth: '800px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '24px' }}>
            <Card>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '24px' }}>
                    <h2 className="text-xl font-bold">General Settings</h2>
                </div>

                <div className="space-y-6" style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                    <div className="flex items-center justify-between p-4 bg-white/5 rounded-lg border border-white/5 hover:border-white/10 transition-colors" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '20px', background: 'rgba(255,255,255,0.03)', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                            <div style={{ padding: '10px', background: 'rgba(59,130,246,0.1)', borderRadius: '10px', color: '#3b82f6' }}>
                                <RefreshCw size={20} />
                            </div>
                            <div>
                                <h3 className="font-semibold text-gray-100">Auto-Update</h3>
                                <p className="text-sm text-gray-400">Keep your AI engines and environment up to date automatically.</p>
                            </div>
                        </div>
                        <Toggle checked={autoUpdate} onChange={setAutoUpdate} />
                    </div>

                    <div className="flex items-center justify-between p-4 bg-white/5 rounded-lg border border-white/5 hover:border-white/10 transition-colors" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '20px', background: 'rgba(255,255,255,0.03)', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                            <div style={{ padding: '10px', background: 'rgba(16,185,129,0.1)', borderRadius: '10px', color: '#10b981' }}>
                                <Info size={20} />
                            </div>
                            <div>
                                <h3 className="font-semibold text-gray-100">Show Helper Tooltips</h3>
                                <p className="text-sm text-gray-400">Display (i) icons throughout the app for helpful information.</p>
                            </div>
                        </div>
                        <Toggle checked={showInfoTooltips} onChange={setShowInfoTooltips} />
                    </div>
                </div>
            </Card>

            <div className="flex justify-end" style={{ display: 'flex', justifyContent: 'flex-end' }}>
                <Button variant="ghost" className="text-red-400 hover:text-red-300" leftIcon={<RotateCcw size={16} />} onClick={() => {
                    if (confirm("Are you sure you want to reset all settings to defaults?")) {
                        setAutoUpdate(true);
                        setShowInfoTooltips(true);
                    }
                }}>
                    Reset All Settings
                </Button>
            </div>

            <Card>
                <div style={{ borderLeft: '4px solid #ef4444', paddingLeft: '16px' }}>
                    <h2 className="text-xl font-bold mb-2 text-red-400">Advanced Troubleshoot</h2>
                    <p className="text-sm text-gray-400 mb-6">Only use these if the application environment is corrupted or failing to start.</p>
                </div>

                <div className="space-y-4">
                    <div className="p-5 bg-red-500/5 border border-red-500/10 rounded-xl">
                        <h3 className="font-medium text-red-200 mb-2">Repair Environment</h3>
                        <p className="text-sm text-gray-400 mb-6">
                            Forcing a reinstallation will clear the current Python environment and redownload all dependencies.
                        </p>
                        <div className="flex gap-4" style={{ display: 'flex', gap: '12px' }}>
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
