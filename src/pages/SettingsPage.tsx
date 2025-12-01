import React, { useState, useEffect } from 'react';
import { Card } from '../components/Card';
import { Button } from '../components/Button';
import { Input } from '../components/Input';
import { Toggle } from '../components/Toggle';
import { Save, Trash2, RotateCcw } from 'lucide-react';

interface SettingsPageProps {
    onReinstallPython?: () => void;
    onReinstallDependencies?: () => void;
}

const SettingsPage: React.FC<SettingsPageProps> = ({ onReinstallPython, onReinstallDependencies }) => {
    const [profiles, setProfiles] = useState<{ name: string, id: string }[]>([]);
    const [newProfileName, setNewProfileName] = useState('');

    useEffect(() => {
        // Mock loading profiles
        const saved = localStorage.getItem('velox_profiles');
        if (saved) {
            setProfiles(JSON.parse(saved));
        }
    }, []);

    const handleSaveProfile = () => {
        if (!newProfileName.trim()) return;
        const newProfile = { name: newProfileName, id: Date.now().toString() };
        const updated = [...profiles, newProfile];
        setProfiles(updated);
        localStorage.setItem('velox_profiles', JSON.stringify(updated));
        setNewProfileName('');
    };

    const handleDeleteProfile = (id: string) => {
        const updated = profiles.filter(p => p.id !== id);
        setProfiles(updated);
        localStorage.setItem('velox_profiles', JSON.stringify(updated));
    };

    return (
        <div className="space-y-6 max-w-3xl mx-auto" style={{ maxWidth: '800px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '24px' }}>
            <Card>
                <h2 className="text-xl font-bold mb-4">Application Settings</h2>
                <div className="space-y-6">
                    <div className="flex items-center justify-between p-4 bg-white/5 rounded-lg" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '16px', background: 'rgba(255,255,255,0.05)', borderRadius: '8px' }}>
                        <div>
                            <h3 className="font-medium">Hardware Acceleration</h3>
                            <p className="text-sm text-gray-400">Enable GPU acceleration for UI rendering</p>
                        </div>
                        <Toggle checked={true} onChange={() => { }} />
                    </div>

                    <div className="flex items-center justify-between p-4 bg-white/5 rounded-lg" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '16px', background: 'rgba(255,255,255,0.05)', borderRadius: '8px' }}>
                        <div>
                            <h3 className="font-medium">Auto-Update</h3>
                            <p className="text-sm text-gray-400">Automatically check for updates on startup</p>
                        </div>
                        <Toggle checked={false} onChange={() => { }} />
                    </div>
                </div>
            </Card>

            <Card>
                <h2 className="text-xl font-bold mb-4">User Profiles</h2>
                <p className="text-gray-400 mb-6">Save your current configuration as a preset profile.</p>

                <div className="flex gap-3 mb-6" style={{ display: 'flex', gap: '12px', marginBottom: '24px' }}>
                    <Input
                        placeholder="Profile Name"
                        value={newProfileName}
                        onChange={(e) => setNewProfileName(e.target.value)}
                        className="flex-1"
                    />
                    <Button onClick={handleSaveProfile} leftIcon={<Save size={16} />}>
                        Save Current State
                    </Button>
                </div>

                <div className="space-y-3" style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                    {profiles.length === 0 && (
                        <p className="text-center text-gray-500 py-4">No saved profiles yet.</p>
                    )}
                    {profiles.map(profile => (
                        <div key={profile.id} className="flex items-center justify-between p-3 bg-white/5 rounded-lg border border-white/5" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '12px', background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.05)', borderRadius: '8px' }}>
                            <span className="font-medium">{profile.name}</span>
                            <div className="flex gap-2" style={{ display: 'flex', gap: '8px' }}>
                                <Button size="sm" variant="secondary" onClick={() => { }}>Load</Button>
                                <Button size="sm" variant="danger" onClick={() => handleDeleteProfile(profile.id)}><Trash2 size={14} /></Button>
                            </div>
                        </div>
                    ))}
                </div>
            </Card>

            <div className="flex justify-end" style={{ display: 'flex', justifyContent: 'flex-end' }}>
                <Button variant="ghost" className="text-red-400 hover:text-red-300" leftIcon={<RotateCcw size={16} />}>
                    Reset All Settings
                </Button>
            </div>

            <Card>
                <h2 className="text-xl font-bold mb-4 text-red-400">Advanced</h2>
                <div className="space-y-4">
                    <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-lg">
                        <h3 className="font-medium text-red-200 mb-2">Danger Zone</h3>
                        <p className="text-sm text-gray-400 mb-4">
                            Use these options if you are experiencing issues with the Python environment.
                        </p>
                        <div className="flex gap-4">
                            <Button variant="danger" onClick={() => {
                                if (confirm("Are you sure you want to reinstall the Python runtime? This will delete the current installation.")) {
                                    onReinstallPython?.();
                                }
                            }}>
                                Reinstall Python Runtime
                            </Button>
                            <Button variant="secondary" onClick={() => {
                                if (confirm("Are you sure you want to reinstall dependencies? This may take a while.")) {
                                    onReinstallDependencies?.();
                                }
                            }}>
                                Reinstall Dependencies
                            </Button>
                        </div>
                    </div>
                </div>
            </Card>
        </div>
    );
};

export default SettingsPage;
