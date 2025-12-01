import React, { useEffect, useState } from 'react';
import { BrainCircuit } from 'lucide-react';

interface SplashScreenProps {
    onComplete: () => void;
}

export const SplashScreen: React.FC<SplashScreenProps> = ({ onComplete }) => {
    const [progress, setProgress] = useState(0);
    const [status, setStatus] = useState('Initializing...');

    useEffect(() => {
        // Simulate loading phases
        const phases = [
            { duration: 300, progress: 30, status: 'Loading resources...' },
            { duration: 400, progress: 60, status: 'Preparing environment...' },
            { duration: 300, progress: 90, status: 'Almost ready...' },
            { duration: 200, progress: 100, status: 'Ready!' },
        ];

        let currentPhase = 0;
        const runPhase = () => {
            if (currentPhase >= phases.length) {
                setTimeout(onComplete, 200);
                return;
            }

            const phase = phases[currentPhase];
            setProgress(phase.progress);
            setStatus(phase.status);
            currentPhase++;
            setTimeout(runPhase, phase.duration);
        };

        runPhase();
    }, [onComplete]);

    return (
        <div
            style={{
                position: 'fixed',
                inset: 0,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                background: 'linear-gradient(135deg, #0a0a0c 0%, #1a1a24 50%, #0a0a0c 100%)',
                zIndex: 9999,
                animation: 'fadeIn 0.3s ease-out'
            }}
        >
            {/* Animated background particles */}
            <div style={{
                position: 'absolute',
                inset: 0,
                overflow: 'hidden',
                opacity: 0.4
            }}>
                {[...Array(20)].map((_, i) => (
                    <div
                        key={i}
                        style={{
                            position: 'absolute',
                            width: `${Math.random() * 4 + 2}px`,
                            height: `${Math.random() * 4 + 2}px`,
                            borderRadius: '50%',
                            background: i % 2 === 0 ? '#a78bfa' : '#7dd3fc',
                            left: `${Math.random() * 100}%`,
                            top: `${Math.random() * 100}%`,
                            animation: `float ${Math.random() * 3 + 2}s ease-in-out infinite`,
                            animationDelay: `${Math.random() * 2}s`,
                            boxShadow: `0 0 ${Math.random() * 20 + 10}px currentColor`
                        }}
                    />
                ))}
            </div>

            {/* Main content */}
            <div style={{
                textAlign: 'center',
                position: 'relative',
                zIndex: 1
            }}>
                {/* Logo with glow effect */}
                <div style={{
                    width: '120px',
                    height: '120px',
                    margin: '0 auto 32px',
                    borderRadius: '24px',
                    background: 'linear-gradient(135deg, #a78bfa 0%, #7dd3fc 100%)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    position: 'relative',
                    animation: 'pulse 2s ease-in-out infinite',
                    boxShadow: `
                        0 0 60px rgba(167, 139, 250, 0.5),
                        0 0 120px rgba(125, 211, 252, 0.3),
                        inset 0 0 60px rgba(255, 255, 255, 0.1)
                    `
                }}>
                    <BrainCircuit size={64} color="white" strokeWidth={1.5} />
                </div>

                {/* App name */}
                <h1 style={{
                    fontSize: '48px',
                    fontWeight: 700,
                    background: 'linear-gradient(135deg, #a78bfa 0%, #7dd3fc 100%)',
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent',
                    backgroundClip: 'text',
                    marginBottom: '12px',
                    letterSpacing: '-0.02em'
                }}>
                    Velox
                </h1>

                <p style={{
                    fontSize: '16px',
                    color: '#9ca3af',
                    marginBottom: '48px',
                    fontWeight: 500
                }}>
                    AI Fine-Tuning Platform
                </p>

                {/* Loading bar */}
                <div style={{
                    width: '320px',
                    margin: '0 auto',
                    position: 'relative'
                }}>
                    <div style={{
                        height: '4px',
                        background: 'rgba(255, 255, 255, 0.1)',
                        borderRadius: '2px',
                        overflow: 'hidden',
                        position: 'relative'
                    }}>
                        <div style={{
                            height: '100%',
                            background: 'linear-gradient(90deg, #a78bfa 0%, #7dd3fc 100%)',
                            width: `${progress}%`,
                            transition: 'width 0.3s ease-out',
                            boxShadow: '0 0 20px rgba(167, 139, 250, 0.6)',
                            position: 'relative'
                        }}>
                            {/* Shimmer effect */}
                            <div style={{
                                position: 'absolute',
                                inset: 0,
                                background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent)',
                                animation: 'shimmer 1.5s infinite'
                            }} />
                        </div>
                    </div>

                    <p style={{
                        marginTop: '12px',
                        fontSize: '13px',
                        color: '#6b7280',
                        fontWeight: 500
                    }}>
                        {status}
                    </p>
                </div>
            </div>

            <style>{`
                @keyframes fadeIn {
                    from { opacity: 0; }
                    to { opacity: 1; }
                }

                @keyframes pulse {
                    0%, 100% {
                        transform: scale(1);
                        opacity: 1;
                    }
                    50% {
                        transform: scale(1.05);
                        opacity: 0.9;
                    }
                }

                @keyframes float {
                    0%, 100% {
                        transform: translateY(0) translateX(0);
                    }
                    33% {
                        transform: translateY(-20px) translateX(10px);
                    }
                    66% {
                        transform: translateY(-10px) translateX(-10px);
                    }
                }

                @keyframes shimmer {
                    0% { transform: translateX(-100%); }
                    100% { transform: translateX(100%); }
                }
            `}</style>
        </div>
    );
};
