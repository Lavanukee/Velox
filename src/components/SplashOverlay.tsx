import React, { useEffect, useState, useRef, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useApp } from "../context/AppContext";
import { CyberBackground, ForgeBackground } from "./ThemeBackground";

interface SplashOverlayProps {
    isVisible: boolean;
    progress: number; // 0-100
    message: string;
    loadedBytes?: number;
    totalBytes?: number;
    onComplete?: () => void;
}

// Configuration for the Installation Steps
const STEPS = [
    { id: 'python', label: 'PYTHON RUNTIME', sub_default: 'WAITING FOR INITIALIZATION' },
    { id: 'torch', label: 'ML LIBRARIES', sub_default: 'PENDING REPOSITORY FETCH' },
    { id: 'cuda', label: 'ENVIRONMENT SETUP', sub_default: 'PENDING CONFIGURATION' },
    { id: 'deps', label: 'GPU ACCELERATION', sub_default: 'WAITING FOR HARDWARE CHECK' },
];

const StepItem: React.FC<{
    label: string,
    sub: string,
    status: 'pending' | 'active' | 'complete'
}> = ({ label, sub, status }) => {
    const isActive = status === 'active';
    const isComplete = status === 'complete';

    return (
        <motion.li
            animate={{ opacity: isActive ? 1 : isComplete ? 0.6 : 0.3 }}
            style={{
                display: 'flex', alignItems: 'center', marginBottom: '1.5rem',
                color: isComplete ? 'var(--accent-secondary)' : isActive ? 'var(--text-main)' : 'var(--text-muted)',
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: '0.7rem'
            }}
        >
            <motion.div
                animate={isActive ? {
                    scale: [1, 1.4, 1],
                    opacity: [1, 0.5, 1],
                    background: 'var(--accent-primary)',
                    boxShadow: '0 0 10px var(--accent-primary)'
                } : { scale: 1, opacity: 1, background: 'transparent' }}
                transition={isActive ? { duration: 1.5, repeat: Infinity } : {}}
                style={{
                    width: 8, height: 8, border: '1px solid currentColor',
                    marginRight: '1.5rem', transform: 'rotate(45deg)',
                }}
            />
            <div>
                <div style={{ fontWeight: 500, textTransform: 'uppercase' }}>{label}</div>
                <div style={{ fontSize: '0.5rem', opacity: 0.7, textTransform: 'uppercase' }}>{sub}</div>
            </div>
        </motion.li>
    );
};

export const SplashOverlay: React.FC<SplashOverlayProps> = ({
    isVisible,
    progress,
    message,
    loadedBytes = 0,
    totalBytes = 0,
    onComplete
}) => {
    const { theme } = useApp();
    const [speed, setSpeed] = useState("CALCULATING...");
    const [eta, setEta] = useState("--:--");
    const [smoothProgress, setSmoothProgress] = useState(0);
    const [isComplete, setIsComplete] = useState(false);

    const lastUpdateRef = useRef<number>(Date.now());
    const lastBytesRef = useRef<number>(0);

    useEffect(() => {
        if (!isVisible || totalBytes === 0) return;
        const now = Date.now();
        const timeDelta = (now - lastUpdateRef.current) / 1000;

        if (timeDelta > 0.5) {
            const bytesDelta = loadedBytes - lastBytesRef.current;
            const avgSpeed = bytesDelta / timeDelta;

            setSpeed(avgSpeed > 1024 * 1024
                ? `${(avgSpeed / (1024 * 1024)).toFixed(1)} MB/S`
                : `${(avgSpeed / 1024).toFixed(0)} KB/S`
            );

            if (avgSpeed > 0) {
                const sec = (totalBytes - loadedBytes) / avgSpeed;
                setEta(sec < 60 ? `00:${sec.toFixed(0).padStart(2, '0')}` : `${Math.ceil(sec / 60)} MIN`);
            }
            lastUpdateRef.current = now;
            lastBytesRef.current = loadedBytes;
        }
    }, [loadedBytes, totalBytes, isVisible]);

    useEffect(() => {
        if (!isVisible) return;

        let animationFrame: number;
        let lastTime = Date.now();

        const animate = () => {
            const now = Date.now();
            const delta = now - lastTime;
            lastTime = now;

            setSmoothProgress(prev => {
                const target = progress;
                if (prev < target) {
                    const step = Math.max(0.1, (target - prev) * 0.05);
                    return Math.min(target, prev + step);
                }
                if (prev < 99 && progress < 100) {
                    return prev + 0.01 * (delta / 16.6);
                }
                return prev;
            });

            if (progress >= 100 && smoothProgress >= 99.9) {
                setIsComplete(true);
                return;
            }

            animationFrame = requestAnimationFrame(animate);
        };

        animationFrame = requestAnimationFrame(animate);
        return () => cancelAnimationFrame(animationFrame);
    }, [progress, isVisible, smoothProgress]);

    useEffect(() => {
        if (isComplete && isVisible && onComplete) {
            const timer = setTimeout(() => {
                onComplete();
            }, 1000);
            return () => clearTimeout(timer);
        }
    }, [isComplete, isVisible, onComplete]);

    const formatBytes = (bytes: number) => {
        if (bytes === 0) return "0.0 GB";
        const gb = bytes / (1024 * 1024 * 1024);
        return `${gb.toFixed(2)} GB`;
    };

    const currentStepIndex = useMemo(() => {
        const lowerMsg = message?.toLowerCase() || "";
        if (progress >= 100) return 4;
        if (lowerMsg.includes('python') || lowerMsg.includes('conda')) return 0;
        if (lowerMsg.includes('torch') || lowerMsg.includes('vision') || lowerMsg.includes('artifact')) return 1;
        if (lowerMsg.includes('cuda') || lowerMsg.includes('cudnn')) return 3;
        if (lowerMsg.includes('env') || lowerMsg.includes('virt')) return 2;
        return 0;
    }, [message, progress]);

    return (
        <AnimatePresence>
            {isVisible && (
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0, scale: 1.1, filter: "blur(20px)" }}
                    style={{
                        position: 'fixed', inset: 0, zIndex: 99999,
                        background: 'var(--bg-app)',
                        display: 'flex', alignItems: 'center', justifyContent: 'center'
                    }}
                >
                    {theme === 'forge' ? <ForgeBackground /> : <CyberBackground />}

                    <main style={{
                        position: 'relative', zIndex: 10, width: '100%', maxWidth: 1000,
                        padding: '4rem', display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '4rem'
                    }}>
                        <section style={{ borderLeft: `2px solid ${theme === 'forge' ? '#b87333' : 'var(--accent-primary)'}`, paddingLeft: '2.5rem' }}>
                            <p style={{
                                fontFamily: "'JetBrains Mono', monospace", textTransform: 'uppercase',
                                fontSize: '0.8rem', letterSpacing: 4, color: 'var(--accent-secondary)'
                            }}>
                                Initializing System
                            </p>

                            <AnimatePresence mode="wait">
                                {!isComplete ? (
                                    <motion.h1
                                        key="velox-title"
                                        initial={{ opacity: 0, x: -20 }}
                                        animate={{ opacity: 1, x: 0 }}
                                        exit={{ opacity: 0, filter: 'blur(10px)', y: -20 }}
                                        style={{
                                            fontSize: '5rem', lineHeight: 0.85, textTransform: 'uppercase',
                                            letterSpacing: -4, fontWeight: 900,
                                            margin: '1rem 0',
                                            background: 'var(--accent-gradient)',
                                            WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
                                            filter: `drop-shadow(0 0 20px ${theme === 'forge' ? 'rgba(184, 115, 51, 0.3)' : 'rgba(139, 92, 246, 0.3)'})`
                                        }}
                                    >
                                        Velox<br />Engine
                                    </motion.h1>
                                ) : (
                                    <motion.div
                                        key="final-logo"
                                        initial={{ opacity: 0, scale: 0.8, filter: 'blur(20px)' }}
                                        animate={{ opacity: 1, scale: 1, filter: 'blur(0px)' }}
                                        style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}
                                    >
                                        <div style={{ position: 'relative', width: 120, height: 120, margin: '2rem 0' }}>
                                            <svg viewBox="0 0 100 100" style={{ width: '100%', height: '100%', filter: `drop-shadow(0 0 30px ${theme === 'forge' ? '#b87333' : 'var(--accent-primary)'})` }}>
                                                <motion.path
                                                    d="M10 20 L50 90 L90 20"
                                                    fill="none"
                                                    stroke={theme === 'forge' ? '#e3a87c' : 'var(--accent-primary)'}
                                                    strokeWidth="8"
                                                    initial={{ pathLength: 0 }}
                                                    animate={{ pathLength: 1 }}
                                                    transition={{ duration: 1.5, ease: "easeInOut" }}
                                                />
                                                <motion.path
                                                    d="M25 20 L50 65 L75 20"
                                                    fill="none"
                                                    stroke={theme === 'forge' ? '#b87333' : 'var(--accent-secondary)'}
                                                    strokeWidth="4"
                                                    initial={{ pathLength: 0 }}
                                                    animate={{ pathLength: 1 }}
                                                    transition={{ duration: 1.5, delay: 0.5, ease: "easeInOut" }}
                                                />
                                            </svg>
                                        </div>
                                        <motion.h1
                                            initial={{ y: 20, opacity: 0 }}
                                            animate={{ y: 0, opacity: 1 }}
                                            transition={{ delay: 0.8 }}
                                            style={{
                                                fontSize: '6rem', lineHeight: 0.85, textTransform: 'uppercase',
                                                letterSpacing: -6, fontWeight: 900, margin: 0,
                                                color: 'var(--accent-primary)', textShadow: `0 0 40px ${theme === 'forge' ? 'rgba(184, 115, 51, 0.5)' : 'rgba(139, 92, 246, 0.5)'}`
                                            }}
                                        >
                                            VELOX
                                        </motion.h1>
                                    </motion.div>
                                )}
                            </AnimatePresence>

                            <div style={{ marginTop: '2rem', opacity: 0.4 }}>
                                <svg width="120" height="2" viewBox="0 0 120 2" fill="none">
                                    <rect width="120" height="2" fill="url(#paint0_linear)" />
                                    <defs>
                                        <linearGradient id="paint0_linear" x1="0" y1="0" x2="120" y2="0" gradientUnits="userSpaceOnUse">
                                            <stop stopColor="#b87333" />
                                            <stop offset="1" stopColor="#b87333" stopOpacity="0" />
                                        </linearGradient>
                                    </defs>
                                </svg>
                            </div>
                        </section>

                        <section style={{ display: 'flex', flexDirection: 'column', justifyContent: 'flex-end', width: 340 }}>
                            <motion.ul
                                animate={isComplete ? { opacity: 0, x: 20 } : { opacity: 1, x: 0 }}
                                style={{ listStyle: 'none', padding: 0, margin: '0 0 2rem 0' }}
                            >
                                {STEPS.map((step, idx) => {
                                    const status = idx < currentStepIndex ? 'complete' : idx === currentStepIndex ? 'active' : 'pending';
                                    const isCurrent = idx === currentStepIndex;
                                    return (
                                        <StepItem
                                            key={step.id}
                                            label={step.label}
                                            sub={isCurrent ? (message || 'PROCESSING...') : (idx < currentStepIndex ? 'COMPLETE' : step.sub_default)}
                                            status={status}
                                        />
                                    );
                                })}
                            </motion.ul>

                            <div style={{ height: 2, background: '#5d3a1a', width: '100%', position: 'relative', marginBottom: isComplete ? '2rem' : 0 }}>
                                <motion.div
                                    animate={{ width: `${smoothProgress}%` }}
                                    transition={{ type: 'spring', damping: 25, stiffness: 50 }}
                                    style={{
                                        position: 'absolute', left: 0, top: -2, height: 6,
                                        background: isComplete ? 'var(--accent-primary)' : 'var(--accent-secondary)',
                                        boxShadow: isComplete ? '0 0 30px var(--accent-primary)' : '0 0 20px rgba(0,0,0,0.5)'
                                    }}
                                />
                            </div>

                            <div style={{
                                display: 'flex', justifyContent: 'space-between', marginTop: '1rem',
                                fontFamily: "'JetBrains Mono', monospace", fontSize: '0.65rem', color: '#b87333'
                            }}>
                                <span style={{ textTransform: 'uppercase' }}>
                                    {isComplete ? 'SYSTEM_READY' : (message || 'INITIALIZING...')}
                                </span>
                                <span>{smoothProgress.toFixed(1)}%</span>
                            </div>

                            <AnimatePresence>
                                {totalBytes > 0 && !isComplete && (
                                    <motion.div
                                        initial={{ opacity: 0, y: 10 }}
                                        animate={{ opacity: 0.6, y: 0 }}
                                        exit={{ opacity: 0, y: 10 }}
                                        style={{
                                            display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem',
                                            marginTop: '2rem', borderTop: '1px solid rgba(184, 115, 51, 0.1)',
                                            paddingTop: '1rem', fontSize: '0.6rem',
                                            fontFamily: "'JetBrains Mono', monospace", color: '#e3a87c'
                                        }}
                                    >
                                        <div>
                                            <div style={{ opacity: 0.5 }}>DOWNLOAD.RATE</div>
                                            <div>{speed}</div>
                                        </div>
                                        <div>
                                            <div style={{ opacity: 0.5 }}>TIME.REMAINING</div>
                                            <div>{eta}</div>
                                        </div>
                                        <div style={{ gridColumn: 'span 2' }}>
                                            <div style={{ opacity: 0.5 }}>DATA.SYNC</div>
                                            <div>{formatBytes(loadedBytes)} / {formatBytes(totalBytes)}</div>
                                        </div>
                                    </motion.div>
                                )}
                            </AnimatePresence>
                        </section>
                    </main>
                </motion.div>
            )}
        </AnimatePresence>
    );
};
