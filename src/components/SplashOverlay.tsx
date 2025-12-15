import React, { useEffect, useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";

interface SplashOverlayProps {
    isVisible: boolean;
    progress: number; // 0-100
    message: string;
    loadedBytes?: number;
    totalBytes?: number;
}

export const SplashOverlay: React.FC<SplashOverlayProps> = ({
    isVisible,
    progress,
    message,
    loadedBytes = 0,
    totalBytes = 0,
}) => {
    const [dots, setDots] = useState("");
    const [speed, setSpeed] = useState("0 MB/s");
    const [eta, setEta] = useState("");

    // Speed calculation state
    const lastUpdateRef = useRef<number>(Date.now());
    const lastBytesRef = useRef<number>(0);
    const speedBufferRef = useRef<number[]>([]); // Rolling average buffer

    // Animated dots for "Downloading..."
    useEffect(() => {
        if (!isVisible) return;
        const interval = setInterval(() => {
            setDots((prev) => (prev.length >= 3 ? "" : prev + "."));
        }, 500);
        return () => clearInterval(interval);
    }, [isVisible]);

    // Speed and ETA calculation
    useEffect(() => {
        if (!isVisible || totalBytes === 0 || loadedBytes === 0) {
            setSpeed("");
            setEta("");
            return;
        }

        const now = Date.now();
        const timeDelta = (now - lastUpdateRef.current) / 1000; // seconds

        // Calculate every ~500ms to avoid jitter
        if (timeDelta > 0.5) {
            const bytesDelta = loadedBytes - lastBytesRef.current;
            const currentSpeed = bytesDelta / timeDelta; // bytes/sec

            // Rolling average for smoothness
            speedBufferRef.current.push(currentSpeed);
            if (speedBufferRef.current.length > 5) speedBufferRef.current.shift();

            const avgSpeed = speedBufferRef.current.reduce((a, b) => a + b, 0) / speedBufferRef.current.length;

            // Speed String
            const speedMb = (avgSpeed / (1024 * 1024)).toFixed(1);
            setSpeed(`${speedMb} MB/s`);

            // ETA
            const remainingBytes = totalBytes - loadedBytes;
            if (avgSpeed > 0) {
                const secondsRemaining = Math.max(0, remainingBytes / avgSpeed);
                if (secondsRemaining < 60) {
                    setEta(`${Math.round(secondsRemaining)}s remaining`);
                } else {
                    setEta(`${Math.ceil(secondsRemaining / 60)}m remaining`);
                }
            }

            lastUpdateRef.current = now;
            lastBytesRef.current = loadedBytes;
        }
    }, [loadedBytes, totalBytes, isVisible]);

    // Reset on close
    useEffect(() => {
        if (!isVisible) {
            setSpeed("");
            setEta("");
            speedBufferRef.current = [];
            lastBytesRef.current = 0;
        }
    }, [isVisible]);

    return (
        <AnimatePresence>
            {isVisible && (
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    style={{
                        position: "fixed",
                        inset: 0,
                        zIndex: 9999,
                        backgroundColor: "rgba(5, 5, 8, 0.85)",
                        backdropFilter: "blur(20px)",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                    }}
                >
                    <div className="flex flex-col items-center max-w-md w-full px-8 text-center space-y-8">
                        {/* Logo or Icon Animation */}
                        <motion.div
                            initial={{ scale: 0.8, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            transition={{ delay: 0.2 }}
                            className="relative w-24 h-24 mb-6 flex items-center justify-center"
                        >
                            {/* Spinning Ring */}
                            <div className="absolute inset-0 rounded-full border-4 border-accent-primary/20 border-t-accent-primary animate-spin" />

                            {/* Inner Pulse */}
                            <div className="absolute inset-4 rounded-full bg-accent-primary/10 animate-pulse" />

                            {/* Icon */}
                            <svg
                                viewBox="0 0 24 24"
                                fill="none"
                                stroke="currentColor"
                                className="w-10 h-10 text-accent-primary relative z-10"
                                strokeWidth={2}
                            >
                                <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
                            </svg>
                        </motion.div>

                        {/* Main Message */}
                        <motion.h2
                            initial={{ y: 20, opacity: 0 }}
                            animate={{ y: 0, opacity: 1 }}
                            transition={{ delay: 0.3 }}
                            className="text-2xl font-bold text-white tracking-wide"
                        >
                            Setting Up Environment
                        </motion.h2>

                        {/* Progress Bar Container */}
                        <div className="w-full space-y-2">
                            <div className="flex justify-between text-xs font-mono text-gray-400 uppercase tracking-wider">
                                <span>{message}{dots}</span>
                                <span>{Math.round(progress)}%</span>
                            </div>

                            <div className="h-2 w-full bg-white/10 rounded-full overflow-hidden relative">
                                {/* Shimmer Effect Background */}
                                <div className="absolute inset-0 opacity-20"
                                    style={{
                                        background: "linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent)",
                                        backgroundSize: "200% 100%",
                                        animation: "shimmer 2s infinite linear"
                                    }}
                                />

                                <motion.div
                                    className="h-full bg-accent-primary relative"
                                    initial={{ width: 0 }}
                                    animate={{ width: `${progress}%` }}
                                    transition={{ type: "spring", stiffness: 50, damping: 15 }}
                                >
                                    {/* Glow at the tip of progress bar */}
                                    <div className="absolute right-0 top-0 bottom-0 w-2 bg-white/50 blur-[4px]" />
                                </motion.div>
                            </div>

                            {/* Stats Row */}
                            <div className="flex justify-between items-center h-4 text-xs font-mono text-accent-primary/80">
                                <span>{speed}</span>
                                <span>{eta}</span>
                            </div>
                        </div>

                        {/* Footer Hint */}
                        <motion.p
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            transition={{ delay: 1.5, duration: 2 }}
                            className="text-xs text-gray-600 mt-8 max-w-xs"
                        >
                            This one-time setup ensures optimal performance for local inference.
                        </motion.p>
                    </div>
                </motion.div>
            )}
        </AnimatePresence>
    );
};
