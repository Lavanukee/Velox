import React from 'react';
import { motion } from 'framer-motion';

export const CyberBackground: React.FC = () => (
    <div style={{
        position: 'absolute', inset: 0, overflow: 'hidden', zIndex: 0, pointerEvents: 'none',
        background: 'radial-gradient(circle at center, #1a1a2e 0%, #0a0a12 100%)'
    }}>
        <div style={{
            position: 'absolute', inset: 0,
            backgroundImage: `linear-gradient(rgba(139, 92, 246, 0.1) 1px, transparent 1px), 
                              linear-gradient(90deg, rgba(139, 92, 246, 0.1) 1px, transparent 1px)`,
            backgroundSize: '40px 40px',
            maskImage: 'radial-gradient(circle at center, black 0%, transparent 80%)'
        }} />
        <motion.div
            animate={{
                scale: [1, 1.2, 1],
                opacity: [0.1, 0.2, 0.1]
            }}
            transition={{ duration: 10, repeat: Infinity, ease: "linear" }}
            style={{
                position: 'absolute', inset: 0,
                background: 'radial-gradient(circle at 50% 50%, rgba(139, 92, 246, 0.15), transparent 60%)'
            }}
        />
    </div>
);

export const ForgeBackground: React.FC = () => (
    <div style={{ position: 'absolute', inset: 0, overflow: 'hidden', zIndex: 0, pointerEvents: 'none' }}>
        <div className="grain" style={{
            position: 'absolute', inset: 0, zIndex: 99, pointerEvents: 'none', opacity: 0.04,
            backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)'/%3E%3C/svg%3E")`
        }} />
        <div className="ortho-canvas" style={{
            position: 'absolute', inset: 0, zIndex: 1, pointerEvents: 'none',
            backgroundImage: `linear-gradient(rgba(184, 115, 51, 0.1) 1px, transparent 1px), 
                              linear-gradient(90deg, rgba(184, 115, 51, 0.1) 1px, transparent 1px)`,
            backgroundSize: '60px 60px', backgroundPosition: 'center center'
        }} />
        <div className="vignette" style={{
            position: 'absolute', inset: 0, zIndex: 2, pointerEvents: 'none',
            background: 'radial-gradient(circle at center, transparent 0%, rgba(10, 10, 12, 0.8) 100%)'
        }} />
        <motion.div
            animate={{ top: ['0%', '100%'] }}
            transition={{ duration: 12, repeat: Infinity, ease: "linear" }}
            style={{
                position: 'absolute', left: 0, width: '100%', height: '2px', zIndex: 5,
                background: 'linear-gradient(90deg, transparent, var(--accent-primary), transparent)',
                opacity: 0.15
            }}
        />
    </div>
);

export const GlobalBackground: React.FC<{ theme: 'cyber' | 'forge' }> = ({ theme }) => {
    return (
        <div style={{
            position: 'fixed', inset: 0, zIndex: -1,
            background: 'var(--bg-app)',
            overflow: 'hidden'
        }}>
            {theme === 'forge' ? <ForgeBackground /> : <CyberBackground />}
        </div>
    );
};
