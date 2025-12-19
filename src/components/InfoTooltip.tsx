import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Info } from 'lucide-react';
import { useApp } from '../context/AppContext';

interface InfoTooltipProps {
    text: string;
    position?: 'top' | 'bottom' | 'left' | 'right';
}

export const InfoTooltip: React.FC<InfoTooltipProps> = ({ text, position = 'top' }) => {
    const { showInfoTooltips } = useApp();
    const [isVisible, setIsVisible] = useState(false);

    if (!showInfoTooltips) return null;

    const getPositionStyles = () => {
        switch (position) {
            case 'top': return { bottom: '100%', left: '50%', transform: 'translateX(-50%) translateY(-8px)' };
            case 'bottom': return { top: '100%', left: '50%', transform: 'translateX(-50%) translateY(8px)' };
            case 'left': return { right: '100%', top: '50%', transform: 'translateY(-50%) translateX(-8px)' };
            case 'right': return { left: '100%', top: '50%', transform: 'translateY(-50%) translateX(8px)' };
            default: return { bottom: '100%', left: '50%', transform: 'translateX(-50%) translateY(-8px)' };
        }
    };

    return (
        <div
            className="relative inline-block"
            style={{ position: 'relative', display: 'inline-block', color: 'rgba(255,255,255,0.4)', verticalAlign: 'middle', cursor: 'help' }}
            onMouseEnter={() => setIsVisible(true)}
            onMouseLeave={() => setIsVisible(false)}
        >
            <Info size={14} />

            <AnimatePresence>
                {isVisible && (
                    <motion.div
                        initial={{ opacity: 0, scale: 0.9, y: position === 'top' ? 5 : -5 }}
                        animate={{ opacity: 1, scale: 1, y: 0 }}
                        exit={{ opacity: 0, scale: 0.9, y: position === 'top' ? 5 : -5 }}
                        transition={{ duration: 0.15, ease: "easeOut" }}
                        style={{
                            position: 'absolute',
                            zIndex: 1000,
                            padding: '10px 14px',
                            background: '#18181b', // zinc-900
                            border: '1px solid rgba(255,255,255,0.1)',
                            borderRadius: '8px',
                            color: '#e4e4e7', // zinc-200
                            fontSize: '12px',
                            width: '240px',
                            lineHeight: '1.5',
                            boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.4)',
                            filter: 'drop-shadow(0 0 8px rgba(0,0,0,0.3))',
                            ...getPositionStyles()
                        }}
                    >
                        {text}
                        {/* Triangle arrow */}
                        <div style={{
                            position: 'absolute',
                            width: '0',
                            height: '0',
                            borderLeft: '6px solid transparent',
                            borderRight: '6px solid transparent',
                            borderTop: position === 'top' ? '6px solid #18181b' : 'none',
                            borderBottom: position === 'bottom' ? '6px solid #18181b' : 'none',
                            left: '50%',
                            marginLeft: '-6px',
                            bottom: position === 'top' ? '-6px' : 'auto',
                            top: position === 'bottom' ? '-6px' : 'auto'
                        }} />
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};
