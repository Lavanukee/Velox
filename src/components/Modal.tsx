import React, { useEffect } from 'react';
import { X } from 'lucide-react';
import { Button } from './Button';
import '../styles/components.css';

interface ModalProps {
    isOpen: boolean;
    onClose: () => void;
    title: string;
    children: React.ReactNode;
    footer?: React.ReactNode;
}

export const Modal: React.FC<ModalProps> = ({
    isOpen,
    onClose,
    title,
    children,
    footer
}) => {
    useEffect(() => {
        const handleEsc = (e: KeyboardEvent) => {
            if (e.key === 'Escape') onClose();
        };
        if (isOpen) window.addEventListener('keydown', handleEsc);
        return () => window.removeEventListener('keydown', handleEsc);
    }, [isOpen, onClose]);

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-fade-in" style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, zIndex: 1000, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)' }}>
            <div className="bg-panel border border-white/10 rounded-xl shadow-2xl w-full max-w-md overflow-hidden glass" style={{ background: '#121216', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '12px', padding: '0', width: '100%', maxWidth: '500px' }}>
                <div className="flex items-center justify-between p-4 border-b border-white/5" style={{ display: 'flex', justifyContent: 'space-between', padding: '16px', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                    <h3 className="text-lg font-semibold text-white m-0">{title}</h3>
                    <button onClick={onClose} className="text-gray-400 hover:text-white transition-colors bg-transparent border-none cursor-pointer">
                        <X size={20} />
                    </button>
                </div>

                <div className="p-6" style={{ padding: '24px' }}>
                    {children}
                </div>

                {footer && (
                    <div className="flex justify-end gap-3 p-4 bg-white/5 border-t border-white/5" style={{ display: 'flex', justifyContent: 'flex-end', gap: '12px', padding: '16px', background: 'rgba(255,255,255,0.02)', borderTop: '1px solid rgba(255,255,255,0.05)' }}>
                        {footer}
                    </div>
                )}
            </div>
        </div>
    );
};

export const ConfirmModal: React.FC<{
    isOpen: boolean;
    onClose: () => void;
    onConfirm: () => void;
    title: string;
    message: string;
    confirmText?: string;
    cancelText?: string;
    isDanger?: boolean;
}> = ({
    isOpen,
    onClose,
    onConfirm,
    title,
    message,
    confirmText = 'Confirm',
    cancelText = 'Cancel',
    isDanger = false
}) => {
        return (
            <Modal
                isOpen={isOpen}
                onClose={onClose}
                title={title}
                footer={
                    <>
                        <Button variant="ghost" onClick={onClose}>{cancelText}</Button>
                        <Button variant={isDanger ? 'danger' : 'primary'} onClick={() => { onConfirm(); onClose(); }}>
                            {confirmText}
                        </Button>
                    </>
                }
            >
                <p className="text-gray-300">{message}</p>
            </Modal>
        );
    };
