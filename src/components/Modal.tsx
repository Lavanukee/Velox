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
    maxWidth?: string;
}

export const Modal: React.FC<ModalProps> = ({
    isOpen,
    onClose,
    title,
    children,
    footer,
    maxWidth = '500px'
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
        <div
            className="modal-overlay animate-fade-in"
            onClick={(e) => e.target === e.currentTarget && onClose()}
        >
            <div
                className="modal-panel"
                style={{ maxWidth }}
            >
                <div className="modal-header">
                    <h3 style={{
                        fontSize: '1.125rem',
                        fontWeight: 600,
                        color: 'var(--text-main)',
                        margin: 0
                    }}>
                        {title}
                    </h3>
                    <button
                        onClick={onClose}
                        style={{
                            background: 'transparent',
                            border: 'none',
                            color: 'var(--text-secondary)',
                            cursor: 'pointer',
                            padding: '4px',
                            borderRadius: '6px',
                            display: 'flex',
                            transition: 'all 0.2s'
                        }}
                        onMouseEnter={(e) => e.currentTarget.style.color = 'var(--text-main)'}
                        onMouseLeave={(e) => e.currentTarget.style.color = 'var(--text-secondary)'}
                    >
                        <X size={20} />
                    </button>
                </div>

                <div className="modal-body">
                    {children}
                </div>

                {footer && (
                    <div className="modal-footer">
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
