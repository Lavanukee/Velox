import React from 'react';
import { Loader2 } from 'lucide-react';
import '../styles/components.css';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
    variant?: 'primary' | 'secondary' | 'ghost' | 'danger' | 'outline';
    size?: 'sm' | 'md' | 'lg' | 'icon';
    isLoading?: boolean;
    leftIcon?: React.ReactNode;
    rightIcon?: React.ReactNode;
}

export const Button: React.FC<ButtonProps> = ({
    children,
    variant = 'primary',
    size = 'md',
    isLoading = false,
    leftIcon,
    rightIcon,
    className = '',
    disabled,
    ...props
}) => {
    const baseClass = 'btn';
    const variantClass = `btn-${variant}`;
    const sizeClass = `btn-${size}`;
    const loadingClass = isLoading ? 'is-loading' : '';

    return (
        <button
            className={`${baseClass} ${variantClass} ${sizeClass} ${loadingClass} ${className}`}
            disabled={disabled || isLoading}
            {...props}
        >
            {isLoading && <Loader2 className="animate-spin" size={16} style={{ marginRight: children ? '8px' : 0 }} />}
            {!isLoading && leftIcon && <span className="btn-icon-left">{leftIcon}</span>}
            {children}
            {!isLoading && rightIcon && <span className="btn-icon-right">{rightIcon}</span>}
        </button>
    );
};
