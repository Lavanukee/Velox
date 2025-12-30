import React from 'react';
import '../styles/components.css';

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
    hoverable?: boolean;
    active?: boolean;
    glass?: boolean;
}

export const Card: React.FC<CardProps> = ({
    children,
    className = '',
    hoverable = false,
    active = false,
    glass = true,
    style,
    ...props
}) => {
    const classes = [
        'card',
        glass ? 'glass' : '',
        hoverable ? 'card-hoverable' : '',
        active ? 'card-active' : '',
        className
    ].filter(Boolean).join(' ');

    return (
        <div className={classes} style={style} {...props}>
            {children}
        </div>
    );
};
