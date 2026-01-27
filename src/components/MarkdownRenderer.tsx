import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';

interface MarkdownRendererProps {
    content: string;
    className?: string;
}

export const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({ content, className }) => {
    return (
        <div className={className} style={{ lineHeight: 1.7 }}>
            <ReactMarkdown
                remarkPlugins={[remarkMath]}
                rehypePlugins={[rehypeKatex]}
                components={{
                    // Code blocks
                    code({ node, inline, className, children, ...props }: any) {
                        const match = /language-(\w+)/.exec(className || '');
                        const language = match ? match[1] : '';

                        if (inline) {
                            return (
                                <code
                                    style={{
                                        background: 'rgba(167,139,250,0.15)',
                                        padding: '2px 6px',
                                        borderRadius: '4px',
                                        fontSize: '0.9em',
                                        color: '#60a5fa'
                                    }}
                                    {...props}
                                >
                                    {children}
                                </code>
                            );
                        }

                        return (
                            <div style={{ position: 'relative', marginBottom: '12px' }}>
                                {language && (
                                    <div style={{
                                        position: 'absolute',
                                        top: '8px',
                                        right: '8px',
                                        fontSize: '10px',
                                        color: '#6b7280',
                                        textTransform: 'uppercase'
                                    }}>
                                        {language}
                                    </div>
                                )}
                                <pre style={{
                                    background: 'rgba(0,0,0,0.4)',
                                    padding: '16px',
                                    borderRadius: '8px',
                                    overflowX: 'auto',
                                    fontSize: '13px',
                                    margin: 0
                                }}>
                                    <code className={className} style={{ color: '#e5e7eb' }} {...props}>
                                        {children}
                                    </code>
                                </pre>
                            </div>
                        );
                    },
                    // Headings
                    h1: ({ children }) => <h1 style={{ fontSize: '1.5em', fontWeight: 600, marginBottom: '12px', color: '#fff' }}>{children}</h1>,
                    h2: ({ children }) => <h2 style={{ fontSize: '1.3em', fontWeight: 600, marginBottom: '10px', color: '#fff' }}>{children}</h2>,
                    h3: ({ children }) => <h3 style={{ fontSize: '1.1em', fontWeight: 600, marginBottom: '8px', color: '#fff' }}>{children}</h3>,
                    // Paragraphs
                    p: ({ children }) => <p style={{ margin: '0 0 12px 0' }}>{children}</p>,
                    // Lists
                    ul: ({ children }) => <ul style={{ margin: '0 0 12px 0', paddingLeft: '20px' }}>{children}</ul>,
                    ol: ({ children }) => <ol style={{ margin: '0 0 12px 0', paddingLeft: '20px' }}>{children}</ol>,
                    li: ({ children }) => <li style={{ marginBottom: '4px' }}>{children}</li>,
                    // Links
                    a: ({ href, children }) => (
                        <a
                            href={href}
                            target="_blank"
                            rel="noopener noreferrer"
                            style={{ color: '#3b82f6', textDecoration: 'underline' }}
                        >
                            {children}
                        </a>
                    ),
                    // Blockquotes
                    blockquote: ({ children }) => (
                        <blockquote style={{
                            borderLeft: '3px solid rgba(167,139,250,0.5)',
                            paddingLeft: '12px',
                            margin: '0 0 12px 0',
                            color: '#a1a1aa'
                        }}>
                            {children}
                        </blockquote>
                    ),
                    // Tables
                    table: ({ children }) => (
                        <div style={{ overflowX: 'auto', marginBottom: '12px' }}>
                            <table style={{ borderCollapse: 'collapse', width: '100%' }}>
                                {children}
                            </table>
                        </div>
                    ),
                    th: ({ children }) => (
                        <th style={{
                            borderBottom: '1px solid rgba(255,255,255,0.2)',
                            padding: '8px 12px',
                            textAlign: 'left',
                            fontWeight: 600
                        }}>
                            {children}
                        </th>
                    ),
                    td: ({ children }) => (
                        <td style={{
                            borderBottom: '1px solid rgba(255,255,255,0.1)',
                            padding: '8px 12px'
                        }}>
                            {children}
                        </td>
                    ),
                    // Horizontal rule
                    hr: () => <hr style={{ border: 'none', borderTop: '1px solid rgba(255,255,255,0.1)', margin: '16px 0' }} />,
                    // Strong/Bold
                    strong: ({ children }) => <strong style={{ fontWeight: 600, color: '#fff' }}>{children}</strong>,
                    // Emphasis/Italic
                    em: ({ children }) => <em style={{ fontStyle: 'italic' }}>{children}</em>,
                }}
            >
                {content}
            </ReactMarkdown>
        </div>
    );
};

export default MarkdownRenderer;
