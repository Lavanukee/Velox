import React from 'react';
import { RecipeBlock } from '../../types/GenerationRecipe';
import { X, Database, Sparkles } from 'lucide-react';

interface DragDropEditorProps {
    value: (string | RecipeBlock)[];
    onChange: (value: (string | RecipeBlock)[]) => void;
    availableSources: { id: string; name: string }[];
    placeholder?: string;
    style?: React.CSSProperties;
}

export interface DragDropEditorHandle {
    insertBlock: (block: RecipeBlock) => void;
}

export const DragDropEditor = React.forwardRef<DragDropEditorHandle, DragDropEditorProps>(({ value, onChange, availableSources, placeholder, style }, ref) => {
    // Ensure value corresponds to at least one empty text string if empty
    const items = value.length > 0 ? value : [''];

    // State to track last active cursor position for insertion
    const lastCursorRef = React.useRef<{ index: number; position: number } | null>(null);

    // Expose insert method
    React.useImperativeHandle(ref, () => ({
        insertBlock: (block: RecipeBlock) => {
            const cursor = lastCursorRef.current;
            const newItems = [...items];

            if (cursor && cursor.index < newItems.length && typeof newItems[cursor.index] === 'string') {
                // Split string at cursor
                const text = newItems[cursor.index] as string;
                const pos = cursor.position;
                const before = text.substring(0, pos);
                const after = text.substring(pos);

                // Replace current string with [before, block, after] (filtering empty strings if needed, but keeping structure safe)
                // Actually we just insert into array: splice
                // We need to remove the original string and insert 3 items: before, block, after.
                newItems.splice(cursor.index, 1, before, block, after);
            } else {
                // Default: Append to end
                // If last item is string, append to it? No, blocks must be separate elements in array
                // If last item is string, we insert block then new string.
                // Actually strict structure: (string | block)[]
                // If we append, we usually want [..., lastString, block, ""]
                newItems.push(block, "");
            }

            // Clean up empty strings if desired, or keep them for spacing/typing. 
            // Better to keep them so user can type between blocks.
            onChange(newItems);
        }
    }));

    const handleTextChange = (index: number, text: string) => {
        const newItems = [...items];
        newItems[index] = text;
        onChange(newItems);
    };



    const handleSelect = (e: React.SyntheticEvent<HTMLTextAreaElement>, index: number) => {
        const target = e.currentTarget;
        lastCursorRef.current = { index, position: target.selectionStart };
    };

    const removeBlock = (index: number) => {
        // Remove block and merge surrounding text if possible
        const newItems = [...items];
        newItems.splice(index, 1);

        // Merge adjacent strings
        const mergedItems: (string | RecipeBlock)[] = [];
        for (const item of newItems) {
            if (typeof item === 'string') {
                if (mergedItems.length > 0 && typeof mergedItems[mergedItems.length - 1] === 'string') {
                    mergedItems[mergedItems.length - 1] += item;
                } else {
                    mergedItems.push(item);
                }
            } else {
                mergedItems.push(item);
            }
        }
        if (mergedItems.length === 0) mergedItems.push('');
        onChange(mergedItems);
    };

    const onDrop = (e: React.DragEvent, index?: number) => {
        e.preventDefault();
        e.stopPropagation(); // Stop parent from handling

        let type = e.dataTransfer.getData('velocity/type');
        let payload = e.dataTransfer.getData('velocity/payload');

        if (!type) {
            try {
                const text = e.dataTransfer.getData('text/plain');
                if (text) {
                    const parsed = JSON.parse(text);
                    if (parsed.type && parsed.payload) {
                        type = parsed.type;
                        payload = parsed.payload;
                    }
                }
            } catch (ignore) { }
        }

        if (!type || (type !== 'source' && type !== 'generator')) return;

        let newBlock: RecipeBlock | null = null;
        if (type === 'source') {
            newBlock = { type: 'source_data', sourceId: payload };
        } else if (type === 'generator') {
            newBlock = { type: 'generator', prompt: [''] };
        }

        if (newBlock) {
            const newItems = [...items];
            // If dropped on specific index (textarea), insert there
            if (index !== undefined && typeof newItems[index] === 'string') {
                // We're dropping ONTO a text area. 
                // Ideally we use selection/caret position from event, but that's hard.
                // We will split at the END of this text block for simplicity of "Drop" logic 
                // unless we have specific caret info.
                // Let's just splice AFTER this index.
                newItems.splice(index + 1, 0, newBlock, '');
            } else {
                // Dropped on container (index undefined) -> Append
                newItems.push(newBlock, '');
            }
            onChange(newItems);
        }
    };

    return (
        <div
            style={{
                border: '1px solid var(--border-input)',
                borderRadius: '6px',
                background: 'var(--bg-input)',
                padding: '12px',
                minHeight: '200px', // Increased default height
                display: 'flex',
                flexWrap: 'wrap',
                alignItems: 'flex-start', // Align items to top
                alignContent: 'flex-start',
                gap: '4px',
                cursor: 'text',
                fontSize: '14px', // Slightly larger text base
                lineHeight: '1.6',
                ...style // Allow overrides
            }}
            onClick={(e) => {
                // Focus last textarea if container clicked
                if (e.target === e.currentTarget) {
                    // Find last textarea ref? simplified: user can click the textareas.
                }
            }}
            onDragOver={(e) => {
                e.preventDefault();
                e.dataTransfer.dropEffect = 'copy';
            }}
            onDrop={(e) => onDrop(e)} // Container drop handling
        >
            {items.map((item, index) => {
                if (typeof item === 'string') {
                    return (
                        <textarea
                            key={`text-${index}`}
                            value={item}
                            onChange={(e) => handleTextChange(index, e.target.value)}
                            onSelect={(e) => handleSelect(e, index)}
                            onClick={(e) => handleSelect(e, index)}
                            onKeyUp={(e) => handleSelect(e, index)}
                            onDrop={(e) => onDrop(e, index)}
                            onDragOver={(e) => {
                                e.preventDefault();
                                e.stopPropagation();
                            }}
                            placeholder={items.length === 1 ? placeholder : undefined}
                            style={{
                                flex: '1 0 auto',
                                minWidth: '40px',
                                background: 'transparent',
                                border: 'none',
                                color: 'var(--text-main)',
                                fontSize: '13px',
                                resize: 'none',
                                outline: 'none',
                                padding: '4px',
                                height: 'auto',
                                fontFamily: 'inherit',
                                overflow: 'hidden'
                            }}
                            ref={el => {
                                if (el) {
                                    el.style.height = 'auto';
                                    el.style.height = el.scrollHeight + 'px';
                                }
                            }}
                        />
                    );
                } else {
                    const block = item as Extract<RecipeBlock, { type: 'source_data' } | { type: 'generator' }>;
                    const sourceName = block.type === 'source_data'
                        ? availableSources.find(s => s.id === (block as any).sourceId)?.name || 'Unknown Source'
                        : '';

                    return (
                        <div key={`block-${index}`} style={{
                            display: 'inline-flex', alignItems: 'center', gap: '6px',
                            background: item.type === 'generator' ? 'rgba(99, 102, 241, 0.2)' : 'rgba(16, 185, 129, 0.2)',
                            border: `1px solid ${item.type === 'generator' ? 'var(--accent-primary)' : 'var(--accent-success, #10b981)'}`,
                            borderRadius: '16px',
                            padding: '2px 8px',
                            fontSize: '12px',
                            userSelect: 'none',
                            cursor: 'default' // Blocks shouldn't be text-selectable cursor
                        }}>
                            {item.type === 'generator' ? (
                                <>
                                    <Sparkles size={12} fill="currentColor" />
                                    <span>Generate Text</span>
                                </>
                            ) : (
                                <>
                                    <Database size={12} />
                                    <span>{sourceName}</span>
                                </>
                            )}
                            <button
                                onClick={() => removeBlock(index)}
                                style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 0, display: 'flex' }}
                            >
                                <X size={12} style={{ opacity: 0.6 }} />
                            </button>
                        </div>
                    );
                }
            })}
        </div>
    );
});
