import React, { useState, useRef, useEffect, useCallback } from 'react';
import { invoke } from "@tauri-apps/api/core";
import { convertFileSrc } from '@tauri-apps/api/core';
// import { listen } from "@tauri-apps/api/event"; // 'listen' is declared but its value is never read.

// --- TYPES ---
interface Box {
    id: number;
    x: number;
    y: number;
    w: number;
    h: number;
    label: string;
    status: 'predicted' | 'accepted' | 'user_created';
    caption?: string;
    score?: number;
}

interface Point { x: number; y: number; }

interface AnnotationEditorProps {
    initialImagePath: string | null; // Initial image path, if any
    modelConfig: any;
    onClose: () => void;
}

// --- CONSTANTS ---
const HANDLE_SIZE = 8;
const COLOR_PREDICTED = 'rgba(255, 193, 7, 0.4)';
const COLOR_ACCEPTED = 'rgba(76, 175, 80, 0.4)';
const COLOR_SELECTED_BORDER = '#2196F3';
const COLOR_HANDLE = '#FFFFFF';

export const AnnotationEditor: React.FC<AnnotationEditorProps> = ({
    initialImagePath,
    modelConfig,
    onClose
}) => {
    // --- STATE ---
    const [currentImagePath, setCurrentImagePath] = useState<string | null>(initialImagePath); // Path is stored for saving individual crops
    const [imgSrc, setImgSrc] = useState<string>(""); // Used as src for the Image object
    const [imageDimensions, setImageDimensions] = useState<{w: number, h: number} | null>(null);
    const [status, setStatus] = useState("Ready");
    const [countdown, setCountdown] = useState<number | null>(null);

    // Annotations
    const [boxes, setBoxes] = useState<Box[]>([]);
    const [selectedIds, setSelectedIds] = useState<number[]>([]);
    
    // Interactions
    const [mode, setMode] = useState<'create' | 'select' | 'yolo'>('create');
    const [isDragging, setIsDragging] = useState(false);
    const [dragAction, setDragAction] = useState<'create' | 'move' | 'resize_tl' | 'resize_tr' | 'resize_bl' | 'resize_br' | 'select_rect' | null>(null);
    const [startPos, setStartPos] = useState<Point>({ x: 0, y: 0 });
    const [currentPos, setCurrentPos] = useState<Point>({ x: 0, y: 0 }); // For drag previews

    // Inputs
    const [captionInput, setCaptionInput] = useState("");
    const [splitCount, setSplitCount] = useState(2);
    const [splitSpacing, setSplitSpacing] = useState(0);

    // Refs
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const imageRef = useRef<HTMLImageElement | null>(null);

    // --- INITIALIZATION & IMAGE LOADING ---
 
    useEffect(() => {
        if (currentImagePath) {
            const url = convertFileSrc(currentImagePath);
            console.log("DEBUG: Generated image URL:", url); // Log the generated URL
            setImgSrc(url);
            setStatus(`Loaded: ${currentImagePath}`);
        } else {
            console.log("DEBUG: No currentImagePath, setting imgSrc to empty.");
            setImgSrc("");
            setImageDimensions(null);
            setStatus("No image loaded.");
        }
    }, [currentImagePath]); // Only depend on currentImagePath
 
    useEffect(() => {
        // Load image object for Canvas drawing
        if (!imgSrc) return;
        
        const img = new Image();
        img.src = imgSrc;
        img.onload = () => {
            setImageDimensions({ w: img.width, h: img.height });
            imageRef.current = img;
            drawCanvas(); // Initial draw
        };
        img.onerror = (err) => {
            console.error("Image load error:", err);
            setStatus("ERROR: Could not load image. Check Tauri `capabilities` permissions.");
        };
    }, [imgSrc]); // Removed drawCanvas from dependencies

    // --- CANVAS DRAWING ---

    const drawCanvas = useCallback(() => {
        const canvas = canvasRef.current;
        const ctx = canvas?.getContext('2d');
        const img = imageRef.current;
        if (!canvas || !ctx || !img) return;

        // 1. Reset Canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // 2. Draw Image (Native Size)
        ctx.drawImage(img, 0, 0);

        // 3. Draw Existing Boxes
        boxes.forEach((box, _index) => {
            const isSelected = selectedIds.includes(box.id);
            
            // Fill
            ctx.fillStyle = box.status === 'predicted' ? COLOR_PREDICTED : COLOR_ACCEPTED;
            if (isSelected) ctx.fillStyle = 'rgba(33, 150, 243, 0.3)';
            ctx.fillRect(box.x, box.y, box.w, box.h);

            // Border
            ctx.lineWidth = isSelected ? 2 : 1;
            ctx.strokeStyle = isSelected ? COLOR_SELECTED_BORDER : (box.status === 'predicted' ? '#FFC107' : '#4CAF50');
            ctx.strokeRect(box.x, box.y, box.w, box.h);

            // Label
            ctx.fillStyle = "white";
            ctx.font = "12px sans-serif";
            ctx.shadowColor = "black";
            ctx.shadowBlur = 4;
            const labelText = `${box.id}: ${box.caption || box.label}`;
            ctx.fillText(labelText, box.x + 4, box.y + 14);
            ctx.shadowBlur = 0;

            // Handles (if selected and single selection)
            if (isSelected && selectedIds.length === 1) {
                drawHandle(ctx, box.x, box.y); // TL
                drawHandle(ctx, box.x + box.w, box.y); // TR
                drawHandle(ctx, box.x, box.y + box.h); // BL
                drawHandle(ctx, box.x + box.w, box.y + box.h); // BR
            }
        });

        // 4. Draw Drag Preview
        if (isDragging && dragAction) {
            const x = Math.min(startPos.x, currentPos.x);
            const y = Math.min(startPos.y, currentPos.y);
            const w = Math.abs(currentPos.x - startPos.x);
            const h = Math.abs(currentPos.y - startPos.y);

            if (dragAction === 'create') {
                ctx.strokeStyle = '#9C27B0';
                ctx.lineWidth = 2;
                ctx.setLineDash([5, 5]);
                ctx.strokeRect(x, y, w, h);
                ctx.setLineDash([]);
            } else if (dragAction === 'select_rect') {
                ctx.fillStyle = 'rgba(255, 0, 0, 0.2)';
                ctx.strokeStyle = 'red';
                ctx.fillRect(x, y, w, h);
                ctx.strokeRect(x, y, w, h);
            }
        }
    }, [boxes, selectedIds, isDragging, dragAction, startPos, currentPos]);

    const drawHandle = (ctx: CanvasRenderingContext2D, x: number, y: number) => {
        ctx.fillStyle = COLOR_HANDLE;
        ctx.strokeStyle = "black";
        ctx.lineWidth = 1;
        ctx.fillRect(x - HANDLE_SIZE/2, y - HANDLE_SIZE/2, HANDLE_SIZE, HANDLE_SIZE);
        ctx.strokeRect(x - HANDLE_SIZE/2, y - HANDLE_SIZE/2, HANDLE_SIZE, HANDLE_SIZE);
    };

    // Trigger redraw on state change
    useEffect(() => { drawCanvas(); }, [drawCanvas]);


    // --- MOUSE EVENTS ---

    const getMousePos = (e: React.MouseEvent) => {
        const rect = canvasRef.current!.getBoundingClientRect();
        return {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
        };
    };

    const handleMouseDown = (e: React.MouseEvent) => {
        const pos = getMousePos(e);
        setStartPos(pos);
        setCurrentPos(pos);
        setIsDragging(true);

        // 1. Check Handles (Resize) - Priority
        if (selectedIds.length === 1) {
            const box = boxes.find(b => b.id === selectedIds[0]);
            if (box) {
                // Simple hit detection for corners
                const hitDist = HANDLE_SIZE; 
                if (Math.abs(pos.x - box.x) < hitDist && Math.abs(pos.y - box.y) < hitDist) { setDragAction('resize_tl'); return; }
                if (Math.abs(pos.x - (box.x+box.w)) < hitDist && Math.abs(pos.y - box.y) < hitDist) { setDragAction('resize_tr'); return; }
                if (Math.abs(pos.x - box.x) < hitDist && Math.abs(pos.y - (box.y+box.h)) < hitDist) { setDragAction('resize_bl'); return; }
                if (Math.abs(pos.x - (box.x+box.w)) < hitDist && Math.abs(pos.y - (box.y+box.h)) < hitDist) { setDragAction('resize_br'); return; }
            }
        }

        // 2. Check Inside Box (Move or Select)
        // Iterate reverse to select top-most
        for (let i = boxes.length - 1; i >= 0; i--) {
            const b = boxes[i];
            if (pos.x >= b.x && pos.x <= b.x + b.w && pos.y >= b.y && pos.y <= b.y + b.h) {
                if (mode === 'create') {
                    setSelectedIds([b.id]);
                    setDragAction('move');
                    setCaptionInput(b.caption || b.label);
                    return;
                } else if (mode === 'select') {
                    // Toggle selection logic could go here, but simple select for now
                    if (!selectedIds.includes(b.id)) setSelectedIds([b.id]);
                    return;
                }
            }
        }

        // 3. Background Click
        if (mode === 'create') {
            setSelectedIds([]); // Deselect
            setDragAction('create');
        } else if (mode === 'select') {
            setDragAction('select_rect');
        }
    };

    const handleMouseMove = (e: React.MouseEvent) => {
        if (!isDragging) return;
        const pos = getMousePos(e);
        const dx = pos.x - currentPos.x;
        const dy = pos.y - currentPos.y;
        setCurrentPos(pos);

        if (dragAction === 'move' && selectedIds.length === 1) {
            setBoxes(prev => prev.map(b => {
                if (b.id !== selectedIds[0]) return b;
                return { ...b, x: b.x + dx, y: b.y + dy };
            }));
        } else if (dragAction?.startsWith('resize')) {
            const boxId = selectedIds[0];
            setBoxes(prev => prev.map(b => {
                if (b.id !== boxId) return b;
                let { x, y, w, h } = b;
                if (dragAction === 'resize_tl') { x += dx; y += dy; w -= dx; h -= dy; }
                if (dragAction === 'resize_tr') { y += dy; w += dx; h -= dy; }
                if (dragAction === 'resize_bl') { x += dx; w -= dx; h += dy; }
                if (dragAction === 'resize_br') { w += dx; h += dy; }
                return { ...b, x, y, w, h };
            }));
        }
        // 'create' and 'select_rect' handled visually in drawCanvas, finalized in MouseUp
    };

    const handleMouseUp = () => {
        setIsDragging(false);
        if (!dragAction) return;

        if (dragAction === 'create') {
            const x = Math.min(startPos.x, currentPos.x);
            const y = Math.min(startPos.y, currentPos.y);
            const w = Math.abs(currentPos.x - startPos.x);
            const h = Math.abs(currentPos.y - startPos.y);
            
            if (w > 5 && h > 5) {
                const newId = Date.now(); // Simple ID generation
                const newBox: Box = {
                    id: newId, x, y, w, h,
                    label: 'user_created',
                    status: 'user_created'
                };
                setBoxes(prev => [...prev, newBox]);
                setSelectedIds([newId]);
            }
        } else if (dragAction === 'select_rect') {
            const x1 = Math.min(startPos.x, currentPos.x);
            const y1 = Math.min(startPos.y, currentPos.y);
            const x2 = Math.max(startPos.x, currentPos.x);
            const y2 = Math.max(startPos.y, currentPos.y);

            const newSelection = boxes.filter(b => 
                b.x < x2 && b.x + b.w > x1 &&
                b.y < y2 && b.y + b.h > y1
            ).map(b => b.id);
            
            setSelectedIds(newSelection);
        }

        setDragAction(null);
    };


    // --- LOGIC FUNCTIONS ---

    const handleDelete = () => {
        setBoxes(prev => prev.filter(b => !selectedIds.includes(b.id)));
        setSelectedIds([]);
    };

    const handleAccept = () => {
        setBoxes(prev => prev.map(b => 
            selectedIds.includes(b.id) ? { ...b, status: 'accepted' } : b
        ));
    };

    const handleUpdateCaption = () => {
        if (selectedIds.length === 1) {
            setBoxes(prev => prev.map(b => 
                b.id === selectedIds[0] ? { ...b, caption: captionInput } : b
            ));
        }
    };

    const handleSplit = (direction: 'h' | 'v') => {
        if (selectedIds.length !== 1) {
            setStatus("Select exactly one box to split.");
            return;
        }
        const original = boxes.find(b => b.id === selectedIds[0]);
        if (!original) return;

        const count = splitCount;
        const spacing = splitSpacing;
        const newBoxes: Box[] = [];

        if (direction === 'h') {
            const totalSpacing = spacing * (count - 1);
            const boxW = (original.w - totalSpacing) / count;
            for(let i=0; i<count; i++) {
                newBoxes.push({
                    ...original,
                    id: Date.now() + i,
                    x: original.x + i * (boxW + spacing),
                    w: boxW,
                    label: `${original.label}_split_${i+1}`,
                    status: 'user_created'
                });
            }
        } else {
            const totalSpacing = spacing * (count - 1);
            const boxH = (original.h - totalSpacing) / count;
            for(let i=0; i<count; i++) {
                newBoxes.push({
                    ...original,
                    id: Date.now() + i,
                    y: original.y + i * (boxH + spacing),
                    h: boxH,
                    label: `${original.label}_split_${i+1}`,
                    status: 'user_created'
                });
            }
        }

        // Replace original with new ones
        setBoxes(prev => [
            ...prev.filter(b => b.id !== original.id),
            ...newBoxes
        ]);
        setSelectedIds([]);
    };

    const handleCapture = async () => {
        setCountdown(3);
        setStatus("Capturing in...");
        const timer = setInterval(() => {
            setCountdown(prev => {
                if (prev === 1) {
                    clearInterval(timer);
                    return null;
                }
                return (prev || 0) - 1;
            });
        }, 1000);

        setTimeout(async () => {
            try {
                // Call the new command, which only returns the path
                const path = await invoke<string>("take_screenshot_path_command", {
                    outputDir: modelConfig.rawScreenshotDir,
                    filenamePrefix: "annotator_snap"
                });
                setCurrentImagePath(path);
                setBoxes([]); // Clear old boxes
                setStatus("New screenshot captured.");
            } catch (e) {
                setStatus(`Capture failed: ${e}`);
            } finally {
                setCountdown(null);
            }
        }, 3000); // Wait 3 seconds for screenshot + 1 second buffer
    };

    const handleSave = async () => {
        if (!currentImagePath) {
            setStatus("No image to save annotations for.");
            return;
        }

        if (selectedIds.length === 0) {
            setStatus("Select at least one box to save.");
            return;
        }

        for (const boxId of selectedIds) {
            const box = boxes.find(b => b.id === boxId);
            if (box) {
                const annotation = {
                    box_2d: [Math.round(box.x), Math.round(box.y), Math.round(box.x + box.w), Math.round(box.y + box.h)],
                    label: box.caption || box.label,
                };
                try {
                    await invoke("save_annotation_crop_command", {
                        imagePath: currentImagePath,
                        box: annotation.box_2d,
                        label: annotation.label,
                        saveDir: modelConfig.annotatedDataDir,
                    });
                    setStatus(`Saved crop for box ID ${box.id}!`);
                } catch (e) {
                    setStatus(`Save failed for box ID ${box.id}: ${e}`);
                    console.error(`Error saving crop for box ID ${box.id}:`, e);
                }
            }
        }
        setStatus("Selected annotations saved.");
    };


    // --- UI RENDER ---

    return (
        <div style={{ position: 'fixed', inset: 0, background: '#1e1e1e', color: '#eee', display: 'flex', fontFamily: 'sans-serif' }}>
            
            {/* LEFT SIDEBAR */}
            <div style={{ width: '300px', borderRight: '1px solid #333', display: 'flex', flexDirection: 'column', background: '#252526' }}>
                <div style={{ padding: '15px', borderBottom: '1px solid #333', background: '#333' }}>
                    <h3 style={{ margin: 0 }}>Annotation Tool</h3>
                    <div style={{ fontSize: '0.8rem', color: '#aaa', marginTop: '5px' }}>{status}</div>
                </div>

                <div style={{ padding: '15px', overflowY: 'auto', flex: 1 }}>
                    {/* Mode Select */}
                    <div style={{ marginBottom: '15px' }}>
                        <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>Mode</label>
                        <div style={{ display: 'flex', gap: '5px' }}>
                            <button className={mode === 'create' ? 'btn-active' : ''} onClick={() => setMode('create')} style={btnStyle}>Create/Move</button>
                            <button className={mode === 'select' ? 'btn-active' : ''} onClick={() => setMode('select')} style={btnStyle}>Select Area</button>
                        </div>
                    </div>

                    {/* Split Controls */}
                    <div style={{ marginBottom: '15px', padding: '10px', background: '#2d2d2d', borderRadius: '4px' }}>
                        <label style={{ fontWeight: 'bold', display:'block', marginBottom:'5px' }}>Split Selection</label>
                        <div style={{ display: 'flex', gap: '5px', marginBottom: '5px' }}>
                            <input type="number" value={splitCount} onChange={e => setSplitCount(Number(e.target.value))} style={inputStyle} placeholder="Count" />
                            <input type="number" value={splitSpacing} onChange={e => setSplitSpacing(Number(e.target.value))} style={inputStyle} placeholder="Gap px" />
                        </div>
                        <div style={{ display: 'flex', gap: '5px' }}>
                            <button onClick={() => handleSplit('h')} style={btnStyle}>Horiz</button>
                            <button onClick={() => handleSplit('v')} style={btnStyle}>Vert</button>
                        </div>
                    </div>

                    {/* Caption Input */}
                    <div style={{ marginBottom: '15px' }}>
                        <label style={{ fontWeight: 'bold' }}>Caption</label>
                        <input 
                            value={captionInput} 
                            onChange={e => setCaptionInput(e.target.value)} 
                            style={{ ...inputStyle, width: '100%' }} 
                            placeholder="Selected label..."
                        />
                        <button onClick={handleUpdateCaption} style={{ ...btnStyle, width: '100%', marginTop: '5px' }}>Update Caption</button>
                    </div>

                    {/* Actions */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '5px' }}>
                        <button onClick={handleAccept} style={{ ...btnStyle, background: '#388E3C' }}>Accept Selected</button>
                        <button onClick={handleDelete} style={{ ...btnStyle, background: '#D32F2F' }}>Delete Selected</button>
                        <button onClick={() => setBoxes([])} style={{ ...btnStyle, background: '#F57C00' }}>Clear All</button>
                    </div>

                    {/* List */}
                    <div style={{ marginTop: '20px' }}>
                        <label style={{ fontWeight: 'bold' }}>Detected ({boxes.length})</label>
                        <ul style={{ listStyle: 'none', padding: 0, marginTop: '5px', fontSize: '0.9rem' }}>
                            {boxes.map((box, i) => (
                                <li 
                                    key={box.id} 
                                    onClick={() => { setSelectedIds([box.id]); setCaptionInput(box.caption || box.label); }}
                                    style={{ 
                                        padding: '4px', 
                                        cursor: 'pointer',
                                        background: selectedIds.includes(box.id) ? '#37373D' : 'transparent',
                                        borderLeft: `3px solid ${box.status === 'accepted' ? '#4CAF50' : '#FFC107'}`
                                    }}
                                >
                                    {i}: {box.caption || box.label}
                                </li>
                            ))}
                        </ul>
                    </div>
                </div>

                {/* Footer Buttons */}
                <div style={{ padding: '15px', borderTop: '1px solid #333', display: 'flex', flexDirection: 'column', gap: '10px' }}>
                    <button onClick={handleCapture} style={{ ...btnStyle, padding: '10px' }}>📸 Capture Screenshot</button>
                    <button onClick={handleSave} style={{ ...btnStyle, padding: '10px', background: '#1976D2', fontWeight: 'bold' }}>💾 Save & Exit</button>
                    <button onClick={onClose} style={{ ...btnStyle, background: 'transparent', border: '1px solid #555' }}>Close</button>
                </div>
            </div>

            {/* MAIN CANVAS AREA (SCROLLABLE) */}
            <div
                ref={containerRef}
                style={{ flex: 1, overflow: 'auto', background: '#111', position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
            >
                {currentImagePath ? (
                    <div style={{ position: 'relative', width: 'fit-content', height: 'fit-content', maxWidth: '100%', maxHeight: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        <canvas
                            ref={canvasRef}
                            width={imageDimensions?.w || 800}
                            height={imageDimensions?.h || 600}
                            onMouseDown={handleMouseDown}
                            onMouseMove={handleMouseMove}
                            onMouseUp={handleMouseUp}
                            style={{ display: 'block', cursor: 'crosshair', maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }}
                        />
                    </div>
                ) : (
                    <div style={{ textAlign: 'center', color: '#666' }}>
                        <h2>No Image Loaded</h2>
                        {countdown !== null ? (
                            <p style={{ fontSize: '2em', fontWeight: 'bold' }}>{countdown}</p>
                        ) : (
                            <button onClick={handleCapture} style={{ ...btnStyle, padding: '10px 20px', fontSize: '1.2rem' }}>Take Screenshot</button>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
};

// --- STYLES HELPER ---
// change these to fit the rest of the app's styling
const btnStyle: React.CSSProperties = {
    padding: '6px 12px',
    background: '#444',
    color: 'white',
    border: 'none',
    borderRadius: '3px',
    cursor: 'pointer',
    fontSize: '0.85rem'
};

const inputStyle: React.CSSProperties = {
    padding: '5px',
    background: '#1e1e1e',
    border: '1px solid #555',
    color: 'white',
    borderRadius: '3px',
    width: '60px'
};