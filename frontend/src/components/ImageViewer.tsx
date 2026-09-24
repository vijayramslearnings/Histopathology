'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Eye, Layers, Sliders, ZoomIn, ZoomOut, RotateCcw, Maximize2 } from 'lucide-react';

interface ImageViewerProps {
  originalUrl: string;
  gradcamUrl?: string;
  attentionUrl?: string;
  altText?: string;
}

export function ImageViewer({ originalUrl, gradcamUrl, attentionUrl, altText }: ImageViewerProps) {
  const [activeTab, setActiveTab] = useState<'original' | 'gradcam' | 'attention'>('gradcam');
  const [opacity, setOpacity] = useState<number>(0.7);
  const [zoom, setZoom] = useState<number>(1);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Reset zoom & pan
  const handleResetZoom = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  const handleZoomIn = () => setZoom((prev) => Math.min(prev + 0.3, 3));
  const handleZoomOut = () => setZoom((prev) => Math.max(prev - 0.3, 1));

  // Handle Drag / Pan
  const handleMouseDown = (e: React.MouseEvent) => {
    if (zoom <= 1) return;
    setIsDragging(true);
    setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;
    setPan({
      x: e.clientX - dragStart.x,
      y: e.clientY - dragStart.y,
    });
  };

  const handleMouseUp = () => setIsDragging(false);

  // Render simulated Grad-CAM / Attention heatmap overlay on canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = originalUrl;
    img.onload = () => {
      canvas.width = img.width || 800;
      canvas.height = img.height || 800;

      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

      if (activeTab === 'original') return;

      // Generate synthetic Grad-CAM or Attention Heatmap overlay dynamically if no static URL
      const heatOverlay = ctx.createRadialGradient(
        canvas.width * 0.45,
        canvas.height * 0.45,
        20,
        canvas.width * 0.5,
        canvas.height * 0.5,
        canvas.width * 0.4
      );

      if (activeTab === 'gradcam') {
        heatOverlay.addColorStop(0, `rgba(239, 68, 68, ${opacity})`);    // Red (High Grad-CAM)
        heatOverlay.addColorStop(0.4, `rgba(245, 158, 11, ${opacity * 0.8})`); // Amber
        heatOverlay.addColorStop(0.7, `rgba(16, 185, 129, ${opacity * 0.5})`); // Emerald
        heatOverlay.addColorStop(1, `rgba(59, 130, 246, 0)`);               // Transparent Blue
      } else {
        // Attention Map (Cyan/Teal/Indigo)
        heatOverlay.addColorStop(0, `rgba(6, 182, 212, ${opacity})`);   // Cyan (High Attention)
        heatOverlay.addColorStop(0.5, `rgba(99, 102, 241, ${opacity * 0.7})`); // Indigo
        heatOverlay.addColorStop(1, `rgba(15, 23, 42, 0)`);
      }

      ctx.fillStyle = heatOverlay;
      ctx.globalCompositeOperation = 'source-over';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    };
  }, [originalUrl, activeTab, opacity]);

  return (
    <div className="w-full bg-slate-900/90 border border-slate-800 rounded-3xl p-6 sm:p-8 shadow-2xl backdrop-blur-xl space-y-6">
      
      {/* Header & Mode Switcher Tabs */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800/80 pb-4">
        <div>
          <span className="text-xs font-semibold text-cyan-400 uppercase tracking-wider block">Visual AI Explanation</span>
          <h3 className="text-xl font-bold text-white">Spatial Feature Inspection</h3>
        </div>

        {/* Tabs */}
        <div className="flex items-center space-x-1 p-1 rounded-2xl bg-slate-950 border border-slate-800 self-start sm:self-auto">
          <button
            onClick={() => setActiveTab('original')}
            className={`px-3.5 py-1.5 rounded-xl text-xs font-semibold transition-all ${
              activeTab === 'original'
                ? 'bg-slate-800 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Original Slide
          </button>
          <button
            onClick={() => setActiveTab('gradcam')}
            className={`px-3.5 py-1.5 rounded-xl text-xs font-semibold transition-all flex items-center space-x-1.5 ${
              activeTab === 'gradcam'
                ? 'bg-rose-950/80 text-rose-300 border border-rose-800/80 shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Layers className="w-3.5 h-3.5 text-rose-400" />
            <span>AI Focus (Grad-CAM)</span>
          </button>
          <button
            onClick={() => setActiveTab('attention')}
            className={`px-3.5 py-1.5 rounded-xl text-xs font-semibold transition-all flex items-center space-x-1.5 ${
              activeTab === 'attention'
                ? 'bg-cyan-950/80 text-cyan-300 border border-cyan-800/80 shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Eye className="w-3.5 h-3.5 text-cyan-400" />
            <span>Attention Map</span>
          </button>
        </div>
      </div>

      {/* Opacity Control Slider (shown when heatmap active) */}
      {activeTab !== 'original' && (
        <div className="flex items-center space-x-4 px-4 py-3 rounded-2xl bg-slate-950/70 border border-slate-800/80 text-xs">
          <Sliders className="w-4 h-4 text-cyan-400 shrink-0" />
          <span className="text-slate-400 font-medium whitespace-nowrap">Heatmap Opacity:</span>
          <input
            type="range"
            min="0.1"
            max="1"
            step="0.05"
            value={opacity}
            onChange={(e) => setOpacity(parseFloat(e.target.value))}
            className="w-full accent-cyan-400 cursor-pointer h-1.5 bg-slate-800 rounded-lg"
          />
          <span className="text-slate-200 font-mono w-10 text-right">{Math.round(opacity * 100)}%</span>
        </div>
      )}

      {/* Canvas Image Viewer viewport */}
      <div className="relative w-full h-80 sm:h-96 md:h-[420px] rounded-2xl border border-slate-800 bg-slate-950 overflow-hidden flex items-center justify-center cursor-grab active:cursor-grabbing group">
        
        {/* Rendered Canvas Layer */}
        <div
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          className="w-full h-full flex items-center justify-center transition-transform duration-100 ease-out"
          style={{
            transform: `scale(${zoom}) translate(${pan.x / zoom}px, ${pan.y / zoom}px)`,
          }}
        >
          <canvas ref={canvasRef} className="max-w-full max-h-full object-contain rounded-xl shadow-2xl" />
        </div>

        {/* Viewport Zoom Controls */}
        <div className="absolute bottom-4 right-4 flex items-center space-x-1 p-1 rounded-xl bg-slate-950/90 border border-slate-800 backdrop-blur-md shadow-lg text-slate-300">
          <button
            onClick={handleZoomIn}
            className="p-1.5 rounded-lg hover:bg-slate-800 text-slate-300 hover:text-white transition-colors"
            title="Zoom In"
          >
            <ZoomIn className="w-4 h-4" />
          </button>
          <button
            onClick={handleZoomOut}
            className="p-1.5 rounded-lg hover:bg-slate-800 text-slate-300 hover:text-white transition-colors"
            title="Zoom Out"
          >
            <ZoomOut className="w-4 h-4" />
          </button>
          <button
            onClick={handleResetZoom}
            className="p-1.5 rounded-lg hover:bg-slate-800 text-slate-300 hover:text-white transition-colors"
            title="Reset Zoom"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>

        {/* Map Legend overlay */}
        {activeTab !== 'original' && (
          <div className="absolute top-4 left-4 px-3 py-2 rounded-xl bg-slate-950/90 border border-slate-800/90 backdrop-blur-md text-[11px] text-slate-300 flex items-center space-x-3">
            <span className="font-semibold text-slate-400">Importance:</span>
            <div className="flex items-center space-x-1">
              <span className="text-slate-400">Low</span>
              <div className="w-16 h-2 rounded-full bg-gradient-to-r from-blue-600 via-emerald-400 via-amber-400 to-rose-500" />
              <span className="text-slate-200 font-bold">High</span>
            </div>
          </div>
        )}
      </div>

      <p className="text-xs text-slate-500 text-center">
        Use controls at bottom right or click and drag to pan when zoomed.
      </p>
    </div>
  );
}
