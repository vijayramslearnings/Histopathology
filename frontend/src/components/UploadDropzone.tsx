'use client';

import React, { useState, useRef, DragEvent, ChangeEvent } from 'react';
import { UploadCloud, Image as ImageIcon, X, ArrowRight, AlertCircle, FileCheck, Stethoscope } from 'lucide-react';

interface UploadDropzoneProps {
  onFileSelect: (file: File) => void;
  selectedFile: File | null;
  onClearFile: () => void;
  onStartAnalysis: () => void;
  isAnalyzing: boolean;
  organType?: string;
  onOrganChange?: (organ: string) => void;
}

export function UploadDropzone({
  onFileSelect,
  selectedFile,
  onClearFile,
  onStartAnalysis,
  isAnalyzing,
  organType = 'auto',
  onOrganChange,
}: UploadDropzoneProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [imageMeta, setImageMeta] = useState<{ width: number; height: number; sizeMb: string } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const ALLOWED_TYPES = ['image/png', 'image/jpeg', 'image/jpg'];
  const MAX_SIZE_MB = 25;

  const validateAndHandleFile = (file: File) => {
    setErrorMsg(null);
    if (!ALLOWED_TYPES.includes(file.type)) {
      setErrorMsg('Unsupported format. Please upload a PNG, JPG, or JPEG histopathology image.');
      return;
    }
    if (file.size > MAX_SIZE_MB * 1024 * 1024) {
      setErrorMsg(`File size exceeds limit (${MAX_SIZE_MB}MB maximum).`);
      return;
    }

    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);

    // Read image dimensions
    const img = new Image();
    img.src = objectUrl;
    img.onload = () => {
      setImageMeta({
        width: img.width,
        height: img.height,
        sizeMb: (file.size / (1024 * 1024)).toFixed(2),
      });
    };

    onFileSelect(file);
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndHandleFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      validateAndHandleFile(e.target.files[0]);
    }
  };

  const handleClear = () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    setImageMeta(null);
    setErrorMsg(null);
    onClearFile();
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  return (
    <div className="w-full max-w-3xl mx-auto space-y-6">
      
      {/* Organ / Tissue Site Selection Card */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-4 px-6 backdrop-blur-md flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="flex items-center space-x-3">
          <div className="p-2 rounded-xl bg-cyan-950/80 border border-cyan-800/80 text-cyan-400">
            <Stethoscope className="w-5 h-5" />
          </div>
          <div>
            <h4 className="text-sm font-bold text-white">Select Tissue / Organ Origin</h4>
            <p className="text-xs text-slate-400">Specify tissue type for targeted cancer and subtype classification</p>
          </div>
        </div>

        <select
          value={organType}
          onChange={(e) => onOrganChange && onOrganChange(e.target.value)}
          className="bg-slate-950 border border-slate-700/80 text-cyan-300 text-xs font-semibold rounded-xl px-4 py-2.5 focus:outline-none focus:border-cyan-500 cursor-pointer"
        >
          <option value="breast">Breast Tissue (BRCA)</option>
          <option value="lung">Lung Tissue (LUAD)</option>
          <option value="brain">Brain Tissue (GBM)</option>
          <option value="auto">Auto-Detect Organ</option>
        </select>
      </div>

      {!selectedFile ? (
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`relative border-2 border-dashed rounded-3xl p-8 sm:p-12 text-center cursor-pointer transition-all duration-300 ${
            isDragOver
              ? 'border-cyan-400 bg-cyan-950/30 scale-[1.01] shadow-2xl shadow-cyan-500/10'
              : 'border-slate-800 hover:border-slate-700 bg-slate-900/40 hover:bg-slate-900/60'
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/png, image/jpeg, image/jpg"
            onChange={handleFileInputChange}
            className="hidden"
          />

          <div className="w-16 h-16 mx-auto mb-6 rounded-2xl bg-gradient-to-tr from-cyan-950 to-blue-900/60 border border-cyan-800/40 flex items-center justify-center text-cyan-400 shadow-inner">
            <UploadCloud className="w-8 h-8" />
          </div>

          <h3 className="text-xl font-bold text-white mb-2">Upload Histopathology Image</h3>
          <p className="text-slate-400 text-sm mb-6 max-w-md mx-auto">
            Drag & drop your microscopy slide image here or click below to browse files
          </p>

          <button
            type="button"
            className="inline-flex items-center space-x-2 px-6 py-3 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white font-semibold text-sm shadow-lg shadow-cyan-600/30 hover:shadow-cyan-500/40 transition-all duration-200"
          >
            <ImageIcon className="w-4 h-4" />
            <span>Choose Image</span>
          </button>

          <p className="text-xs text-slate-500 mt-6 font-mono">
            Supported Formats: PNG • JPG • JPEG (Up to 25MB)
          </p>

          {errorMsg && (
            <div className="mt-4 p-3 rounded-xl bg-rose-950/60 border border-rose-800/60 text-rose-300 text-xs flex items-center justify-center space-x-2">
              <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
              <span>{errorMsg}</span>
            </div>
          )}
        </div>
      ) : (
        /* Image Preview State */
        <div className="bg-slate-900/90 border border-slate-800 rounded-3xl p-6 sm:p-8 shadow-2xl backdrop-blur-xl">
          <div className="flex items-center justify-between pb-4 mb-6 border-b border-slate-800">
            <div className="flex items-center space-x-3">
              <div className="w-9 h-9 rounded-lg bg-cyan-950 border border-cyan-800/60 flex items-center justify-center text-cyan-400">
                <FileCheck className="w-5 h-5" />
              </div>
              <div>
                <h4 className="text-sm font-semibold text-white">Image Loaded Successfully</h4>
                <p className="text-xs text-slate-400">Ready for {organType.toUpperCase()} tissue cancer and subtype analysis</p>
              </div>
            </div>

            <button
              onClick={handleClear}
              disabled={isAnalyzing}
              className="p-2 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors disabled:opacity-50"
              title="Remove image"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-12 gap-6 items-center">
            {/* Thumbnail */}
            <div className="md:col-span-5 relative group rounded-2xl overflow-hidden border border-slate-800 bg-slate-950 aspect-square max-h-64 mx-auto md:mx-0 w-full flex items-center justify-center">
              {previewUrl && (
                <img
                  src={previewUrl}
                  alt="Histopathology Slide Preview"
                  className="w-full h-full object-cover rounded-2xl"
                />
              )}
            </div>

            {/* File Info & Action */}
            <div className="md:col-span-7 space-y-4">
              <div>
                <span className="text-xs font-semibold text-cyan-400 uppercase tracking-wider">Source File</span>
                <h3 className="text-lg font-bold text-white truncate max-w-xs">{selectedFile.name}</h3>
              </div>

              {imageMeta && (
                <div className="grid grid-cols-2 gap-3 py-3 border-y border-slate-800/80 text-xs">
                  <div>
                    <span className="text-slate-500 block">Dimensions</span>
                    <span className="font-mono text-slate-200 font-medium">
                      {imageMeta.width} × {imageMeta.height} px
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-500 block">File Size</span>
                    <span className="font-mono text-slate-200 font-medium">{imageMeta.sizeMb} MB</span>
                  </div>
                </div>
              )}

              <div className="pt-2">
                <button
                  onClick={onStartAnalysis}
                  disabled={isAnalyzing}
                  className="w-full inline-flex items-center justify-center space-x-2 py-4 px-6 rounded-2xl bg-gradient-to-r from-cyan-600 via-blue-600 to-indigo-600 hover:from-cyan-500 hover:to-indigo-500 text-white font-bold text-base shadow-xl shadow-cyan-600/20 hover:shadow-cyan-500/30 transition-all duration-300 disabled:opacity-60"
                >
                  <span>{isAnalyzing ? 'Analyzing Image...' : `Analyze ${organType.toUpperCase()} Slide`}</span>
                  <ArrowRight className="w-5 h-5" />
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
