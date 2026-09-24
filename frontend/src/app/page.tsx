'use client';

import React, { useState, useRef } from 'react';
import Link from 'next/link';
import { UploadDropzone } from '@/components/UploadDropzone';
import { AnalysisProgress } from '@/components/AnalysisProgress';
import { PredictionCard } from '@/components/PredictionCard';
import { ImageViewer } from '@/components/ImageViewer';
import { ImportantRegions } from '@/components/ImportantRegions';
import { ScaleAttention } from '@/components/ScaleAttention';
import { TopPredictions } from '@/components/TopPredictions';
import { ResearchDetails } from '@/components/ResearchDetails';
import { Disclaimer } from '@/components/Disclaimer';
import { analyzeHistopathologyImage } from '@/lib/api';
import { AnalysisResult } from '@/lib/types';
import { Sparkles, ArrowDown, RefreshCw, Layers, Eye, ShieldCheck, Microscope, HelpCircle } from 'lucide-react';

export default function AnalyzePage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [organType, setOrganType] = useState<string>('breast');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progressStep, setProgressStep] = useState<string>('');
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);

  const uploadSectionRef = useRef<HTMLDivElement>(null);
  const resultSectionRef = useRef<HTMLDivElement>(null);

  const handleStartAnalysis = async () => {
    if (!selectedFile) return;
    setIsAnalyzing(true);
    setAnalysisResult(null);

    try {
      const result = await analyzeHistopathologyImage(selectedFile, (step) => {
        setProgressStep(step);
      }, organType);
      setAnalysisResult(result);
      
      // Smooth scroll to results
      setTimeout(() => {
        resultSectionRef.current?.scrollIntoView({ behavior: 'smooth' });
      }, 300);
    } catch (err) {
      console.error('Analysis failed:', err);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleResetWorkflow = () => {
    setSelectedFile(null);
    setAnalysisResult(null);
    setIsAnalyzing(false);
    uploadSectionRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const scrollToUpload = () => {
    uploadSectionRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <div className="space-y-12 sm:space-y-16 animate-fade-in">
      
      {/* Hero Header */}
      <section className="text-center space-y-6 pt-4 sm:pt-8 max-w-4xl mx-auto">
        <div className="inline-flex items-center space-x-2 px-4 py-1.5 rounded-full bg-slate-900 border border-slate-800 text-xs text-cyan-400 font-medium shadow-inner">
          <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
          <span>Explainable Multi-Scale Computational Pathology</span>
        </div>

        <h1 className="text-3xl sm:text-5xl lg:text-6xl font-black tracking-tight text-white leading-tight">
          AI-Powered <span className="bg-gradient-to-r from-cyan-400 via-blue-500 to-indigo-400 bg-clip-text text-transparent">Histopathology Analysis</span>
        </h1>

        <p className="text-base sm:text-lg text-slate-400 max-w-2xl mx-auto leading-relaxed">
          Explore experimental cancer and subtype predictions with visual AI explanations, spatial attention heatmaps, and multi-scale feature inspection.
        </p>

        {!analysisResult && !isAnalyzing && (
          <div className="flex flex-wrap items-center justify-center gap-4 pt-2">
            <button
              onClick={scrollToUpload}
              className="px-8 py-4 rounded-2xl bg-gradient-to-r from-cyan-600 via-blue-600 to-indigo-600 hover:from-cyan-500 hover:to-indigo-500 text-white font-bold text-base shadow-xl shadow-cyan-600/25 hover:shadow-cyan-500/35 transition-all duration-300 flex items-center space-x-2"
            >
              <span>Start Analysis</span>
              <ArrowDown className="w-4 h-4" />
            </button>

            <Link
              href="/about"
              className="px-6 py-4 rounded-2xl bg-slate-900 hover:bg-slate-800 text-slate-300 hover:text-white font-semibold text-base border border-slate-800 transition-colors"
            >
              How it works
            </Link>
          </div>
        )}
      </section>

      {/* Main Workflow Section */}
      <div ref={uploadSectionRef} className="scroll-mt-24">
        
        {/* State 1 & 2: Dropzone & Progress */}
        {!analysisResult && (
          <>
            {isAnalyzing ? (
              <AnalysisProgress currentStep={progressStep} />
            ) : (
              <UploadDropzone
                onFileSelect={setSelectedFile}
                selectedFile={selectedFile}
                onClearFile={() => setSelectedFile(null)}
                onStartAnalysis={handleStartAnalysis}
                isAnalyzing={isAnalyzing}
                organType={organType}
                onOrganChange={setOrganType}
              />
            )}
          </>
        )}

        {/* State 3: Analysis Results */}
        {analysisResult && (
          <div ref={resultSectionRef} className="space-y-10 scroll-mt-24">
            
            {/* Top Bar Action */}
            <div className="flex items-center justify-between bg-slate-900/60 border border-slate-800/80 rounded-2xl p-4 px-6 backdrop-blur-md">
              <div className="flex items-center space-x-3">
                <Microscope className="w-5 h-5 text-cyan-400" />
                <div>
                  <h4 className="text-sm font-bold text-white">Active Slide: {analysisResult.filename}</h4>
                  <p className="text-xs text-slate-400">Analysis completed successfully</p>
                </div>
              </div>

              <button
                onClick={handleResetWorkflow}
                className="inline-flex items-center space-x-2 px-4 py-2 rounded-xl bg-cyan-950/80 hover:bg-cyan-900 text-cyan-300 border border-cyan-800/80 text-xs font-semibold transition-all"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                <span>+ New Analysis</span>
              </button>
            </div>

            {/* Layout Grid: Level 1 & 2 (Prediction & ImageViewer) */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
              
              {/* Left Column: Prediction Card */}
              <div className="lg:col-span-5 space-y-6">
                <PredictionCard
                  result={analysisResult}
                  onNewAnalysis={handleResetWorkflow}
                />

                <TopPredictions predictions={analysisResult.prediction.alternatives} />
              </div>

              {/* Right Column: Visual AI Explanation */}
              <div className="lg:col-span-7">
                <ImageViewer
                  originalUrl={analysisResult.image_url}
                  gradcamUrl={analysisResult.xai.gradcam_url}
                  attentionUrl={analysisResult.xai.attention_map_url}
                  altText={analysisResult.filename}
                />
              </div>
            </div>

            {/* Level 5: Important Regions (Top Patches) */}
            {analysisResult.xai.top_patches && analysisResult.xai.top_patches.length > 0 && (
              <ImportantRegions
                patches={analysisResult.xai.top_patches}
                originalUrl={analysisResult.image_url}
              />
            )}

            {/* Level 6: Multi-Scale Scale Attention */}
            <ScaleAttention scaleAttention={analysisResult.scale_attention} />

            {/* Level 7: Research Details (Collapsed) */}
            <ResearchDetails details={analysisResult.research_details} />

            {/* Research Disclaimer */}
            <Disclaimer />
          </div>
        )}
      </div>
    </div>
  );
}
