'use client';

import React, { use } from 'react';
import Link from 'next/link';
import { getAnalysisById } from '@/lib/api';
import { PredictionCard } from '@/components/PredictionCard';
import { ImageViewer } from '@/components/ImageViewer';
import { ImportantRegions } from '@/components/ImportantRegions';
import { ScaleAttention } from '@/components/ScaleAttention';
import { TopPredictions } from '@/components/TopPredictions';
import { ResearchDetails } from '@/components/ResearchDetails';
import { Disclaimer } from '@/components/Disclaimer';
import { ChatbotPanel } from '@/components/ChatbotPanel';
import { ArrowLeft, RefreshCw } from 'lucide-react';

interface ResultPageProps {
  params: Promise<{ id: string }>;
}

export default function ResultDetailPage({ params }: ResultPageProps) {
  const { id } = use(params);
  const analysisResult = getAnalysisById(id);

  if (!analysisResult) {
    return (
      <div className="text-center py-20 space-y-6">
        <h2 className="text-2xl font-bold text-white">Analysis Record Not Found</h2>
        <p className="text-slate-400 text-sm">No recorded analysis matches ID: {id}</p>
        <Link
          href="/analyze"
          className="inline-flex items-center space-x-2 px-6 py-3 rounded-xl bg-cyan-600 text-white font-semibold text-sm"
        >
          <ArrowLeft className="w-4 h-4" />
          <span>Start New Analysis</span>
        </Link>
      </div>
    );
  }

  return (
    <div className="space-y-10 animate-fade-in">
      
      {/* Navigation Header */}
      <div className="flex items-center justify-between bg-slate-900/60 border border-slate-800/80 rounded-2xl p-4 px-6 backdrop-blur-md">
        <div className="flex items-center space-x-3">
          <Link
            href="/history"
            className="p-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors"
            title="Back to History"
          >
            <ArrowLeft className="w-4 h-4" />
          </Link>
          <div>
            <h4 className="text-sm font-bold text-white">Slide: {analysisResult.filename}</h4>
            <p className="text-xs text-slate-400 font-mono">ID: {analysisResult.analysis_id}</p>
          </div>
        </div>

        <Link
          href="/analyze"
          className="inline-flex items-center space-x-2 px-4 py-2 rounded-xl bg-cyan-950/80 hover:bg-cyan-900 text-cyan-300 border border-cyan-800/80 text-xs font-semibold transition-all"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          <span>+ New Analysis</span>
        </Link>
      </div>

      {/* Grid: Level 1 & 2 */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        <div className="lg:col-span-5 space-y-6">
          <PredictionCard result={analysisResult} />
          <TopPredictions predictions={analysisResult.prediction.alternatives} />
        </div>

        <div className="lg:col-span-7">
          <ImageViewer
            originalUrl={analysisResult.image_url}
            gradcamUrl={analysisResult.xai.gradcam_url}
            attentionUrl={analysisResult.xai.attention_map_url}
            altText={analysisResult.filename}
          />
        </div>
      </div>

      {/* Chatbot & Important Regions Split Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        <div className="lg:col-span-7 space-y-6">
          {analysisResult.xai.top_patches && analysisResult.xai.top_patches.length > 0 && (
            <ImportantRegions
              patches={analysisResult.xai.top_patches}
              originalUrl={analysisResult.image_url}
            />
          )}
          <ScaleAttention scaleAttention={analysisResult.scale_attention} />
        </div>

        <div className="lg:col-span-5">
          <ChatbotPanel
            analysisResult={analysisResult}
            sessionId={analysisResult.analysis_id}
          />
        </div>
      </div>

      {/* Technical Research Details */}
      <ResearchDetails details={analysisResult.research_details} />

      {/* Disclaimer */}
      <Disclaimer />
    </div>
  );
}
