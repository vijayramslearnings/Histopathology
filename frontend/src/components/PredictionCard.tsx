'use client';

import React from 'react';
import { AnalysisResult } from '@/lib/types';
import { ShieldAlert, CheckCircle2, Award, Sparkles, HelpCircle, ArrowRight } from 'lucide-react';
import { formatPercentage } from '@/lib/utils';

interface PredictionCardProps {
  result: AnalysisResult;
  onExplainClick?: () => void;
  onNewAnalysis?: () => void;
}

export function PredictionCard({ result, onExplainClick, onNewAnalysis }: PredictionCardProps) {
  const { prediction, status, uncertainty } = result;
  const isLowConfidence = status === 'LOW_CONFIDENCE' || prediction.cancer_type.confidence < 0.60;

  if (isLowConfidence) {
    return (
      <div className="w-full bg-slate-900/90 border border-amber-800/60 rounded-3xl p-8 shadow-2xl backdrop-blur-xl text-center space-y-6">
        <div className="w-16 h-16 mx-auto rounded-2xl bg-amber-950/80 border border-amber-700/60 flex items-center justify-center text-amber-400 shadow-inner">
          <ShieldAlert className="w-8 h-8" />
        </div>

        <div>
          <span className="px-3 py-1 text-xs font-semibold tracking-wider text-amber-300 uppercase bg-amber-950/80 border border-amber-800/80 rounded-full">
            Low Confidence Assessment
          </span>
          <h3 className="text-2xl font-bold text-white mt-3 mb-2">Uncertain Tissue Features</h3>
          <p className="text-slate-300 text-sm max-w-md mx-auto leading-relaxed">
            The model could not establish a reliable prediction for this slide section. Tissue characteristics or image clarity did not meet configured confidence criteria.
          </p>
        </div>

        <div className="p-4 rounded-2xl bg-slate-950 border border-slate-800 max-w-xs mx-auto text-xs space-y-1 text-slate-400">
          <div className="flex justify-between">
            <span>Model Confidence:</span>
            <span className="font-mono text-amber-400 font-semibold">{formatPercentage(prediction.cancer_type.confidence)}</span>
          </div>
          <div className="flex justify-between">
            <span>Uncertainty Level:</span>
            <span className="font-mono text-slate-300 font-semibold">{uncertainty.level}</span>
          </div>
        </div>

        <div className="pt-2">
          {onNewAnalysis && (
            <button
              onClick={onNewAnalysis}
              className="inline-flex items-center space-x-2 px-6 py-3 rounded-xl bg-slate-800 hover:bg-slate-700 text-white font-semibold text-sm transition-colors"
            >
              <span>Try Another Image</span>
            </button>
          )}
        </div>
      </div>
    );
  }

  const confidencePct = Math.round(prediction.cancer_type.confidence * 100);

  return (
    <div className="w-full bg-slate-900/90 border border-slate-800 rounded-3xl p-6 sm:p-8 shadow-2xl backdrop-blur-xl space-y-6 relative overflow-hidden">
      {/* Decorative subtle background glow */}
      <div className="absolute -top-24 -right-24 w-60 h-60 bg-blue-600/10 rounded-full blur-3xl pointer-events-none" />

      {/* Card Header */}
      <div className="flex items-center justify-between border-b border-slate-800/80 pb-4">
        <div className="flex items-center space-x-2">
          <Sparkles className="w-5 h-5 text-cyan-400" />
          <h3 className="text-xs font-bold tracking-wider text-slate-400 uppercase">AI Analysis Result</h3>
        </div>
        <span className="text-xs text-slate-500 font-mono">ID: {result.analysis_id.slice(0, 12)}</span>
      </div>

      {/* Primary Prediction Block */}
      <div className="space-y-4">
        <div>
          <span className="text-xs font-medium text-slate-400 block mb-1">Predicted Cancer Type</span>
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <h2 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
              {prediction.cancer_type.display_name}
            </h2>
            <span className="text-xs font-semibold px-3 py-1 rounded-full bg-cyan-950/80 text-cyan-300 border border-cyan-800/60 font-mono">
              {prediction.cancer_type.label}
            </span>
          </div>
        </div>

        {/* Confidence Meter Bar */}
        <div className="space-y-1.5 pt-1">
          <div className="flex justify-between text-xs">
            <span className="text-slate-400 font-medium">Model Confidence</span>
            <span className="text-cyan-400 font-bold font-mono">{confidencePct}%</span>
          </div>
          <div className="w-full h-3 rounded-full bg-slate-950 border border-slate-800 overflow-hidden p-0.5">
            <div
              className="h-full rounded-full bg-gradient-to-r from-cyan-500 via-blue-500 to-indigo-500 transition-all duration-1000 shadow-sm"
              style={{ width: `${confidencePct}%` }}
            />
          </div>
        </div>
      </div>

      <hr className="border-slate-800/80" />

      {/* Subtype Block */}
      <div className="p-4 rounded-2xl bg-slate-950/70 border border-slate-800/90 space-y-2">
        <span className="text-xs font-semibold text-slate-400 block uppercase tracking-wider">Subtype Classification</span>

        {prediction.subtype.available && prediction.subtype.display_name ? (
          <div className="flex items-center justify-between flex-wrap gap-2">
            <div>
              <p className="text-base font-bold text-slate-100">{prediction.subtype.display_name}</p>
              {prediction.subtype.confidence && (
                <p className="text-xs text-slate-400 font-mono">
                  Subtype confidence: {formatPercentage(prediction.subtype.confidence)}
                </p>
              )}
            </div>
            <span className="px-2.5 py-1 text-xs rounded-lg bg-slate-900 text-emerald-400 border border-slate-800 font-medium flex items-center space-x-1">
              <CheckCircle2 className="w-3.5 h-3.5" />
              <span>Validated Label</span>
            </span>
          </div>
        ) : (
          <p className="text-sm text-slate-400 italic">
            {prediction.subtype.message || 'Subtype analysis unavailable for this dataset configuration.'}
          </p>
        )}
      </div>

      {/* Explanation CTA Button */}
      {onExplainClick && (
        <div className="pt-2">
          <button
            onClick={onExplainClick}
            className="w-full inline-flex items-center justify-center space-x-2 py-3.5 px-6 rounded-2xl bg-slate-800 hover:bg-slate-700 text-cyan-300 hover:text-cyan-200 font-semibold text-sm border border-slate-700/60 shadow-lg transition-all duration-200 group"
          >
            <span>Why this prediction? View Explanation</span>
            <ArrowRight className="w-4 h-4 text-cyan-400 group-hover:translate-x-1 transition-transform" />
          </button>
        </div>
      )}
    </div>
  );
}
