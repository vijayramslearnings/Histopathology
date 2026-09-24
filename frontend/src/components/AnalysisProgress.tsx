'use client';

import React from 'react';
import { Loader2, Sparkles, CheckCircle, Eye, Cpu, BrainCircuit } from 'lucide-react';

interface AnalysisProgressProps {
  currentStep: string;
}

export function AnalysisProgress({ currentStep }: AnalysisProgressProps) {
  const steps = [
    'Preparing image & tissue segmentation',
    'Extracting multi-scale patch features (5x, 10x, 20x)',
    'Computing scale & patch attention weights',
    'Evaluating cancer & subtype classifiers',
    'Generating Grad-CAM & explanation maps',
  ];

  const currentStepIndex = steps.findIndex((s) => currentStep.includes(s.slice(0, 15))) >= 0
    ? steps.findIndex((s) => currentStep.includes(s.slice(0, 15)))
    : 2;

  return (
    <div className="w-full max-w-2xl mx-auto my-12 p-8 rounded-3xl bg-slate-900/90 border border-slate-800 shadow-2xl backdrop-blur-xl text-center space-y-8 animate-fade-in">
      
      {/* Animated Spinner & Icon */}
      <div className="relative w-24 h-24 mx-auto flex items-center justify-center">
        <div className="absolute inset-0 rounded-full border-4 border-cyan-500/20 border-t-cyan-400 animate-spin" />
        <div className="absolute inset-2 rounded-full border-4 border-blue-500/20 border-b-blue-400 animate-spin duration-700" />
        <div className="w-14 h-14 rounded-full bg-slate-950 border border-cyan-800/60 flex items-center justify-center text-cyan-400 shadow-inner">
          <BrainCircuit className="w-7 h-7 animate-pulse" />
        </div>
      </div>

      <div>
        <h3 className="text-xl font-bold text-white mb-2">Analyzing Histopathology Tissue</h3>
        <p className="text-cyan-400 text-sm font-medium animate-pulse">
          {currentStep || 'Multi-scale neural network evaluation in progress...'}
        </p>
      </div>

      {/* Indeterminate Shimmer Bar */}
      <div className="w-full h-2 rounded-full bg-slate-950 overflow-hidden relative border border-slate-800">
        <div className="absolute inset-y-0 bg-gradient-to-r from-cyan-500 via-blue-500 to-indigo-500 rounded-full w-1/3 animate-[shimmer_1.5s_infinite] left-0" style={{
          animation: 'shimmer 1.8s ease-in-out infinite alternate'
        }} />
      </div>

      {/* Steps checklist */}
      <div className="space-y-3 text-left pt-4 border-t border-slate-800/80">
        {steps.map((step, idx) => {
          const isDone = idx < currentStepIndex;
          const isCurrent = idx === currentStepIndex;
          return (
            <div
              key={step}
              className={`flex items-center space-x-3 text-xs sm:text-sm py-1.5 transition-all ${
                isDone
                  ? 'text-slate-400'
                  : isCurrent
                  ? 'text-cyan-300 font-semibold'
                  : 'text-slate-600'
              }`}
            >
              <div className="w-5 h-5 rounded-full flex items-center justify-center shrink-0">
                {isDone ? (
                  <CheckCircle className="w-4 h-4 text-emerald-400" />
                ) : isCurrent ? (
                  <Loader2 className="w-4 h-4 text-cyan-400 animate-spin" />
                ) : (
                  <div className="w-2 h-2 rounded-full bg-slate-800" />
                )}
              </div>
              <span>{step}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
