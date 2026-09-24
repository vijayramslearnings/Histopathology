'use client';

import React, { useState } from 'react';
import { ResearchDetails as ResearchDetailsType } from '@/lib/types';
import { ChevronDown, ChevronUp, Terminal, Cpu, Database, HardDrive, Shield } from 'lucide-react';

interface ResearchDetailsProps {
  details: ResearchDetailsType;
}

export function ResearchDetails({ details }: ResearchDetailsProps) {
  const [isOpen, setIsOpen] = useState(false);

  if (!details) return null;

  return (
    <div className="w-full bg-slate-950/80 border border-slate-800 rounded-2xl overflow-hidden transition-all">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-6 py-4 flex items-center justify-between text-left hover:bg-slate-900/60 transition-colors"
      >
        <div className="flex items-center space-x-2">
          <Terminal className="w-4 h-4 text-cyan-400" />
          <span className="text-xs font-semibold text-slate-300">Technical Research Details</span>
        </div>
        <div className="flex items-center space-x-2 text-slate-400 text-xs font-medium">
          <span>{isOpen ? 'Collapse' : 'Expand technical specs'}</span>
          {isOpen ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </div>
      </button>

      {isOpen && (
        <div className="px-6 pb-6 pt-2 border-t border-slate-800/80 grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4 text-xs font-mono">
          <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
            <span className="text-slate-500 block text-[10px] uppercase">Model Architecture</span>
            <span className="text-slate-200 font-semibold">{details.model_name}</span>
          </div>

          <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
            <span className="text-slate-500 block text-[10px] uppercase">Backbone Feature Extractor</span>
            <span className="text-slate-200 font-semibold">{details.backbone}</span>
          </div>

          <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
            <span className="text-slate-500 block text-[10px] uppercase">Scales Evaluated</span>
            <span className="text-slate-200 font-semibold">{details.scales.join(' / ')}</span>
          </div>

          <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
            <span className="text-slate-500 block text-[10px] uppercase">Active Compute Device</span>
            <span className="text-cyan-400 font-semibold">{details.device}</span>
          </div>

          <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
            <span className="text-slate-500 block text-[10px] uppercase">Framework Version</span>
            <span className="text-slate-200 font-semibold">{details.model_version}</span>
          </div>

          <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
            <span className="text-slate-500 block text-[10px] uppercase">Analysis UUID</span>
            <span className="text-slate-400 truncate block">{details.analysis_id}</span>
          </div>
        </div>
      )}
    </div>
  );
}
