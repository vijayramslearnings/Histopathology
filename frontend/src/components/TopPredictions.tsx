'use client';

import React, { useState } from 'react';
import { TopPrediction } from '@/lib/types';
import { ChevronDown, ChevronUp, PieChart } from 'lucide-react';
import { formatPercentage } from '@/lib/utils';

interface TopPredictionsProps {
  predictions: TopPrediction[];
}

export function TopPredictions({ predictions }: TopPredictionsProps) {
  const [isOpen, setIsOpen] = useState(false);

  if (!predictions || predictions.length === 0) return null;

  return (
    <div className="w-full bg-slate-950/60 border border-slate-800 rounded-2xl overflow-hidden transition-all">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-6 py-4 flex items-center justify-between text-left hover:bg-slate-900/60 transition-colors"
      >
        <div className="flex items-center space-x-2">
          <PieChart className="w-4 h-4 text-slate-400" />
          <span className="text-xs font-semibold text-slate-300">Other Model Class Predictions</span>
        </div>
        <div className="flex items-center space-x-2 text-slate-400 text-xs font-medium">
          <span>{isOpen ? 'Hide' : 'Show breakdown'}</span>
          {isOpen ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </div>
      </button>

      {isOpen && (
        <div className="px-6 pb-5 pt-2 border-t border-slate-800/80 space-y-3">
          {predictions.map((item) => (
            <div key={item.label} className="flex items-center justify-between text-xs py-1">
              <div className="flex items-center space-x-2">
                <span className="font-mono text-slate-200 font-semibold px-2 py-0.5 rounded bg-slate-900 border border-slate-800">
                  {item.label}
                </span>
                <span className="text-slate-300">{item.display_name}</span>
              </div>
              <span className="font-mono font-bold text-slate-200">
                {formatPercentage(item.probability)}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
