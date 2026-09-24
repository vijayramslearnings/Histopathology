'use client';

import React from 'react';
import { ScaleAttention as ScaleAttentionType } from '@/lib/types';
import { BarChart2, Layers } from 'lucide-react';
import { formatPercentage } from '@/lib/utils';

interface ScaleAttentionProps {
  scaleAttention: ScaleAttentionType;
}

export function ScaleAttention({ scaleAttention }: ScaleAttentionProps) {
  if (!scaleAttention) return null;

  const scales = [
    {
      key: '5x',
      label: 'Low-detail view (5× context)',
      description: 'Overall tissue architecture & cellular density',
      value: scaleAttention['5x'] || 0.20,
      color: 'from-blue-500 to-indigo-500',
    },
    {
      key: '10x',
      label: 'Medium-detail view (10× structure)',
      description: 'Glandular pattern & histological boundaries',
      value: scaleAttention['10x'] || 0.35,
      color: 'from-cyan-500 to-blue-500',
    },
    {
      key: '20x',
      label: 'High-detail view (20× cellular features)',
      description: 'Nuclear atypia & fine cytological detail',
      value: scaleAttention['20x'] || 0.45,
      color: 'from-teal-400 to-cyan-500',
    },
  ];

  return (
    <div className="w-full bg-slate-900/90 border border-slate-800 rounded-3xl p-6 sm:p-8 shadow-2xl backdrop-blur-xl space-y-6">
      
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-800/80 pb-4">
        <div>
          <span className="text-xs font-semibold text-cyan-400 uppercase tracking-wider block">Multi-Scale Attention</span>
          <h3 className="text-xl font-bold text-white flex items-center space-x-2">
            <Layers className="w-5 h-5 text-cyan-400" />
            <span>How the Model Viewed the Image</span>
          </h3>
        </div>
        <span className="text-xs text-slate-400 font-mono">Learnable Scale Fusion</span>
      </div>

      {/* Horizontal Bar Chart */}
      <div className="space-y-5">
        {scales.map((item) => {
          const pct = Math.round(item.value * 100);
          return (
            <div key={item.key} className="space-y-2">
              <div className="flex justify-between items-baseline text-xs sm:text-sm">
                <div>
                  <span className="font-bold text-slate-100">{item.label}</span>
                  <span className="text-slate-400 text-xs block font-normal">{item.description}</span>
                </div>
                <span className="font-bold font-mono text-cyan-400 text-sm ml-2">{pct}%</span>
              </div>

              {/* Bar track */}
              <div className="w-full h-3.5 rounded-full bg-slate-950 border border-slate-800 overflow-hidden p-0.5">
                <div
                  className={`h-full rounded-full bg-gradient-to-r ${item.color} transition-all duration-1000 shadow-sm`}
                  style={{ width: `${pct}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>

      <p className="text-xs text-slate-500">
        Scale attention weights are dynamically learned by the framework to balance macro tissue context against fine cytological nuclear features.
      </p>
    </div>
  );
}
