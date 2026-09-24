'use client';

import React from 'react';
import { ImportantPatch } from '@/lib/types';
import { Target, Search } from 'lucide-react';
import { formatPercentage } from '@/lib/utils';

interface ImportantRegionsProps {
  patches: ImportantPatch[];
  originalUrl: string;
}

export function ImportantRegions({ patches, originalUrl }: ImportantRegionsProps) {
  if (!patches || patches.length === 0) return null;

  return (
    <div className="w-full bg-slate-900/90 border border-slate-800 rounded-3xl p-6 sm:p-8 shadow-2xl backdrop-blur-xl space-y-6">
      
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-800/80 pb-4">
        <div>
          <span className="text-xs font-semibold text-cyan-400 uppercase tracking-wider block">Patch Attention Ranking</span>
          <h3 className="text-xl font-bold text-white flex items-center space-x-2">
            <Target className="w-5 h-5 text-cyan-400" />
            <span>Important Regions (Model Focus)</span>
          </h3>
        </div>
        <span className="text-xs text-slate-400 font-mono">Top-{patches.length} Tissue Patches</span>
      </div>

      {/* Grid of Patches */}
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-4">
        {patches.map((patch, idx) => (
          <div
            key={patch.patch_id}
            className="group relative bg-slate-950 border border-slate-800 hover:border-cyan-800/80 rounded-2xl p-2.5 transition-all duration-300 hover:scale-[1.03] shadow-md hover:shadow-cyan-500/10"
          >
            {/* Patch Image Container */}
            <div className="relative aspect-square rounded-xl overflow-hidden bg-slate-900 mb-2 border border-slate-800">
              <img
                src={patch.patch_url || originalUrl}
                alt={`Model focus patch ${patch.patch_id}`}
                className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500"
              />
              <span className="absolute top-1.5 left-1.5 px-2 py-0.5 text-[10px] font-bold rounded-md bg-slate-950/80 text-cyan-300 border border-slate-800 font-mono">
                {patch.scale}
              </span>
            </div>

            {/* Metadata */}
            <div className="space-y-1 text-xs">
              <div className="flex justify-between items-center text-[11px]">
                <span className="text-slate-400 font-medium">Rank #{idx + 1}</span>
                <span className="text-cyan-400 font-bold font-mono">
                  {formatPercentage(patch.importance_score)}
                </span>
              </div>

              <p className="text-[10px] text-slate-500 font-mono truncate">
                ({patch.x}, {patch.y})
              </p>
            </div>
          </div>
        ))}
      </div>

      <p className="text-xs text-slate-500">
        These top histopathological patches received the highest spatial attention weights during Multiple Instance Learning (MIL) pooling.
      </p>
    </div>
  );
}
