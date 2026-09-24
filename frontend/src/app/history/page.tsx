'use client';

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { getHistoryFromStorage } from '@/lib/api';
import { AnalysisHistoryItem } from '@/lib/types';
import { Search, Filter, History as HistoryIcon, ArrowRight, Calendar, Sparkles, FileText } from 'lucide-react';
import { formatDate, formatPercentage } from '@/lib/utils';

export default function HistoryPage() {
  const [historyItems, setHistoryItems] = useState<AnalysisHistoryItem[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [cancerFilter, setCancerFilter] = useState<string>('ALL');

  useEffect(() => {
    setHistoryItems(getHistoryFromStorage());
  }, []);

  const filteredItems = historyItems.filter((item) => {
    const matchesSearch =
      item.analysis_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
      item.filename.toLowerCase().includes(searchQuery.toLowerCase());

    const matchesFilter =
      cancerFilter === 'ALL' || item.cancer_type.toUpperCase().includes(cancerFilter);

    return matchesSearch && matchesFilter;
  });

  return (
    <div className="space-y-8 animate-fade-in">
      
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-6">
        <div>
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-slate-900 border border-slate-800 text-xs text-cyan-400 font-medium mb-2">
            <HistoryIcon className="w-3.5 h-3.5" />
            <span>Analysis Archive</span>
          </div>
          <h1 className="text-3xl font-extrabold text-white">Analysis History</h1>
          <p className="text-slate-400 text-sm">Browse past research slide inferences and generated XAI explanations.</p>
        </div>

        <Link
          href="/analyze"
          className="inline-flex items-center space-x-2 px-5 py-2.5 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white font-semibold text-xs shadow-lg shadow-cyan-600/20 transition-all self-start sm:self-auto"
        >
          <span>+ New Analysis</span>
        </Link>
      </div>

      {/* Filter & Search Controls */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-4 bg-slate-900/60 p-4 rounded-2xl border border-slate-800">
        
        {/* Search Input */}
        <div className="relative w-full sm:w-80">
          <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-3" />
          <input
            type="text"
            placeholder="Search by analysis ID or filename..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-2 bg-slate-950 border border-slate-800 rounded-xl text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500 transition-colors"
          />
        </div>

        {/* Cancer Type Filter */}
        <div className="flex items-center space-x-2 w-full sm:w-auto">
          <Filter className="w-4 h-4 text-slate-400 shrink-0" />
          <div className="flex items-center space-x-1 p-1 bg-slate-950 border border-slate-800 rounded-xl">
            {['ALL', 'BRCA', 'LUAD', 'GBM'].map((type) => (
              <button
                key={type}
                onClick={() => setCancerFilter(type)}
                className={`px-3 py-1 rounded-lg text-xs font-semibold transition-all ${
                  cancerFilter === type
                    ? 'bg-slate-800 text-cyan-400 border border-slate-700'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {type}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* History Grid */}
      {filteredItems.length === 0 ? (
        /* Empty State */
        <div className="text-center py-16 bg-slate-900/40 border border-slate-800 rounded-3xl p-8 space-y-4">
          <div className="w-16 h-16 mx-auto rounded-2xl bg-slate-900 border border-slate-800 flex items-center justify-center text-slate-500">
            <FileText className="w-8 h-8" />
          </div>
          <h3 className="text-xl font-bold text-white">No Analyses Recorded</h3>
          <p className="text-slate-400 text-sm max-w-sm mx-auto">
            Upload a histopathology microscopy slide to generate your first multi-scale AI prediction.
          </p>
          <Link
            href="/analyze"
            className="inline-flex items-center space-x-2 px-6 py-3 rounded-xl bg-gradient-to-r from-cyan-600 to-blue-600 text-white font-semibold text-sm shadow-lg shadow-cyan-600/20"
          >
            <span>Start Analysis</span>
          </Link>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredItems.map((item) => (
            <div
              key={item.analysis_id}
              className="group bg-slate-900/80 border border-slate-800 hover:border-slate-700 rounded-3xl p-5 shadow-xl backdrop-blur-md flex flex-col justify-between transition-all hover:scale-[1.01]"
            >
              <div className="space-y-4">
                
                {/* Thumbnail & Badges */}
                <div className="relative aspect-video rounded-2xl overflow-hidden bg-slate-950 border border-slate-800">
                  <img
                    src={item.image_url}
                    alt={item.filename}
                    className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
                  />
                  <div className="absolute top-3 right-3 px-2.5 py-1 rounded-full bg-slate-950/80 backdrop-blur-md border border-slate-800 font-mono text-[11px] font-bold text-cyan-400">
                    {formatPercentage(item.confidence)}
                  </div>
                </div>

                {/* Info */}
                <div>
                  <div className="flex items-center justify-between text-xs text-slate-500 mb-1">
                    <span className="font-mono">ID: {item.analysis_id.slice(0, 14)}</span>
                    <span className="flex items-center space-x-1">
                      <Calendar className="w-3 h-3" />
                      <span>{formatDate(item.created_at)}</span>
                    </span>
                  </div>

                  <h3 className="text-lg font-bold text-white truncate">{item.cancer_type}</h3>
                  {item.subtype && (
                    <p className="text-xs text-slate-400 truncate mt-0.5">{item.subtype}</p>
                  )}
                </div>
              </div>

              {/* View Action */}
              <div className="pt-4 mt-4 border-t border-slate-800/80">
                <Link
                  href={`/results/${item.analysis_id}`}
                  className="w-full inline-flex items-center justify-center space-x-2 py-2.5 px-4 rounded-xl bg-slate-950 hover:bg-slate-800 text-slate-300 hover:text-cyan-300 text-xs font-semibold border border-slate-800 transition-colors"
                >
                  <span>View Full Analysis</span>
                  <ArrowRight className="w-3.5 h-3.5 text-cyan-400" />
                </Link>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
