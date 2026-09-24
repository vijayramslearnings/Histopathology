'use client';

import React from 'react';
import Link from 'next/link';

export function Footer() {
  return (
    <footer className="w-full border-t border-slate-800/80 bg-slate-950 py-8 text-xs text-slate-500">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
        <div>
          <span className="font-semibold text-slate-300">OncoVision AI Framework</span>
          <span className="ml-2 text-slate-500">• Multi-Scale Computational Pathology Research Prototype</span>
        </div>

        <div className="flex items-center space-x-6">
          <Link href="/analyze" className="hover:text-slate-300 transition-colors">Analyze</Link>
          <Link href="/history" className="hover:text-slate-300 transition-colors">History</Link>
          <Link href="/about" className="hover:text-slate-300 transition-colors">About & Methodology</Link>
        </div>
      </div>
    </footer>
  );
}
