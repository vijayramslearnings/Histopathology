'use client';

import React from 'react';
import { AlertTriangle } from 'lucide-react';

export function Disclaimer() {
  return (
    <div className="w-full p-4 rounded-2xl bg-slate-950 border border-slate-800 text-xs text-slate-400 flex items-center space-x-3 shadow-inner">
      <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
      <p className="leading-relaxed">
        <strong className="text-slate-300">Research Use Only:</strong> This AI output is an experimental model prediction intended solely for research inspection. It is not a clinical diagnostic system and must not be used for medical decisions.
      </p>
    </div>
  );
}
