'use client';

import React from 'react';
import Link from 'next/link';
import { UploadCloud, Layers, BrainCircuit, CheckCircle2, Eye, ShieldAlert, Sparkles, ArrowRight } from 'lucide-react';

export default function AboutPage() {
  const steps = [
    {
      num: '01',
      title: 'Upload Slide Image',
      desc: 'High-resolution histopathology microscopy image (PNG, JPG, JPEG) is provided.',
      icon: UploadCloud,
    },
    {
      num: '02',
      title: 'Multi-Scale Feature Extraction',
      desc: 'Simultaneous feature extraction across low (5×), medium (10×), and high (20×) detail contexts.',
      icon: Layers,
    },
    {
      num: '03',
      title: 'AI Prediction',
      desc: 'Learned scale attention and Multiple Instance Learning (MIL) aggregate slide representations.',
      icon: BrainCircuit,
    },
    {
      num: '04',
      title: 'Subtype Analysis',
      desc: 'Hierarchical subtype classifier evaluates validated histological sub-categories.',
      icon: CheckCircle2,
    },
    {
      num: '05',
      title: 'Visual Explanation',
      desc: 'Grad-CAM and patch attention spatial maps highlight decision-critical histological regions.',
      icon: Eye,
    },
  ];

  return (
    <div className="space-y-16 animate-fade-in max-w-4xl mx-auto">
      
      {/* Header */}
      <section className="text-center space-y-6 pt-4">
        <div className="inline-flex items-center space-x-2 px-4 py-1.5 rounded-full bg-slate-900 border border-slate-800 text-xs text-cyan-400 font-medium">
          <Sparkles className="w-3.5 h-3.5" />
          <span>Research Methodology</span>
        </div>

        <h1 className="text-4xl sm:text-5xl font-black tracking-tight text-white">
          About OncoVision<span className="text-cyan-400">.AI</span>
        </h1>

        <p className="text-base sm:text-lg text-slate-300 leading-relaxed max-w-2xl mx-auto">
          OncoVision AI is an experimental research platform exploring explainable multi-scale deep learning for histopathology image analysis.
        </p>
      </section>

      {/* How It Works - 5 Step Process */}
      <section className="bg-slate-900/60 border border-slate-800 rounded-3xl p-8 sm:p-10 shadow-2xl space-y-8 backdrop-blur-xl">
        <div className="border-b border-slate-800 pb-4">
          <span className="text-xs font-semibold text-cyan-400 uppercase tracking-wider block">Workflow Overview</span>
          <h2 className="text-2xl font-bold text-white">How It Works</h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
          {steps.map((step) => {
            const Icon = step.icon;
            return (
              <div
                key={step.num}
                className="bg-slate-950 border border-slate-800/90 rounded-2xl p-5 flex flex-col justify-between space-y-4 hover:border-cyan-800/80 transition-all"
              >
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-xs font-mono font-bold text-slate-500">{step.num}</span>
                    <Icon className="w-5 h-5 text-cyan-400" />
                  </div>
                  <h3 className="text-sm font-bold text-white mb-1">{step.title}</h3>
                  <p className="text-xs text-slate-400 leading-normal">{step.desc}</p>
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* Target Datasets & Architecture */}
      <section className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div className="bg-slate-900/60 border border-slate-800 rounded-3xl p-8 space-y-4 backdrop-blur-xl">
          <h3 className="text-xl font-bold text-white">Supported Cancer Cohorts</h3>
          <ul className="space-y-3 text-xs text-slate-300 font-mono">
            <li className="flex items-center space-x-2">
              <span className="w-2 h-2 rounded-full bg-cyan-400" />
              <span>BRCA — Breast Invasive Carcinoma</span>
            </li>
            <li className="flex items-center space-x-2">
              <span className="w-2 h-2 rounded-full bg-blue-400" />
              <span>LUAD — Lung Adenocarcinoma</span>
            </li>
            <li className="flex items-center space-x-2">
              <span className="w-2 h-2 rounded-full bg-indigo-400" />
              <span>GBM — Glioblastoma Multiforme</span>
            </li>
          </ul>
        </div>

        <div className="bg-slate-900/60 border border-slate-800 rounded-3xl p-8 space-y-4 backdrop-blur-xl">
          <h3 className="text-xl font-bold text-white">Explainability & Faithfulness</h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            The platform combines convolutional feature maps (Grad-CAM) with patch attention mechanisms to quantify faithfulness (deletion/insertion testing) and stability across visual perturbations.
          </p>
        </div>
      </section>

      {/* Research Use Only Banner */}
      <section className="bg-slate-950 border border-slate-800 rounded-3xl p-8 text-center space-y-4">
        <div className="w-12 h-12 mx-auto rounded-xl bg-amber-950/80 border border-amber-800/80 flex items-center justify-center text-amber-400">
          <ShieldAlert className="w-6 h-6" />
        </div>
        <h3 className="text-lg font-bold text-white">Research Disclaimer Notice</h3>
        <p className="text-xs text-slate-400 max-w-xl mx-auto leading-relaxed">
          This system is strictly an experimental research prototype. It is not a clinical diagnostic tool and must not be used to formulate medical diagnoses or guide clinical care.
        </p>

        <div className="pt-2">
          <Link
            href="/analyze"
            className="inline-flex items-center space-x-2 px-6 py-3 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white font-semibold text-xs shadow-lg shadow-cyan-600/20"
          >
            <span>Try Analysis Prototype</span>
            <ArrowRight className="w-4 h-4" />
          </Link>
        </div>
      </section>
    </div>
  );
}
