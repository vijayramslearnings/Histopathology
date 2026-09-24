import { AnalysisResult, AnalysisHistoryItem } from './types';

export const MOCK_ANALYSES: Record<string, AnalysisResult> = {
  'demo-brca-001': {
    analysis_id: 'demo-brca-001',
    status: 'COMPLETED',
    created_at: new Date(Date.now() - 1000 * 60 * 25).toISOString(),
    filename: 'breast_biopsy_slide_042.png',
    image_url: 'https://images.unsplash.com/photo-1579154204601-01588f351e67?auto=format&fit=crop&w=1000&q=80',
    dimensions: {
      width: 2048,
      height: 2048,
    },
    prediction: {
      cancer_type: {
        label: 'BRCA',
        display_name: 'Breast Invasive Carcinoma',
        confidence: 0.914,
      },
      subtype: {
        available: true,
        label: 'IDC_SUBTYPE',
        display_name: 'Infiltrating Ductal Carcinoma (IDC)',
        confidence: 0.862,
      },
      alternatives: [
        { label: 'BRCA', display_name: 'Breast Invasive Carcinoma', probability: 0.914 },
        { label: 'LUAD', display_name: 'Lung Adenocarcinoma', probability: 0.052 },
        { label: 'GBM', display_name: 'Glioblastoma Multiforme', probability: 0.034 },
      ],
    },
    scale_attention: {
      '5x': 0.18,
      '10x': 0.32,
      '20x': 0.50,
    },
    uncertainty: {
      entropy: 0.24,
      level: 'LOW',
      confidence_score: 0.914,
    },
    xai: {
      gradcam_available: true,
      attention_available: true,
      gradcam_url: 'https://images.unsplash.com/photo-1579154204601-01588f351e67?auto=format&fit=crop&w=1000&q=80',
      attention_map_url: 'https://images.unsplash.com/photo-1579154204601-01588f351e67?auto=format&fit=crop&w=1000&q=80',
      top_patches: [
        { patch_id: 'p_01', scale: '20x', x: 256, y: 512, width: 224, height: 224, importance_score: 0.94 },
        { patch_id: 'p_02', scale: '20x', x: 480, y: 512, width: 224, height: 224, importance_score: 0.89 },
        { patch_id: 'p_03', scale: '10x', x: 704, y: 256, width: 224, height: 224, importance_score: 0.85 },
        { patch_id: 'p_04', scale: '20x', x: 256, y: 736, width: 224, height: 224, importance_score: 0.81 },
        { patch_id: 'p_05', scale: '10x', x: 480, y: 736, width: 224, height: 224, importance_score: 0.77 },
        { patch_id: 'p_06', scale: '5x',  x: 1024, y: 1024, width: 224, height: 224, importance_score: 0.72 },
      ],
    },
    research_details: {
      model_name: 'Multi-Scale Attention MIL',
      model_version: 'v1.0.0',
      backbone: 'ResNet-50 (Pretrained ImageNet)',
      scales: ['5x', '10x', '20x'],
      device: 'MPS (Apple Silicon GPU)',
      analysis_id: 'demo-brca-001',
      dataset_name: 'TCGA-BRCA Research Cohort',
    },
    research_disclaimer: 'Experimental AI research output only. Not for clinical diagnosis or therapeutic decisions.',
  },

  'demo-luad-002': {
    analysis_id: 'demo-luad-002',
    status: 'COMPLETED',
    created_at: new Date(Date.now() - 1000 * 60 * 60 * 3).toISOString(),
    filename: 'lung_resection_specimen_108.png',
    image_url: 'https://images.unsplash.com/photo-1530497610245-94d3c16cda28?auto=format&fit=crop&w=1000&q=80',
    dimensions: {
      width: 2048,
      height: 2048,
    },
    prediction: {
      cancer_type: {
        label: 'LUAD',
        display_name: 'Lung Adenocarcinoma',
        confidence: 0.887,
      },
      subtype: {
        available: true,
        label: 'ACINAR_SUBTYPE',
        display_name: 'Acinar Predominant Adenocarcinoma',
        confidence: 0.815,
      },
      alternatives: [
        { label: 'LUAD', display_name: 'Lung Adenocarcinoma', probability: 0.887 },
        { label: 'BRCA', display_name: 'Breast Invasive Carcinoma', probability: 0.076 },
        { label: 'GBM', display_name: 'Glioblastoma Multiforme', probability: 0.037 },
      ],
    },
    scale_attention: {
      '5x': 0.22,
      '10x': 0.45,
      '20x': 0.33,
    },
    uncertainty: {
      entropy: 0.31,
      level: 'LOW',
      confidence_score: 0.887,
    },
    xai: {
      gradcam_available: true,
      attention_available: true,
      top_patches: [
        { patch_id: 'luad_01', scale: '10x', x: 512, y: 512, width: 224, height: 224, importance_score: 0.91 },
        { patch_id: 'luad_02', scale: '10x', x: 736, y: 512, width: 224, height: 224, importance_score: 0.87 },
        { patch_id: 'luad_03', scale: '20x', x: 512, y: 736, width: 224, height: 224, importance_score: 0.82 },
      ],
    },
    research_details: {
      model_name: 'Multi-Scale Attention MIL',
      model_version: 'v1.0.0',
      backbone: 'ResNet-50',
      scales: ['5x', '10x', '20x'],
      device: 'CUDA:0 (NVIDIA RTX)',
      analysis_id: 'demo-luad-002',
      dataset_name: 'TCGA-LUAD Cohort',
    },
    research_disclaimer: 'Experimental AI research output only. Not for clinical diagnosis or therapeutic decisions.',
  },

  'demo-low-003': {
    analysis_id: 'demo-low-003',
    status: 'LOW_CONFIDENCE',
    created_at: new Date(Date.now() - 1000 * 60 * 60 * 24).toISOString(),
    filename: 'necrotic_tissue_fragment_012.png',
    image_url: 'https://images.unsplash.com/photo-1516549655169-df83a0774514?auto=format&fit=crop&w=1000&q=80',
    dimensions: { width: 1024, height: 1024 },
    prediction: {
      cancer_type: { label: 'UNKNOWN', display_name: 'Uncertain Prediction', confidence: 0.42 },
      subtype: { available: false, message: 'Subtype analysis unavailable due to low confidence' },
      alternatives: [
        { label: 'BRCA', display_name: 'Breast Invasive Carcinoma', probability: 0.42 },
        { label: 'LUAD', display_name: 'Lung Adenocarcinoma', probability: 0.35 },
        { label: 'GBM', display_name: 'Glioblastoma Multiforme', probability: 0.23 },
      ],
    },
    scale_attention: { '5x': 0.33, '10x': 0.33, '20x': 0.34 },
    uncertainty: { entropy: 0.92, level: 'HIGH', confidence_score: 0.42 },
    xai: { gradcam_available: false, attention_available: false, top_patches: [] },
    research_details: {
      model_name: 'Multi-Scale Attention MIL',
      model_version: 'v1.0.0',
      backbone: 'ResNet-50',
      scales: ['5x', '10x', '20x'],
      device: 'CPU',
      analysis_id: 'demo-low-003',
    },
    research_disclaimer: 'Experimental AI research output only. Not for clinical diagnosis or therapeutic decisions.',
    message: 'The model confidence was below the configured threshold (0.65). No diagnostic inference was drawn.',
  },
};

export const MOCK_HISTORY: AnalysisHistoryItem[] = [
  {
    analysis_id: 'demo-brca-001',
    filename: 'breast_biopsy_slide_042.png',
    image_url: 'https://images.unsplash.com/photo-1579154204601-01588f351e67?auto=format&fit=crop&w=400&q=80',
    created_at: new Date(Date.now() - 1000 * 60 * 25).toISOString(),
    cancer_type: 'BRCA',
    subtype: 'IDC Subtype',
    confidence: 0.914,
    status: 'COMPLETED',
  },
  {
    analysis_id: 'demo-luad-002',
    filename: 'lung_resection_specimen_108.png',
    image_url: 'https://images.unsplash.com/photo-1530497610245-94d3c16cda28?auto=format&fit=crop&w=400&q=80',
    created_at: new Date(Date.now() - 1000 * 60 * 60 * 3).toISOString(),
    cancer_type: 'LUAD',
    subtype: 'Acinar Subtype',
    confidence: 0.887,
    status: 'COMPLETED',
  },
  {
    analysis_id: 'demo-low-003',
    filename: 'necrotic_tissue_fragment_012.png',
    image_url: 'https://images.unsplash.com/photo-1516549655169-df83a0774514?auto=format&fit=crop&w=400&q=80',
    created_at: new Date(Date.now() - 1000 * 60 * 60 * 24).toISOString(),
    cancer_type: 'Low Confidence',
    confidence: 0.42,
    status: 'LOW_CONFIDENCE',
  },
];
