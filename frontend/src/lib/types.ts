export type AnalysisStatus = 'COMPLETED' | 'LOW_CONFIDENCE' | 'FAILED' | 'PROCESSING';

export interface CancerPrediction {
  label: string;
  display_name: string;
  confidence: number;
}

export interface SubtypePrediction {
  available: boolean;
  label?: string;
  display_name?: string;
  confidence?: number;
  message?: string;
}

export interface TopPrediction {
  label: string;
  display_name: string;
  probability: number;
}

export interface ScaleAttention {
  '5x': number;
  '10x': number;
  '20x': number;
}

export interface UncertaintyInfo {
  entropy: number;
  level: 'LOW' | 'MEDIUM' | 'HIGH';
  confidence_score: number;
}

export interface ImportantPatch {
  patch_id: string;
  scale: '5x' | '10x' | '20x';
  x: number;
  y: number;
  width: number;
  height: number;
  importance_score: number; // 0.0 - 1.0
  patch_url?: string;
}

export interface XAIArtifacts {
  gradcam_available: boolean;
  attention_available: boolean;
  gradcam_url?: string;
  attention_map_url?: string;
  top_patches: ImportantPatch[];
}

export interface ResearchDetails {
  model_name: string;
  model_version: string;
  backbone: string;
  scales: string[];
  device: string;
  analysis_id: string;
  dataset_name?: string;
}

export interface AnalysisResult {
  analysis_id: string;
  status: AnalysisStatus;
  created_at: string;
  filename: string;
  image_url: string;
  dimensions: {
    width: number;
    height: number;
  };
  prediction: {
    cancer_type: CancerPrediction;
    subtype: SubtypePrediction;
    alternatives: TopPrediction[];
  };
  scale_attention: ScaleAttention;
  uncertainty: UncertaintyInfo;
  xai: XAIArtifacts;
  research_details: ResearchDetails;
  research_disclaimer: string;
  message?: string;
}

export interface AnalysisHistoryItem {
  analysis_id: string;
  filename: string;
  image_url: string;
  created_at: string;
  cancer_type: string;
  subtype?: string;
  confidence: number;
  status: AnalysisStatus;
}
