export interface Explanation {
  summary: string;
  evidence: string[];
  uncertainty: string;
  confidence: number;
}

export interface Provenance {
  exif: { present: boolean; fields: Record<string, unknown> };
  c2pa: { present: boolean; status: string; note: string };
}

export interface RobustnessCheck {
  included: boolean;
  conditions: Record<string, { probability_ai?: number; verdict?: string; error?: string }>;
}

export interface PredictResponse {
  verdict: string;
  label: string;
  binary_label?: string;
  probability_ai: number;
  probability_real: number;
  confidence: number;
  threshold: number;
  status: string;
  model: { name: string; architecture: string };
  explanation?: Explanation | null;
  visualization?: { original?: string; heatmap?: string; overlay?: string } | null;
  provenance?: Provenance | null;
  robustness?: RobustnessCheck | null;
  latency_ms?: Record<string, number>;
  disclaimer: string;
}
