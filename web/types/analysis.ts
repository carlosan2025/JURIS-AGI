// Analysis and Decision Types

export type DecisionOutcome = "invest" | "pass" | "defer";

export interface CriticalClaim {
  claim_index: number;
  claim_type: string;
  field: string;
  value: string | number | boolean;
  confidence: number;
  polarity: "supportive" | "risk" | "neutral";
  criticality_score: number;
  flip_description?: string;
  sensitivity: Record<string, number>;
}

export interface Robustness {
  overall_score: number;
  stability_margin: number;
  num_critical_claims: number;
  most_fragile_claim_index?: number;
  counterfactuals_tested: number;
  flips_found: number;
  robustness_by_claim_type: Record<string, number>;
  // Uncertainty decomposition
  epistemic_uncertainty: number;
  aleatoric_uncertainty: number;
}

export interface CounterfactualExplanation {
  original_decision: DecisionOutcome;
  flipped_decision: DecisionOutcome;
  explanation: string;
  key_changes: string[];
  confidence: number;
  perturbation_summary: string;
  perturbation_magnitude: number;
}

export interface DecisionFlip {
  perturbation_summary: string;
  total_magnitude: number;
  original_decision: DecisionOutcome;
  new_decision: DecisionOutcome;
}

export interface TraceEntry {
  timestamp: string;
  type: string;
  message: string;
  details?: Record<string, unknown>;
}

export interface AnalysisResult {
  company_id: string;
  decision: DecisionOutcome;
  confidence: number;
  created_at: string;
  summary: {
    company_id: string;
    decision: DecisionOutcome;
    confidence: number;
    num_critical_claims: number;
    robustness_score?: number;
    stability_margin?: number;
  };
  critical_claims: CriticalClaim[];
  robustness: Robustness;
  counterfactual_explanations: CounterfactualExplanation[];
  decision_flips: DecisionFlip[];
  flip_summary: string[];
  trace_entries: TraceEntry[];
}

// Job status for API polling
export type JobStatus = "pending" | "running" | "completed" | "failed";

export interface JobResult {
  job_id: string;
  status: JobStatus;
  task_id?: string;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  runtime_seconds?: number;
  success?: boolean;
  error_message?: string;
  analysis_result?: AnalysisResult;
}
