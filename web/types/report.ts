// Decision Report types for JURIS-AGI

export type ReportFormat = "html" | "md" | "pdf";

export interface EvidenceItem {
  claim_type: string;
  field: string;
  value: unknown;
  confidence: number;
  polarity: "supportive" | "risk" | "neutral";
  source_document?: string;
  source_locator?: string;
  quote?: string;
}

export interface RuleEvaluation {
  rule_name: string;
  rule_description: string;
  result: "pass" | "fail" | "partial";
  contributing_evidence: string[];
  weight: number;
  notes?: string;
}

export interface SensitivityItem {
  claim_type: string;
  claim_field: string;
  original_value: unknown;
  perturbed_value: unknown;
  original_decision: "invest" | "pass" | "defer";
  new_decision: "invest" | "pass" | "defer";
  perturbation_magnitude: number;
  criticality_score: number;
  explanation: string;
}

export interface ExecutiveSummary {
  decision: "invest" | "pass" | "defer";
  confidence: number;
  confidence_level: "high" | "medium" | "low";
  one_line_summary: string;
  key_strengths: string[];
  key_risks: string[];
  critical_uncertainties: string[];
  recommendation_text: string;
}

export interface InferredDecisionLogic {
  primary_factors: string[];
  decision_threshold_description: string;
  rule_evaluations: RuleEvaluation[];
  net_signal_score: number;
  decision_rationale: string;
}

export interface EvidenceBasis {
  total_claims: number;
  supportive_claims: EvidenceItem[];
  risk_claims: EvidenceItem[];
  neutral_claims: EvidenceItem[];
  coverage_by_type: Record<string, number>;
  average_confidence: number;
  low_confidence_claims: EvidenceItem[];
}

export interface DealEvaluation {
  company_id: string;
  company_name?: string;
  sector?: string;
  stage?: string;
  traction_summary?: string;
  team_summary?: string;
  market_summary?: string;
  risk_summary?: string;
  traction_score?: number;
  team_score?: number;
  market_score?: number;
  risk_score?: number;
  overall_score?: number;
}

export interface CounterfactualAnalysis {
  robustness_score: number;
  stability_margin: number;
  total_counterfactuals_tested: number;
  flips_found: number;
  critical_claims: SensitivityItem[];
  decision_flip_scenarios: SensitivityItem[];
  robustness_interpretation: string;
}

export interface UncertaintyLimitations {
  epistemic_uncertainty: number;
  aleatoric_uncertainty: number;
  total_uncertainty: number;
  confidence_level: "high" | "medium" | "low";
  data_gaps: string[];
  low_confidence_areas: string[];
  assumptions_made: string[];
  limitations: string[];
  recommendations_for_diligence: string[];
}

export interface AuditMetadata {
  report_id: string;
  generated_at: string;
  model_version: string;
  trace_id?: string;
  analyst_id?: string;
  evidence_graph_version?: string;
  num_claims_analyzed: number;
  num_counterfactuals_tested: number;
  analysis_runtime_seconds?: number;
  reproducibility_seed?: number;
}

export interface DecisionReport {
  executive_summary: ExecutiveSummary;
  decision_logic: InferredDecisionLogic;
  evidence_basis: EvidenceBasis;
  deal_evaluation: DealEvaluation;
  counterfactual_analysis: CounterfactualAnalysis;
  uncertainty_limitations: UncertaintyLimitations;
  audit_metadata: AuditMetadata;
  appendix?: Record<string, unknown>;
}
