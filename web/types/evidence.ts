// VC Evidence Graph Types

export type ClaimType =
  | "company_identity"
  | "round_terms"
  | "use_of_proceeds"
  | "team_quality"
  | "team_composition"
  | "product_readiness"
  | "technical_moat"
  | "differentiation"
  | "market_scope"
  | "business_model"
  | "traction"
  | "execution_risk"
  | "regulatory_risk"
  | "capital_intensity"
  | "exit_logic";

export type Polarity = "supportive" | "risk" | "neutral";

export interface Source {
  doc_id: string;
  locator?: string;
  quote?: string;
  doc_type?: string;
}

export interface Claim {
  id: string;
  claim_type: ClaimType;
  field: string;
  value: string | number | boolean;
  confidence: number;
  polarity: Polarity;
  source?: Source;
  unit?: string;
  notes?: string;
}

export interface EvidenceGraph {
  company_id: string;
  claims: Claim[];
  analyst_id?: string;
  version?: string;
}

// Ontology metadata
export const CLAIM_TYPE_LABELS: Record<ClaimType, string> = {
  company_identity: "Company Identity",
  round_terms: "Round Terms",
  use_of_proceeds: "Use of Proceeds",
  team_quality: "Team Quality",
  team_composition: "Team Composition",
  product_readiness: "Product Readiness",
  technical_moat: "Technical Moat",
  differentiation: "Differentiation",
  market_scope: "Market Scope",
  business_model: "Business Model",
  traction: "Traction",
  execution_risk: "Execution Risk",
  regulatory_risk: "Regulatory Risk",
  capital_intensity: "Capital Intensity",
  exit_logic: "Exit Logic",
};

export const CLAIM_TYPE_CATEGORIES: Record<string, ClaimType[]> = {
  "Identity & Structure": ["company_identity", "round_terms", "use_of_proceeds"],
  Team: ["team_quality", "team_composition"],
  "Product & Technology": ["product_readiness", "technical_moat", "differentiation"],
  Market: ["market_scope", "business_model", "traction"],
  "Risk Assessment": ["execution_risk", "regulatory_risk", "capital_intensity"],
  Exit: ["exit_logic"],
};

export const POLARITY_COLORS: Record<Polarity, string> = {
  supportive: "text-green-600 bg-green-50 border-green-200",
  risk: "text-red-600 bg-red-50 border-red-200",
  neutral: "text-gray-600 bg-gray-50 border-gray-200",
};
