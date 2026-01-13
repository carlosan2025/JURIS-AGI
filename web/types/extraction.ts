// Document extraction types

export type DocumentType = "pitch_deck" | "financial_model" | "tech_description" | "ic_memo";

export type ClaimStatus = "pending" | "approved" | "rejected" | "modified" | "merged";

// Claim values can be primitives or arrays
export type ClaimValue = string | number | boolean | string[] | number[];

export interface ProposedClaim {
  proposal_id: string;
  claim_type: string;
  field: string;
  value: ClaimValue;
  confidence: number;
  polarity: "supportive" | "risk" | "neutral";
  locator?: string;
  quote?: string;
  rationale: string;
  status: ClaimStatus;
  reviewer_notes?: string;
  modified_value?: ClaimValue;
  modified_confidence?: number;
  modified_polarity?: "supportive" | "risk" | "neutral";
}

export interface ExtractionResult {
  doc_id: string;
  doc_type: string;
  proposed_claims: ProposedClaim[];
  extraction_time_seconds: number;
  errors: string[];
  success: boolean;
}

export interface ExtractionRequest {
  doc_id: string;
  doc_type: DocumentType;
  content: string;
  company_id?: string;
}

export interface ClaimReviewRequest {
  proposal_id: string;
  action: "approve" | "reject" | "modify";
  modified_value?: ClaimValue;
  modified_confidence?: number;
  modified_polarity?: "supportive" | "risk" | "neutral";
  reviewer_notes?: string;
}

export interface MergeClaimsRequest {
  company_id: string;
  proposal_ids: string[];
}

export const DOCUMENT_TYPE_LABELS: Record<DocumentType, string> = {
  pitch_deck: "Pitch Deck",
  financial_model: "Financial Model",
  tech_description: "Technical Description",
  ic_memo: "IC Memo",
};

export const DOCUMENT_TYPE_DESCRIPTIONS: Record<DocumentType, string> = {
  pitch_deck: "Startup pitch presentations, investor decks",
  financial_model: "Financial projections, P&L, cash flow models",
  tech_description: "Technical architecture, product specs, engineering docs",
  ic_memo: "Investment committee memos, deal memos",
};
