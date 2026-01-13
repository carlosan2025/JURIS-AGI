import type { EvidenceGraph } from "@/types/evidence";
import type { AnalysisResult, JobResult, JobStatus } from "@/types/analysis";
import type {
  ExtractionRequest,
  ExtractionResult,
  ClaimReviewRequest,
  MergeClaimsRequest,
  ProposedClaim,
  DocumentType,
} from "@/types/extraction";
import type { ReportFormat } from "@/types/report";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export class APIError extends Error {
  constructor(
    message: string,
    public status: number,
    public details?: unknown
  ) {
    super(message);
    this.name = "APIError";
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new APIError(
      error.message || `Request failed with status ${response.status}`,
      response.status,
      error
    );
  }
  return response.json();
}

// Submit evidence graph for VC analysis
export async function submitAnalysis(
  evidenceGraph: EvidenceGraph
): Promise<{ job_id: string; status: JobStatus }> {
  const response = await fetch(`${API_BASE}/vc/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(evidenceGraph),
  });
  return handleResponse(response);
}

// Get job status and result
export async function getJobResult(jobId: string): Promise<JobResult> {
  const response = await fetch(`${API_BASE}/jobs/${jobId}`);
  return handleResponse(response);
}

// Poll for job completion
export async function pollJobResult(
  jobId: string,
  options: {
    maxAttempts?: number;
    intervalMs?: number;
    onProgress?: (status: JobStatus) => void;
  } = {}
): Promise<JobResult> {
  const { maxAttempts = 60, intervalMs = 1000, onProgress } = options;

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    const result = await getJobResult(jobId);
    onProgress?.(result.status);

    if (result.status === "completed" || result.status === "failed") {
      return result;
    }

    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }

  throw new APIError("Job polling timed out", 408);
}

// Health check
export async function healthCheck(): Promise<{
  status: string;
  version: string;
}> {
  const response = await fetch(`${API_BASE}/health`);
  return handleResponse(response);
}

// =============================================================================
// Document Extraction APIs
// =============================================================================

// Extract claims from a document
export async function extractClaims(
  request: ExtractionRequest
): Promise<ExtractionResult> {
  const response = await fetch(`${API_BASE}/extract`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  return handleResponse(response);
}

// Get extraction result by document ID
export async function getExtractionResult(
  docId: string
): Promise<ExtractionResult> {
  const response = await fetch(`${API_BASE}/extract/${docId}`);
  return handleResponse(response);
}

// Review a proposed claim
export async function reviewClaim(
  request: ClaimReviewRequest
): Promise<{ status: string; proposal: ProposedClaim }> {
  const response = await fetch(`${API_BASE}/extract/review`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  return handleResponse(response);
}

// Merge approved claims into evidence graph
export async function mergeClaims(
  request: MergeClaimsRequest
): Promise<{
  company_id: string;
  merged_count: number;
  merged_claims: unknown[];
  errors: string[];
}> {
  const response = await fetch(`${API_BASE}/extract/merge`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  return handleResponse(response);
}

// Get supported document types
export async function getSupportedDocumentTypes(): Promise<{
  types: string[];
  aliases: Record<string, string[]>;
}> {
  const response = await fetch(`${API_BASE}/extract/supported-types`);
  return handleResponse(response);
}

// =============================================================================
// Decision Report APIs
// =============================================================================

// Get decision report for a job
export async function getDecisionReport(
  jobId: string,
  format: ReportFormat = "html"
): Promise<string> {
  const response = await fetch(
    `${API_BASE}/vc/jobs/${jobId}/report?format=${format}`
  );
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new APIError(
      error.message || `Failed to get report`,
      response.status,
      error
    );
  }
  return response.text();
}

// Download report as file
export async function downloadReport(
  jobId: string,
  format: ReportFormat = "pdf",
  filename?: string
): Promise<void> {
  const response = await fetch(
    `${API_BASE}/vc/jobs/${jobId}/report?format=${format}`
  );
  if (!response.ok) {
    throw new APIError("Failed to download report", response.status);
  }

  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename || `decision-report-${jobId}.${format}`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// Generate report directly (for demo mode)
export async function generateReportDirect(
  evidenceGraph: Record<string, unknown>,
  trace: Record<string, unknown>,
  decision: string,
  format: ReportFormat = "html"
): Promise<string> {
  const response = await fetch(
    `${API_BASE}/vc/report/generate?format=${format}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ evidence_graph: evidenceGraph, trace, decision }),
    }
  );
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new APIError(
      error.message || `Failed to generate report`,
      response.status,
      error
    );
  }
  return response.text();
}

// Simulate report generation for demo mode (no backend required)
export function simulateReportHTML(
  evidenceGraph: EvidenceGraph,
  analysisResult: AnalysisResult
): string {
  const company = evidenceGraph.company_id;
  const decision = analysisResult.decision.toUpperCase();
  const confidence = (analysisResult.confidence * 100).toFixed(0);
  const robustness = (analysisResult.robustness.overall_score * 100).toFixed(0);

  const supportive = evidenceGraph.claims.filter(
    (c) => c.polarity === "supportive"
  );
  const risks = evidenceGraph.claims.filter((c) => c.polarity === "risk");

  const strengthsList = supportive
    .slice(0, 5)
    .map((c) => `<li>${c.claim_type}: ${c.field} (${c.value})</li>`)
    .join("");
  const risksList = risks
    .slice(0, 5)
    .map((c) => `<li>${c.claim_type}: ${c.field} (${c.value})</li>`)
    .join("");

  const criticalClaimsList = analysisResult.critical_claims
    .slice(0, 5)
    .map(
      (c) =>
        `<li><strong>${c.claim_type}.${c.field}</strong>: Criticality ${(c.criticality_score * 100).toFixed(0)}%</li>`
    )
    .join("");

  const flipsList = analysisResult.counterfactual_explanations
    .slice(0, 3)
    .map((cf) => `<li>${cf.explanation}</li>`)
    .join("");

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Investment Decision Report: ${company}</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #1a1a1a; margin: 0; padding: 20px; background: #f5f5f5; }
    .container { max-width: 800px; margin: 0 auto; background: white; padding: 40px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }
    h1 { color: #1e3a5f; border-bottom: 2px solid #2d5a87; padding-bottom: 10px; }
    h2 { color: #1e3a5f; margin-top: 30px; }
    .decision-box { padding: 20px; border-radius: 8px; margin: 20px 0; text-align: center; }
    .decision-invest { background: #e8f5e9; border: 2px solid #4caf50; }
    .decision-pass { background: #ffebee; border: 2px solid #f44336; }
    .decision-defer { background: #fff8e1; border: 2px solid #ff9800; }
    .decision-badge { font-size: 24px; font-weight: bold; }
    .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin: 20px 0; }
    .stat-box { text-align: center; padding: 15px; background: #f8f9fa; border-radius: 8px; }
    .stat-value { font-size: 24px; font-weight: bold; color: #1e3a5f; }
    .stat-label { font-size: 12px; color: #666; }
    .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    .strengths { background: #e8f5e9; padding: 15px; border-radius: 8px; border-left: 4px solid #4caf50; }
    .risks { background: #ffebee; padding: 15px; border-radius: 8px; border-left: 4px solid #f44336; }
    ul { margin: 10px 0; padding-left: 20px; }
    li { margin: 5px 0; }
    .disclaimer { background: #fff8e1; padding: 15px; border-radius: 8px; margin-top: 30px; font-size: 13px; }
    .footer { text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid #e0e0e0; font-size: 12px; color: #666; }
  </style>
</head>
<body>
  <div class="container">
    <h1>Investment Decision Report: ${company}</h1>
    <p style="color: #666;">Generated: ${new Date().toISOString()}</p>

    <div class="decision-box decision-${analysisResult.decision}">
      <div class="decision-badge">${decision}</div>
      <div style="margin-top: 10px;">Confidence: ${confidence}%</div>
    </div>

    <div class="stats">
      <div class="stat-box">
        <div class="stat-value">${robustness}%</div>
        <div class="stat-label">Robustness Score</div>
      </div>
      <div class="stat-box">
        <div class="stat-value">${evidenceGraph.claims.length}</div>
        <div class="stat-label">Total Claims</div>
      </div>
      <div class="stat-box">
        <div class="stat-value">${analysisResult.critical_claims.length}</div>
        <div class="stat-label">Critical Claims</div>
      </div>
    </div>

    <h2>Executive Summary</h2>
    <div class="two-col">
      <div class="strengths">
        <h4 style="margin-top: 0;">Key Strengths</h4>
        <ul>${strengthsList || "<li>No supportive claims</li>"}</ul>
      </div>
      <div class="risks">
        <h4 style="margin-top: 0;">Key Risks</h4>
        <ul>${risksList || "<li>No risk claims</li>"}</ul>
      </div>
    </div>

    <h2>Critical Claims</h2>
    <p>Claims that most significantly impact the decision:</p>
    <ul>${criticalClaimsList || "<li>No critical claims identified</li>"}</ul>

    <h2>Counterfactual Analysis</h2>
    <p>Scenarios where the decision would flip:</p>
    <ul>${flipsList || "<li>No decision flip scenarios found</li>"}</ul>

    <div class="disclaimer">
      <strong>DISCLAIMER:</strong> This report is decision support only, not investment advice.
      All investment decisions remain the sole responsibility of the investment committee.
      Confidence scores reflect model uncertainty, not probability of investment success.
    </div>

    <div class="footer">
      JURIS-AGI VC Decision Intelligence | Demo Report
    </div>
  </div>
</body>
</html>`;
}

// =============================================================================
// Document Extraction APIs (continued)
// =============================================================================

// Helper function to extract patterns from content
function extractFromContent(content: string, patterns: { regex: RegExp; field: string; claimType: string; polarity: "supportive" | "risk" | "neutral"; confidenceBase: number }[]): ProposedClaim[] {
  const claims: ProposedClaim[] = [];
  const contentLower = content.toLowerCase();

  for (const pattern of patterns) {
    const match = content.match(pattern.regex);
    if (match) {
      // Find the line containing this match for the quote
      const lines = content.split("\n");
      let quoteLine = "";
      let locator = "Document";
      for (let i = 0; i < lines.length; i++) {
        if (lines[i].match(pattern.regex)) {
          quoteLine = lines[i].trim();
          locator = `Line ${i + 1}`;
          break;
        }
      }

      claims.push({
        proposal_id: `prop_${Math.random().toString(36).slice(2, 9)}`,
        claim_type: pattern.claimType,
        field: pattern.field,
        value: match[1] || match[0],
        confidence: pattern.confidenceBase + Math.random() * 0.1,
        polarity: pattern.polarity,
        locator,
        quote: quoteLine.slice(0, 150),
        rationale: `Extracted via pattern matching from document content`,
        status: "pending",
      });
    }
  }

  return claims;
}

// Simulate extraction for demo mode (no backend required)
// This now analyzes actual document content to generate relevant claims
export async function simulateExtraction(
  docId: string,
  docType: DocumentType,
  content: string
): Promise<ExtractionResult> {
  // Simulate processing delay
  await new Promise((resolve) => setTimeout(resolve, 1500));

  const mockClaims: ProposedClaim[] = [];
  const contentLower = content.toLowerCase();

  // Common patterns to extract from any document
  const commonPatterns = [
    // Company name
    { regex: /^([A-Z][A-Za-z0-9\s]+(?:Inc\.|LLC|Corp\.|Co\.|AI|Labs)?)\s*$/m, field: "legal_name", claimType: "company_identity", polarity: "neutral" as const, confidenceBase: 0.85 },
    // ARR/Revenue
    { regex: /\$?([\d,]+)K?\s*ARR/i, field: "arr", claimType: "traction", polarity: "supportive" as const, confidenceBase: 0.82 },
    { regex: /ARR[:\s]*\$?([\d,.]+)([KMB])?/i, field: "arr", claimType: "traction", polarity: "supportive" as const, confidenceBase: 0.82 },
    // MRR
    { regex: /\$?([\d,]+)K?\s*MRR/i, field: "mrr", claimType: "traction", polarity: "supportive" as const, confidenceBase: 0.80 },
    // Valuation
    { regex: /\$?([\d,.]+)([MB])?\s*pre-money/i, field: "pre_money_valuation", claimType: "round_terms", polarity: "neutral" as const, confidenceBase: 0.88 },
    // Raise amount
    { regex: /rais(?:e|ing)\s*\$?([\d,.]+)([MB])?/i, field: "raise_amount", claimType: "round_terms", polarity: "neutral" as const, confidenceBase: 0.85 },
    // TAM
    { regex: /\$?([\d,.]+)([TB])?\s*TAM/i, field: "tam", claimType: "market_scope", polarity: "supportive" as const, confidenceBase: 0.60 },
    // Gross Margin
    { regex: /gross\s*margin[:\s]*([\d.]+)%?/i, field: "gross_margin", claimType: "business_model", polarity: "supportive" as const, confidenceBase: 0.82 },
    // Team size
    { regex: /team\s*(?:size)?[:\s]*([\d]+)\s*employees?/i, field: "team_size", claimType: "team_composition", polarity: "neutral" as const, confidenceBase: 0.85 },
    // Monthly burn
    { regex: /monthly\s*burn[:\s]*\$?([\d,]+)K?/i, field: "monthly_burn", claimType: "capital_intensity", polarity: "neutral" as const, confidenceBase: 0.80 },
    // Runway
    { regex: /runway[:\s]*([\d]+)\s*months/i, field: "runway_months", claimType: "capital_intensity", polarity: "neutral" as const, confidenceBase: 0.78 },
    // NPS
    { regex: /NPS[:\s]*([\d]+)/i, field: "nps", claimType: "traction", polarity: "supportive" as const, confidenceBase: 0.75 },
    // DAU/Users
    { regex: /([\d,]+)K?\s*(?:daily\s*active\s*users|DAU)/i, field: "dau", claimType: "traction", polarity: "supportive" as const, confidenceBase: 0.78 },
  ];

  // Extract common patterns
  const extractedClaims = extractFromContent(content, commonPatterns);
  mockClaims.push(...extractedClaims);

  // Add document-type specific claims based on content analysis
  if (docType === "pitch_deck") {
    // Look for team backgrounds
    if (contentLower.includes("ceo") || contentLower.includes("founder")) {
      const founderMatch = content.match(/(?:CEO|Founder)[:\s-]*([^,\n]+(?:PhD|years|ex-\w+|former)[^,\n]*)/i);
      if (founderMatch) {
        mockClaims.push({
          proposal_id: `prop_${Math.random().toString(36).slice(2, 9)}`,
          claim_type: "team_quality",
          field: "founder_background",
          value: founderMatch[1].trim(),
          confidence: 0.85,
          polarity: "supportive",
          locator: "Team Section",
          quote: founderMatch[0].slice(0, 100),
          rationale: "Founder background extracted from team description",
          status: "pending",
        });
      }
    }

    // Look for competitive landscape
    if (contentLower.includes("competitor") || contentLower.includes("competitive")) {
      mockClaims.push({
        proposal_id: `prop_${Math.random().toString(36).slice(2, 9)}`,
        claim_type: "differentiation",
        field: "competitive_landscape",
        value: "Competitors identified in document",
        confidence: 0.70,
        polarity: "neutral",
        locator: "Competitive Section",
        quote: "See competitive landscape analysis",
        rationale: "Document contains competitive analysis section",
        status: "pending",
      });
    }
  }

  if (docType === "financial_model") {
    // Look for growth metrics
    const growthMatch = content.match(/([\d]+)%\s*(?:MoM|YoY|growth)/i);
    if (growthMatch) {
      mockClaims.push({
        proposal_id: `prop_${Math.random().toString(36).slice(2, 9)}`,
        claim_type: "traction",
        field: "growth_rate",
        value: `${growthMatch[1]}%`,
        confidence: 0.80,
        polarity: "supportive",
        locator: "Growth Metrics",
        quote: growthMatch[0],
        rationale: "Growth rate extracted from financial metrics",
        status: "pending",
      });
    }

    // Look for LTV/CAC
    const ltvCacMatch = content.match(/LTV\/CAC[:\s]*([\d.]+)/i);
    if (ltvCacMatch) {
      const ratio = parseFloat(ltvCacMatch[1]);
      mockClaims.push({
        proposal_id: `prop_${Math.random().toString(36).slice(2, 9)}`,
        claim_type: "business_model",
        field: "ltv_cac_ratio",
        value: ratio,
        confidence: 0.85,
        polarity: ratio > 3 ? "supportive" : ratio > 1 ? "neutral" : "risk",
        locator: "Unit Economics",
        quote: ltvCacMatch[0],
        rationale: `LTV/CAC ratio of ${ratio}x ${ratio > 3 ? "indicates healthy unit economics" : "needs improvement"}`,
        status: "pending",
      });
    }
  }

  if (docType === "tech_description") {
    // Look for production status
    if (contentLower.includes("production") || contentLower.includes("deployed")) {
      mockClaims.push({
        proposal_id: `prop_${Math.random().toString(36).slice(2, 9)}`,
        claim_type: "product_readiness",
        field: "stage",
        value: "production",
        confidence: 0.88,
        polarity: "supportive",
        locator: "Overview Section",
        quote: "Production deployment",
        rationale: "Document indicates product is in production",
        status: "pending",
      });
    }

    // Look for proprietary data mentions
    const dataMatch = content.match(/([\d,.]+[KMB]?)\+?\s*(?:proprietary|training|labeled)\s*(?:examples?|data|samples?)/i);
    if (dataMatch) {
      mockClaims.push({
        proposal_id: `prop_${Math.random().toString(36).slice(2, 9)}`,
        claim_type: "technical_moat",
        field: "proprietary_data",
        value: dataMatch[0],
        confidence: 0.72,
        polarity: "supportive",
        locator: "Technical Section",
        quote: dataMatch[0],
        rationale: "Proprietary data asset identified",
        status: "pending",
      });
    }
  }

  if (docType === "ic_memo") {
    // Look for recommendation
    const recMatch = content.match(/recommend(?:ation)?[:\s]*(\w+)/i);
    if (recMatch) {
      mockClaims.push({
        proposal_id: `prop_${Math.random().toString(36).slice(2, 9)}`,
        claim_type: "exit_logic",
        field: "ic_recommendation",
        value: recMatch[1],
        confidence: 0.95,
        polarity: "neutral",
        locator: "Recommendation Section",
        quote: recMatch[0],
        rationale: "IC recommendation extracted from memo",
        status: "pending",
      });
    }

    // Look for key risks section
    if (contentLower.includes("key risk") || contentLower.includes("risks")) {
      mockClaims.push({
        proposal_id: `prop_${Math.random().toString(36).slice(2, 9)}`,
        claim_type: "execution_risk",
        field: "risk_assessment",
        value: "Key risks identified in IC memo",
        confidence: 0.85,
        polarity: "risk",
        locator: "Risks Section",
        quote: "See key risks analysis",
        rationale: "Document contains formal risk assessment",
        status: "pending",
      });
    }
  }

  // If no claims were extracted, add some generic ones based on doc type
  if (mockClaims.length === 0) {
    mockClaims.push({
      proposal_id: `prop_${Math.random().toString(36).slice(2, 9)}`,
      claim_type: "company_identity",
      field: "document_analyzed",
      value: `${docType} document`,
      confidence: 0.50,
      polarity: "neutral",
      locator: "Full Document",
      quote: content.slice(0, 100) + "...",
      rationale: "Document analyzed but no specific claims extracted. May need manual review.",
      status: "pending",
    });
  }

  return {
    doc_id: docId,
    doc_type: docType,
    proposed_claims: mockClaims,
    extraction_time_seconds: 1.5,
    errors: [],
    success: true,
  };
}

// Fixed seed for reproducible demo results
const DEMO_SEED = 42;

// Seeded random number generator for reproducibility
function seededRandom(seed: number) {
  let state = seed;
  return function () {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    return state / 0x7fffffff;
  };
}

// =============================================================================
// VC Reasoning APIs (New endpoints)
// =============================================================================

// Types for VC Context and Reasoning
export interface VCConstraints {
  max_claims?: number;
  min_confidence?: number;
  per_bucket_cap?: number;
  required_buckets?: string[];
}

export interface ContextClaim {
  claim_id: string;
  claim_type: string;
  field: string;
  value: unknown;
  confidence: number;
  polarity: "supportive" | "risk" | "neutral";
  bucket: string;
  dropped?: boolean;
  drop_reason?: string;
}

export interface ContextConflict {
  claim_ids: string[];
  conflict_type: string;
  description: string;
  resolution?: string;
}

export interface ContextPreview {
  total_claims: number;
  selected_claims: number;
  dropped_claims: number;
  claims_by_bucket: Record<string, ContextClaim[]>;
  dropped_by_reason: Record<string, number>;
  conflicts: ContextConflict[];
  constraints_applied: VCConstraints;
}

export interface VCJobEvent {
  event_type: string;
  timestamp: string;
  message: string;
  details?: Record<string, unknown>;
}

export interface PolicyRule {
  predicate: string;
  weight: number;
  claim_ids: string[];
}

export interface PolicyVariant {
  policy_id: string;
  decision: "invest" | "pass" | "defer";
  confidence: number;
  mdl_score: number;
  rules: PolicyRule[];
  coverage: {
    total: number;
    covered: number;
    correct: number;
  };
}

export interface UncertaintyInfo {
  level: "low" | "medium" | "high" | "very_high";
  epistemic: number;
  aleatoric: number;
  missing_info: string[];
  suggested_questions: string[];
}

export interface Counterfactual {
  description: string;
  original_decision: string;
  flipped_decision: string;
  key_changes: string[];
  magnitude: number;
}

export interface VCJobResult {
  job_id: string;
  status: "pending" | "running" | "completed" | "failed";
  decision?: {
    primary: "invest" | "pass" | "defer";
    confidence: number;
    explanation: string;
  };
  policies?: PolicyVariant[];
  uncertainty?: UncertaintyInfo;
  counterfactuals?: Counterfactual[];
  context_summary?: {
    claims_used: number;
    buckets: string[];
  };
  error?: string;
}

// Build context preview from evidence graph
export async function buildContextPreview(
  evidenceGraph: EvidenceGraph,
  constraints: VCConstraints = {}
): Promise<ContextPreview> {
  const response = await fetch(`${API_BASE}/vc/context/preview`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ evidence_graph: evidenceGraph, constraints }),
  });
  return handleResponse(response);
}

// Submit VC reasoning job
export async function submitVCJob(
  evidenceGraph: EvidenceGraph,
  constraints?: VCConstraints
): Promise<{ job_id: string; status: string }> {
  const response = await fetch(`${API_BASE}/vc/solve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      evidence_graph: evidenceGraph,
      constraints,
    }),
  });
  return handleResponse(response);
}

// Get VC job status and result
export async function getVCJobResult(jobId: string): Promise<VCJobResult> {
  const response = await fetch(`${API_BASE}/vc/jobs/${jobId}`);
  return handleResponse(response);
}

// Get job events for live timeline
export async function getVCJobEvents(jobId: string): Promise<VCJobEvent[]> {
  const response = await fetch(`${API_BASE}/vc/jobs/${jobId}/events`);
  return handleResponse(response);
}

// Poll for VC job completion with event updates
export async function pollVCJob(
  jobId: string,
  options: {
    maxAttempts?: number;
    intervalMs?: number;
    onEvent?: (event: VCJobEvent) => void;
    onProgress?: (status: string) => void;
  } = {}
): Promise<VCJobResult> {
  const { maxAttempts = 120, intervalMs = 1000, onEvent, onProgress } = options;
  let lastEventCount = 0;

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    const result = await getVCJobResult(jobId);
    onProgress?.(result.status);

    // Fetch and emit new events
    if (onEvent) {
      try {
        const events = await getVCJobEvents(jobId);
        for (let i = lastEventCount; i < events.length; i++) {
          onEvent(events[i]);
        }
        lastEventCount = events.length;
      } catch {
        // Ignore event fetch errors
      }
    }

    if (result.status === "completed" || result.status === "failed") {
      return result;
    }

    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }

  throw new APIError("VC job polling timed out", 408);
}

// =============================================================================
// Demo/Simulation Functions
// =============================================================================

// Simulate context preview for demo mode
export function simulateContextPreview(
  evidenceGraph: EvidenceGraph,
  constraints: VCConstraints = {}
): ContextPreview {
  const maxClaims = constraints.max_claims || 50;
  const minConfidence = constraints.min_confidence || 0.5;
  const perBucketCap = constraints.per_bucket_cap || 10;

  // Group claims by ontology bucket
  const bucketMap: Record<string, ContextClaim[]> = {};
  const droppedReasons: Record<string, number> = {};
  const conflicts: ContextConflict[] = [];

  let selectedCount = 0;
  let droppedCount = 0;

  // Process each claim
  evidenceGraph.claims.forEach((claim, index) => {
    const bucket = claim.claim_type || "uncategorized";
    if (!bucketMap[bucket]) {
      bucketMap[bucket] = [];
    }

    const contextClaim: ContextClaim = {
      claim_id: `claim_${index}`,
      claim_type: claim.claim_type,
      field: claim.field,
      value: claim.value,
      confidence: claim.confidence,
      polarity: claim.polarity,
      bucket,
    };

    // Check if claim should be dropped
    if (claim.confidence < minConfidence) {
      contextClaim.dropped = true;
      contextClaim.drop_reason = "low_confidence";
      droppedReasons["low_confidence"] = (droppedReasons["low_confidence"] || 0) + 1;
      droppedCount++;
    } else if (bucketMap[bucket].filter(c => !c.dropped).length >= perBucketCap) {
      contextClaim.dropped = true;
      contextClaim.drop_reason = "bucket_cap_exceeded";
      droppedReasons["bucket_cap_exceeded"] = (droppedReasons["bucket_cap_exceeded"] || 0) + 1;
      droppedCount++;
    } else if (selectedCount >= maxClaims) {
      contextClaim.dropped = true;
      contextClaim.drop_reason = "max_claims_exceeded";
      droppedReasons["max_claims_exceeded"] = (droppedReasons["max_claims_exceeded"] || 0) + 1;
      droppedCount++;
    } else {
      selectedCount++;
    }

    bucketMap[bucket].push(contextClaim);
  });

  // Detect conflicts (claims with same field but different values)
  const fieldValues: Record<string, ContextClaim[]> = {};
  Object.values(bucketMap).flat().filter(c => !c.dropped).forEach(claim => {
    const key = `${claim.claim_type}.${claim.field}`;
    if (!fieldValues[key]) {
      fieldValues[key] = [];
    }
    fieldValues[key].push(claim);
  });

  Object.entries(fieldValues).forEach(([key, claims]) => {
    if (claims.length > 1) {
      const values = new Set(claims.map(c => JSON.stringify(c.value)));
      if (values.size > 1) {
        conflicts.push({
          claim_ids: claims.map(c => c.claim_id),
          conflict_type: "value_mismatch",
          description: `Multiple values found for ${key}`,
          resolution: "Using highest confidence value",
        });
      }
    }
  });

  return {
    total_claims: evidenceGraph.claims.length,
    selected_claims: selectedCount,
    dropped_claims: droppedCount,
    claims_by_bucket: bucketMap,
    dropped_by_reason: droppedReasons,
    conflicts,
    constraints_applied: {
      max_claims: maxClaims,
      min_confidence: minConfidence,
      per_bucket_cap: perBucketCap,
      ...constraints,
    },
  };
}

// Simulate VC job events for demo mode
export function* simulateVCJobEvents(): Generator<VCJobEvent> {
  const events: VCJobEvent[] = [
    {
      event_type: "job_started",
      timestamp: new Date().toISOString(),
      message: "VC reasoning job started",
      details: { phase: "initialization" },
    },
    {
      event_type: "context_built",
      timestamp: new Date(Date.now() + 500).toISOString(),
      message: "Context built from evidence graph",
      details: { claims_selected: 45, buckets: 8 },
    },
    {
      event_type: "working_set_built",
      timestamp: new Date(Date.now() + 1000).toISOString(),
      message: "Working set constructed",
      details: { predicates: 120, examples: 50 },
    },
    {
      event_type: "thresholds_proposed",
      timestamp: new Date(Date.now() + 2000).toISOString(),
      message: "Optimal thresholds identified",
      details: { thresholds_found: 15 },
    },
    {
      event_type: "policies_learning",
      timestamp: new Date(Date.now() + 3000).toISOString(),
      message: "Learning policy hypotheses...",
      details: { iteration: 1 },
    },
    {
      event_type: "policies_learned",
      timestamp: new Date(Date.now() + 5000).toISOString(),
      message: "Generated 3 policy variants",
      details: { num_policies: 3, best_mdl: 45.2 },
    },
    {
      event_type: "evaluation_complete",
      timestamp: new Date(Date.now() + 6000).toISOString(),
      message: "Policy evaluation completed",
      details: { coverage: 0.92, accuracy: 0.88 },
    },
    {
      event_type: "uncertainty_analyzed",
      timestamp: new Date(Date.now() + 7000).toISOString(),
      message: "Uncertainty analysis completed",
      details: { level: "medium", epistemic: 0.15 },
    },
    {
      event_type: "job_completed",
      timestamp: new Date(Date.now() + 8000).toISOString(),
      message: "VC reasoning completed successfully",
      details: { decision: "invest", confidence: 0.82 },
    },
  ];

  for (const event of events) {
    yield event;
  }
}

// Simulate full VC job result for demo mode
export async function simulateVCJob(
  evidenceGraph: EvidenceGraph,
  constraints?: VCConstraints,
  onEvent?: (event: VCJobEvent) => void
): Promise<VCJobResult> {
  // Emit events with delays
  const events = Array.from(simulateVCJobEvents());
  for (const event of events) {
    onEvent?.(event);
    await new Promise(resolve => setTimeout(resolve, 800));
  }

  // Calculate decision based on evidence
  const supportive = evidenceGraph.claims.filter(c => c.polarity === "supportive").length;
  const risks = evidenceGraph.claims.filter(c => c.polarity === "risk").length;
  const total = evidenceGraph.claims.length;

  const supportRatio = supportive / Math.max(total, 1);
  const riskRatio = risks / Math.max(total, 1);

  let decision: "invest" | "pass" | "defer";
  let confidence: number;

  if (supportRatio > 0.5 && riskRatio < 0.3) {
    decision = "invest";
    confidence = 0.75 + supportRatio * 0.15;
  } else if (riskRatio > 0.4) {
    decision = "pass";
    confidence = 0.65 + riskRatio * 0.2;
  } else {
    decision = "defer";
    confidence = 0.55;
  }

  // Generate policy variants
  const policies: PolicyVariant[] = [
    {
      policy_id: "policy_1",
      decision,
      confidence,
      mdl_score: 42.5,
      rules: [
        { predicate: "traction.arr > 500K", weight: 0.3, claim_ids: ["claim_1", "claim_2"] },
        { predicate: "team_quality.founder_background contains 'experienced'", weight: 0.25, claim_ids: ["claim_3"] },
        { predicate: "market_scope.tam > 1B", weight: 0.2, claim_ids: ["claim_4"] },
      ],
      coverage: { total: 50, covered: 46, correct: 42 },
    },
    {
      policy_id: "policy_2",
      decision: decision === "invest" ? "invest" : "defer",
      confidence: confidence - 0.1,
      mdl_score: 48.2,
      rules: [
        { predicate: "business_model.gross_margin > 60%", weight: 0.35, claim_ids: ["claim_5"] },
        { predicate: "traction.growth_rate > 20%", weight: 0.3, claim_ids: ["claim_6", "claim_7"] },
      ],
      coverage: { total: 50, covered: 44, correct: 40 },
    },
    {
      policy_id: "policy_3",
      decision: decision === "pass" ? "pass" : "defer",
      confidence: confidence - 0.15,
      mdl_score: 52.1,
      rules: [
        { predicate: "capital_intensity.runway_months > 18", weight: 0.4, claim_ids: ["claim_8"] },
        { predicate: "differentiation.competitive_moat exists", weight: 0.25, claim_ids: ["claim_9"] },
      ],
      coverage: { total: 50, covered: 42, correct: 38 },
    },
  ];

  // Generate uncertainty info
  const uncertainty: UncertaintyInfo = {
    level: riskRatio > 0.3 ? "high" : supportRatio > 0.5 ? "low" : "medium",
    epistemic: 0.12 + Math.random() * 0.1,
    aleatoric: 0.08 + Math.random() * 0.08,
    missing_info: [
      "Customer retention metrics not provided",
      "Competitive landscape analysis incomplete",
      "Unit economics validation pending",
    ],
    suggested_questions: [
      "What is the customer churn rate?",
      "How does the product compare to top 3 competitors?",
      "What are the validated CAC and LTV figures?",
    ],
  };

  // Generate counterfactuals
  const counterfactuals: Counterfactual[] = [
    {
      description: `Decision would flip to ${decision === "invest" ? "pass" : "invest"} if ARR dropped below $300K`,
      original_decision: decision,
      flipped_decision: decision === "invest" ? "pass" : "invest",
      key_changes: ["traction.arr"],
      magnitude: 0.4,
    },
    {
      description: `Decision would become "defer" if team size reduced by 50%`,
      original_decision: decision,
      flipped_decision: "defer",
      key_changes: ["team_composition.team_size"],
      magnitude: 0.5,
    },
  ];

  return {
    job_id: `demo_job_${Date.now()}`,
    status: "completed",
    decision: {
      primary: decision,
      confidence,
      explanation: `Based on ${total} claims analyzed, the model recommends ${decision.toUpperCase()} with ${(confidence * 100).toFixed(0)}% confidence.`,
    },
    policies,
    uncertainty,
    counterfactuals,
    context_summary: {
      claims_used: Math.min(total, constraints?.max_claims || 50),
      buckets: Array.from(new Set(evidenceGraph.claims.map(c => c.claim_type))),
    },
  };
}

// For local development/demo: simulate analysis
export async function simulateAnalysis(
  evidenceGraph: EvidenceGraph
): Promise<AnalysisResult> {
  // Simulate processing delay
  await new Promise((resolve) => setTimeout(resolve, 2000));

  // Use seeded RNG for reproducible results
  const rng = seededRandom(DEMO_SEED);

  const supportiveCount = evidenceGraph.claims.filter(
    (c) => c.polarity === "supportive"
  ).length;
  const riskCount = evidenceGraph.claims.filter(
    (c) => c.polarity === "risk"
  ).length;
  const totalClaims = evidenceGraph.claims.length;

  const riskRatio = riskCount / Math.max(totalClaims, 1);
  const supportRatio = supportiveCount / Math.max(totalClaims, 1);

  let decision: "invest" | "pass" | "defer";
  let confidence: number;

  if (supportRatio > 0.5 && riskRatio < 0.3) {
    decision = "invest";
    confidence = 0.75 + supportRatio * 0.2;
  } else if (riskRatio > 0.4) {
    decision = "pass";
    confidence = 0.6 + riskRatio * 0.3;
  } else {
    decision = "defer";
    confidence = 0.5;
  }

  // Add small deterministic noise based on seed for realism
  // while maintaining reproducibility
  confidence = Math.min(0.95, Math.max(0.35, confidence + (rng() - 0.5) * 0.1));

  // Generate mock critical claims (using seeded RNG for reproducibility)
  const criticalClaims = evidenceGraph.claims
    .filter((c) => c.polarity !== "neutral")
    .slice(0, 3)
    .map((c, i) => ({
      claim_index: i,
      claim_type: c.claim_type,
      field: c.field,
      value: c.value,
      confidence: c.confidence,
      polarity: c.polarity,
      criticality_score: 0.5 + rng() * 0.4,
      flip_description: `Changing ${c.field} could affect the decision`,
      sensitivity: {
        value_small: rng() > 0.5 ? 1 : 0,
        polarity: rng() > 0.3 ? 1 : 0,
      },
    }));

  // Generate mock counterfactual explanations (deterministic)
  const counterfactualExplanations = criticalClaims.slice(0, 2).map((cc) => ({
    original_decision: decision,
    flipped_decision: decision === "invest" ? "pass" : "invest" as "invest" | "pass" | "defer",
    explanation: `Decision flips from ${decision} to ${decision === "invest" ? "pass" : "invest"} if ${cc.claim_type}.${cc.field} changes significantly`,
    key_changes: [`${cc.claim_type}.${cc.field}`],
    confidence: cc.criticality_score,
    perturbation_summary: `${cc.field} modified`,
    perturbation_magnitude: 0.3 + rng() * 0.4,
  }));

  return {
    company_id: evidenceGraph.company_id,
    decision,
    confidence,
    created_at: new Date().toISOString(),
    summary: {
      company_id: evidenceGraph.company_id,
      decision,
      confidence,
      num_critical_claims: criticalClaims.length,
      robustness_score: 0.6 + Math.random() * 0.3,
      stability_margin: 0.4 + Math.random() * 0.4,
    },
    critical_claims: criticalClaims,
    robustness: {
      overall_score: 0.6 + rng() * 0.3,
      stability_margin: 0.4 + rng() * 0.4,
      num_critical_claims: criticalClaims.length,
      most_fragile_claim_index: 0,
      counterfactuals_tested: 20,
      flips_found: Math.floor(rng() * 5),
      robustness_by_claim_type: {},
      epistemic_uncertainty: 0.15 + rng() * 0.2,
      aleatoric_uncertainty: 0.1 + rng() * 0.15,
    },
    counterfactual_explanations: counterfactualExplanations,
    decision_flips: counterfactualExplanations.map((e) => ({
      perturbation_summary: e.perturbation_summary,
      total_magnitude: e.perturbation_magnitude,
      original_decision: e.original_decision,
      new_decision: e.flipped_decision,
    })),
    flip_summary: counterfactualExplanations.map((e) => e.explanation),
    trace_entries: [
      {
        timestamp: new Date().toISOString(),
        type: "analysis_start",
        message: `Started analysis for ${evidenceGraph.company_id}`,
        details: { decision, confidence },
      },
      ...criticalClaims.map((cc) => ({
        timestamp: new Date().toISOString(),
        type: "critical_claim",
        message: `Identified critical claim: ${cc.claim_type}.${cc.field}`,
        details: cc,
      })),
      {
        timestamp: new Date().toISOString(),
        type: "robustness_check",
        message: "Robustness analysis completed",
        details: { overall_score: 0.7, counterfactuals_tested: 20 },
      },
      {
        timestamp: new Date().toISOString(),
        type: "analysis_complete",
        message: `Decision: ${decision.toUpperCase()} with ${(confidence * 100).toFixed(0)}% confidence`,
        details: { decision, confidence },
      },
    ],
  };
}
