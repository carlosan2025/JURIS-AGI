"use client";

import React from "react";

/**
 * Standard disclaimer for JURIS-AGI decision support outputs.
 * Must be displayed with all analysis results.
 */
export function Disclaimer({ compact = false }: { compact?: boolean }) {
  if (compact) {
    return (
      <div className="text-xs text-muted-foreground border-t pt-2 mt-4">
        <strong>Disclaimer:</strong> Decision support only, not investment advice.
        Outputs augment human judgment; they do not replace it.
      </div>
    );
  }

  return (
    <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 text-sm">
      <h4 className="font-semibold text-amber-800 mb-2">Important Disclaimer</h4>
      <ul className="text-amber-700 space-y-1 list-disc list-inside">
        <li>
          JURIS-AGI is a <strong>decision support tool</strong>, not a decision maker.
        </li>
        <li>
          All outputs are intended to augment human judgment, not replace it.
        </li>
        <li>
          Investment decisions remain the sole responsibility of the investment committee.
        </li>
        <li>
          Confidence scores reflect model uncertainty, not probability of investment success.
        </li>
        <li>
          This tool does not constitute financial or investment advice.
        </li>
      </ul>
    </div>
  );
}

/**
 * Warning displayed when analysis confidence is too low or evidence is insufficient.
 */
export function UnderdeterminedWarning({
  reason,
  missingClaims,
}: {
  reason: string;
  missingClaims?: string[];
}) {
  return (
    <div className="bg-red-50 border border-red-200 rounded-lg p-4">
      <h4 className="font-semibold text-red-800 mb-2 flex items-center gap-2">
        <svg
          className="w-5 h-5"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
          />
        </svg>
        Analysis Refused - Insufficient Evidence
      </h4>
      <p className="text-red-700 mb-2">{reason}</p>
      {missingClaims && missingClaims.length > 0 && (
        <div className="mt-2">
          <p className="text-red-700 font-medium">Missing critical information:</p>
          <ul className="text-red-600 list-disc list-inside mt-1">
            {missingClaims.map((claim, i) => (
              <li key={i}>{claim}</li>
            ))}
          </ul>
        </div>
      )}
      <p className="text-red-600 text-sm mt-3 italic">
        The system will not provide a recommendation when evidence is insufficient.
        Please gather additional information before re-analyzing.
      </p>
    </div>
  );
}

/**
 * Confidence warning when decision confidence is low.
 */
export function LowConfidenceWarning({ confidence }: { confidence: number }) {
  if (confidence >= 0.6) return null;

  return (
    <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3 text-sm">
      <div className="flex items-center gap-2 text-yellow-800">
        <svg
          className="w-4 h-4"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
          />
        </svg>
        <span className="font-medium">Low Confidence Warning</span>
      </div>
      <p className="text-yellow-700 mt-1">
        This analysis has {(confidence * 100).toFixed(0)}% confidence, indicating
        high uncertainty. Additional evidence or manual review is strongly
        recommended before proceeding.
      </p>
    </div>
  );
}

/**
 * Safeguard constants
 */
export const SAFEGUARD_THRESHOLDS = {
  // Minimum confidence to provide a recommendation
  MIN_CONFIDENCE_FOR_RECOMMENDATION: 0.4,

  // Minimum number of claims required for analysis
  MIN_CLAIMS_FOR_ANALYSIS: 5,

  // Required claim types for minimal viable analysis
  REQUIRED_CLAIM_TYPES: ["traction", "team_quality"],

  // Warning threshold for low confidence
  LOW_CONFIDENCE_THRESHOLD: 0.6,

  // High uncertainty threshold
  HIGH_UNCERTAINTY_THRESHOLD: 0.35,
};

/**
 * Check if evidence graph meets minimum requirements for analysis.
 */
export function checkEvidenceRequirements(claims: Array<{ claim_type: string }>) {
  const issues: string[] = [];

  // Check minimum claim count
  if (claims.length < SAFEGUARD_THRESHOLDS.MIN_CLAIMS_FOR_ANALYSIS) {
    issues.push(
      `Insufficient evidence: only ${claims.length} claims provided (minimum ${SAFEGUARD_THRESHOLDS.MIN_CLAIMS_FOR_ANALYSIS} required)`
    );
  }

  // Check for required claim types
  const claimTypes = new Set(claims.map((c) => c.claim_type));
  const missingRequired = SAFEGUARD_THRESHOLDS.REQUIRED_CLAIM_TYPES.filter(
    (t) => !claimTypes.has(t)
  );

  if (missingRequired.length > 0) {
    issues.push(`Missing required claim types: ${missingRequired.join(", ")}`);
  }

  return {
    meetsRequirements: issues.length === 0,
    issues,
    missingClaimTypes: missingRequired,
  };
}

/**
 * Check if analysis result should be refused due to uncertainty.
 */
export function shouldRefuseAnalysis(result: {
  confidence: number;
  robustness?: { overall_score: number; epistemic_uncertainty?: number };
}) {
  // Refuse if confidence is too low
  if (result.confidence < SAFEGUARD_THRESHOLDS.MIN_CONFIDENCE_FOR_RECOMMENDATION) {
    return {
      refuse: true,
      reason: `Confidence (${(result.confidence * 100).toFixed(0)}%) is below minimum threshold for recommendation`,
    };
  }

  // Refuse if epistemic uncertainty is too high
  if (
    result.robustness?.epistemic_uncertainty &&
    result.robustness.epistemic_uncertainty > SAFEGUARD_THRESHOLDS.HIGH_UNCERTAINTY_THRESHOLD
  ) {
    return {
      refuse: true,
      reason: `Epistemic uncertainty (${(result.robustness.epistemic_uncertainty * 100).toFixed(0)}%) is too high - more evidence needed`,
    };
  }

  return { refuse: false, reason: null };
}
