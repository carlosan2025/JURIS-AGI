"""
Investment Committee (IC) Memo Extractor.

Extracts evidence claims from internal IC memos, deal memos,
and investment recommendations.
"""

from typing import List, Optional

from .base import DocumentExtractor, ExtractionConfig, LLMCallFn
from ..ontology import ClaimType


class ICMemoExtractor(DocumentExtractor):
    """
    Extractor for Investment Committee memos.

    IC memos typically contain:
    - Deal summary and investment thesis
    - Key risks and mitigants
    - Valuation analysis
    - Team assessment
    - Due diligence findings
    - Recommended terms
    """

    DOC_TYPE = "ic_memo"

    # IC memos can contain all claim types
    SUPPORTED_CLAIM_TYPES = [
        ClaimType.COMPANY_IDENTITY,
        ClaimType.ROUND_TERMS,
        ClaimType.USE_OF_PROCEEDS,
        ClaimType.TEAM_QUALITY,
        ClaimType.TEAM_COMPOSITION,
        ClaimType.PRODUCT_READINESS,
        ClaimType.TECHNICAL_MOAT,
        ClaimType.DIFFERENTIATION,
        ClaimType.MARKET_SCOPE,
        ClaimType.BUSINESS_MODEL,
        ClaimType.TRACTION,
        ClaimType.EXECUTION_RISK,
        ClaimType.REGULATORY_RISK,
        ClaimType.CAPITAL_INTENSITY,
        ClaimType.EXIT_LOGIC,
    ]

    def get_system_prompt(self) -> str:
        return """You are an expert VC analyst extracting structured evidence claims from Investment Committee memos.

IC memos represent curated, analyst-verified information. Your task is to extract claims that have been vetted through the DD process.

## Guidelines

1. **IC memos are high-quality sources**:
   - Claims here have typically been verified
   - Assign higher confidence than raw pitch deck claims
   - But note if something is flagged as "unverified" or "to be confirmed"

2. **Capture the analyst's assessment**:
   - Extract both facts AND the analyst's interpretation
   - Note risk mitigants alongside risks
   - Capture conviction level if stated

3. **Extract deal terms precisely**:
   - Valuation, amount, instrument type
   - Pro rata rights, board seats, protective provisions
   - Cap table implications

4. **Capture risk framework**:
   - Key risks identified
   - Mitigants proposed
   - Risk/reward assessment

## Confidence Calibration for IC Memos

IC memos are analyst-curated, so base confidence is higher:
- **0.95**: Verified facts (legal docs, bank statements cited)
- **0.85**: Standard DD findings
- **0.75**: Analyst estimates or assessments
- **0.60**: Areas flagged for further DD
- **0.50**: Risks or concerns (uncertainty is the point)

## Polarity from IC Memo Context

IC memos explicitly frame things as positive or negative:
- "Key strength" / "Why we like this" → supportive
- "Key risk" / "Concerns" / "Red flags" → risk
- "Background" / "Deal terms" → neutral

## Output Format

```json
[
  {
    "claim_type": "round_terms",
    "field": "pre_money_valuation",
    "value": 15000000,
    "confidence": 0.95,
    "polarity": "neutral",
    "locator": "Deal Terms section",
    "quote": "$15M pre-money, $3M raise on SAFE",
    "rationale": "Clear deal terms from IC memo; likely from term sheet"
  },
  {
    "claim_type": "execution_risk",
    "field": "key_person_risk",
    "value": "Sole technical founder; no succession plan",
    "confidence": 0.80,
    "polarity": "risk",
    "locator": "Key Risks section",
    "quote": "Single founder owns all technical knowledge",
    "rationale": "Analyst-identified risk; company aware but no mitigation in place"
  }
]
```

""" + self.get_claim_types_description()

    def get_extraction_prompt(self, document_text: str, doc_id: str) -> str:
        return f"""Extract evidence claims from this Investment Committee memo.

Document ID: {doc_id}

---
IC MEMO CONTENT:
---

{document_text}

---

## IC Memo Extraction Focus

### 1. Deal Overview
- Company basics
- Sector and business model
- Investment thesis summary

### 2. Deal Terms
- Raise amount and valuation
- Instrument type (SAFE, convertible, priced round)
- Key terms and provisions
- Cap table impact

### 3. Investment Thesis
- Why the analyst recommends this deal
- Key value drivers
- Expected return path

### 4. Due Diligence Findings
- Team assessment results
- Technical DD conclusions
- Customer/market validation
- Financial verification

### 5. Risk Assessment
- Key risks identified
- Mitigants proposed
- Unmitigated risks
- Deal-breaker considerations

### 6. Comparison & Context
- Competitive landscape
- Comparable companies
- Market timing

### 7. Exit Analysis
- Potential acquirers
- IPO potential
- Return expectations

Return comprehensive JSON array capturing the analyst's curated findings."""


class ICMemoExtractorV2(ICMemoExtractor):
    """
    Enhanced IC memo extractor with investment thesis focus.

    Captures not just facts but the investment reasoning framework.
    """

    def get_system_prompt(self) -> str:
        return """You are a VC partner reviewing an IC memo to extract key decision factors.

Your task is to extract structured claims that capture both facts AND the investment reasoning.

## IC Memo Analysis Framework

### Investment Thesis Components
1. **Market opportunity**: Size, growth, timing
2. **Team capability**: Can they execute?
3. **Product-market fit**: Evidence of traction
4. **Competitive position**: Why they win
5. **Return potential**: Path to outcome

### Due Diligence Quality Assessment
Note the depth of DD conducted:
- Customer references completed?
- Technical DD done?
- Financial verification?
- Background checks?
- Legal DD?

### Risk Framework
For each risk, capture:
- Risk description
- Likelihood (if stated)
- Impact (if stated)
- Mitigant (if any)
- Residual risk level

### Deal Terms Analysis
Extract and assess:
- Are terms market-standard?
- Any unusual provisions?
- Governance rights?
- Anti-dilution provisions?

## Confidence Scoring for IC Content

IC memos vary in rigor. Adjust confidence based on:
- **Verified**: "Confirmed via bank statement" → 0.95
- **Standard DD**: "Per customer interviews" → 0.85
- **Analyst view**: "We believe..." → 0.70
- **To verify**: "Pending confirmation" → 0.50
- **Red flag**: "Concerning if true" → 0.60 (confidence in the concern)

## Special IC Memo Signals

**Strong conviction indicators:**
- "High conviction"
- "Strong recommend"
- "Best in class"
- "Must-do deal"

**Concern indicators:**
- "Proceed with caution"
- "Conditional on..."
- "Significant risk"
- "Below the bar unless..."

## Output Format

```json
[
  {
    "claim_type": "team_quality",
    "field": "founder_background",
    "value": "Ex-Google Staff Engineer, 12 years ML experience",
    "confidence": 0.90,
    "polarity": "supportive",
    "locator": "Team Assessment section",
    "quote": "CEO was Staff Engineer at Google Brain (2015-2023)",
    "rationale": "Strong technical pedigree; verified via LinkedIn and references"
  },
  {
    "claim_type": "market_scope",
    "field": "tam",
    "value": 8500000000,
    "confidence": 0.70,
    "polarity": "supportive",
    "locator": "Market Analysis section",
    "quote": "$8.5B TAM by 2027 per Gartner",
    "rationale": "Third-party market sizing cited; reasonable methodology"
  },
  {
    "claim_type": "exit_logic",
    "field": "potential_acquirers",
    "value": ["Microsoft", "Salesforce", "ServiceNow"],
    "confidence": 0.65,
    "polarity": "supportive",
    "locator": "Exit Analysis section",
    "quote": "Strategic interest likely from MSFT, CRM, NOW",
    "rationale": "Reasonable acquirer list given product fit; no confirmed interest"
  }
]
```

""" + self.get_claim_types_description()

    def get_extraction_prompt(self, document_text: str, doc_id: str) -> str:
        return f"""Conduct a thorough extraction from this IC memo, capturing both facts and reasoning.

Document ID: {doc_id}

---
IC MEMO CONTENT:
---

{document_text}

---

## Comprehensive Extraction Required

### Section 1: Deal Basics
- Company, sector, stage
- Deal terms in detail
- Fund allocation / ownership target

### Section 2: Investment Thesis
- Primary thesis points
- Key value drivers
- Expected return path

### Section 3: DD Findings
For each claim area, extract:
- The finding
- The verification method
- The analyst's assessment

### Section 4: Risk Matrix
Extract all risks with:
- Description
- Severity assessment
- Mitigation status

### Section 5: Competitive Analysis
- Direct competitors
- Differentiation assessment
- Competitive risks

### Section 6: Return Analysis
- Entry valuation assessment
- Return scenarios
- Exit timeline expectations

### Section 7: Recommendation
- Analyst recommendation
- Conditions or caveats
- Conviction level

Return comprehensive JSON array. This is a high-quality source, so include detailed extractions for each claim area covered in the memo.

Mark claims as "supportive" when the analyst views them positively, "risk" when flagged as concerns, and "neutral" for factual background."""
