"""
Financial Model Extractor.

Extracts evidence claims from financial models, projections,
and financial data rooms.
"""

from typing import List, Optional

from .base import DocumentExtractor, ExtractionConfig, LLMCallFn
from ..ontology import ClaimType


class FinancialModelExtractor(DocumentExtractor):
    """
    Extractor for financial models and projections.

    Financial models typically contain:
    - Historical financial metrics
    - Forward projections
    - Unit economics
    - Burn rate and runway calculations
    - Revenue model assumptions
    """

    DOC_TYPE = "financial_model"

    SUPPORTED_CLAIM_TYPES = [
        ClaimType.ROUND_TERMS,
        ClaimType.USE_OF_PROCEEDS,
        ClaimType.BUSINESS_MODEL,
        ClaimType.TRACTION,
        ClaimType.CAPITAL_INTENSITY,
    ]

    def get_system_prompt(self) -> str:
        return """You are an expert financial analyst extracting structured evidence claims from startup financial models.

Your task is to identify and extract quantitative claims about business performance, projections, and unit economics.

## Guidelines

1. **Distinguish historical from projected data**:
   - Historical (actual) data: Higher confidence (0.85-0.95)
   - Projections/forecasts: Lower confidence (0.3-0.6)
   - Always note in rationale whether data is historical or projected

2. **Verify internal consistency**:
   - Flag if numbers don't add up
   - Note if growth assumptions seem unrealistic
   - Check if unit economics are plausible

3. **Key metrics to extract**:
   - Revenue (MRR, ARR, GMV)
   - Costs (COGS, OpEx breakdown)
   - Unit economics (LTV, CAC, payback period)
   - Cash position and runway
   - Growth rates (MoM, QoQ, YoY)

4. **Polarity assignment**:
   - "supportive": Healthy metrics (positive unit economics, long runway, strong growth)
   - "risk": Concerning metrics (high burn, negative margins, unrealistic projections)
   - "neutral": Standard reporting data

## Confidence Calibration for Financial Data

- **0.95**: Audited financials or bank statements
- **0.85**: Internal financial reports with clear methodology
- **0.70**: Self-reported metrics without audit
- **0.55**: Projections with reasonable assumptions
- **0.40**: Aggressive projections or hockey-stick forecasts
- **0.30**: Aspirational targets without clear basis

## Output Format

Return a JSON array:
```json
[
  {
    "claim_type": "traction",
    "field": "arr",
    "value": 1200000,
    "confidence": 0.85,
    "polarity": "supportive",
    "locator": "P&L Tab, Row 45, Column D",
    "quote": "ARR: $1.2M (Dec 2024 Actual)",
    "rationale": "Historical ARR from P&L statement, clearly marked as actual"
  },
  {
    "claim_type": "capital_intensity",
    "field": "monthly_burn",
    "value": 85000,
    "confidence": 0.80,
    "polarity": "risk",
    "locator": "Cash Flow Tab",
    "quote": "Monthly Net Burn: $85K",
    "rationale": "High burn rate relative to revenue; only 8 months runway at current rate"
  }
]
```

""" + self.get_claim_types_description()

    def get_extraction_prompt(self, document_text: str, doc_id: str) -> str:
        return f"""Extract financial claims from this financial model/spreadsheet data.

Document ID: {doc_id}

---
FINANCIAL MODEL CONTENT:
---

{document_text}

---

## Extraction Focus

1. **Revenue Metrics**
   - Current MRR/ARR (distinguish actual vs projected)
   - Revenue growth rates
   - Revenue breakdown by product/segment if available

2. **Unit Economics**
   - Customer acquisition cost (CAC)
   - Lifetime value (LTV)
   - LTV/CAC ratio
   - Gross margin
   - Contribution margin

3. **Cash & Runway**
   - Current cash position
   - Monthly burn rate
   - Runway in months
   - Path to profitability

4. **Cost Structure**
   - Fixed vs variable costs
   - Headcount costs
   - Marketing spend

5. **Projections** (mark clearly with lower confidence)
   - Revenue forecasts
   - Expense projections
   - Break-even timeline

For each claim, clearly indicate:
- Whether it's historical (actual) or projected
- The time period it refers to
- Any assumptions underlying the number

Return comprehensive JSON array of financial claims."""


class FinancialModelExtractorV2(FinancialModelExtractor):
    """
    Enhanced financial model extractor with deeper analysis.

    Includes validation checks and cross-references between metrics.
    """

    def get_system_prompt(self) -> str:
        return """You are a senior financial analyst conducting due diligence on a startup's financial model.

Your task is to extract structured claims AND identify potential concerns or inconsistencies.

## Analysis Framework

### 1. Revenue Quality Assessment
- Is revenue recognized appropriately?
- What's the mix of recurring vs one-time?
- Are there any large customer concentrations?

### 2. Unit Economics Validation
- Do LTV/CAC calculations use reasonable assumptions?
- Is payback period realistic for the business model?
- Are cohort economics improving or degrading?

### 3. Burn Rate Analysis
- What's driving the burn?
- Is the burn justified by growth?
- What levers exist to reduce burn if needed?

### 4. Projection Reasonableness
- Are growth assumptions backed by evidence?
- How do projections compare to industry benchmarks?
- What's the sensitivity to key assumptions?

## Red Flags to Mark as Risks

- Revenue growing faster than customer count (pricing games?)
- LTV calculated with unrealistic churn assumptions
- CAC excluding important costs
- Hockey-stick projections without clear catalyst
- Declining gross margins
- Increasing CAC over time
- Cash runway < 12 months without clear plan

## Confidence Scoring

For financial data, also consider:
- **Data source quality**: Audited > internal reports > estimates
- **Recency**: Last month > last quarter > last year
- **Methodology clarity**: Explicit calculations > implied > unclear

## Output Format

```json
[
  {
    "claim_type": "business_model",
    "field": "gross_margin",
    "value": 0.72,
    "confidence": 0.85,
    "polarity": "supportive",
    "locator": "Unit Economics Tab, Cell F12",
    "quote": "Gross Margin: 72%",
    "rationale": "Strong gross margin typical of SaaS; calculated consistently across periods"
  },
  {
    "claim_type": "capital_intensity",
    "field": "runway_months",
    "value": 8,
    "confidence": 0.75,
    "polarity": "risk",
    "locator": "Cash Flow Summary",
    "quote": "$680K cash / $85K monthly burn = 8 months",
    "rationale": "Short runway creates fundraising pressure; limited time to hit milestones"
  }
]
```

""" + self.get_claim_types_description()

    def get_extraction_prompt(self, document_text: str, doc_id: str) -> str:
        return f"""Conduct a thorough analysis of this financial model and extract evidence claims.

Document ID: {doc_id}

---
FINANCIAL MODEL CONTENT:
---

{document_text}

---

## Analysis Required

### Phase 1: Extract Raw Metrics
Extract all quantifiable metrics with their sources and time periods.

### Phase 2: Validate Consistency
Check if numbers are internally consistent:
- Do revenues and costs sum correctly?
- Are growth rates consistent with absolute numbers?
- Does cash flow reconcile with P&L and balance sheet?

### Phase 3: Assess Quality
For each metric, assess:
- Is this historical (verified) or projected?
- What assumptions underlie this number?
- How sensitive is this to key inputs?

### Phase 4: Identify Concerns
Flag any:
- Unusual trends or patterns
- Aggressive assumptions
- Missing standard metrics
- Potential accounting concerns

Return comprehensive JSON array including both positive findings and concerns.
Mark concerns with "risk" polarity and explain in rationale."""
