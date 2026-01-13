"""
Pitch Deck Extractor.

Extracts evidence claims from startup pitch decks, focusing on
company overview, market, team, traction, and funding terms.
"""

from typing import List, Optional

from .base import DocumentExtractor, ExtractionConfig, LLMCallFn
from ..ontology import ClaimType


class PitchDeckExtractor(DocumentExtractor):
    """
    Extractor for pitch deck documents.

    Pitch decks typically contain high-level claims about:
    - Company identity and vision
    - Market size and opportunity
    - Team background
    - Product and traction metrics
    - Funding terms and use of proceeds
    """

    DOC_TYPE = "pitch_deck"

    SUPPORTED_CLAIM_TYPES = [
        ClaimType.COMPANY_IDENTITY,
        ClaimType.ROUND_TERMS,
        ClaimType.USE_OF_PROCEEDS,
        ClaimType.TEAM_QUALITY,
        ClaimType.TEAM_COMPOSITION,
        ClaimType.PRODUCT_READINESS,
        ClaimType.DIFFERENTIATION,
        ClaimType.MARKET_SCOPE,
        ClaimType.BUSINESS_MODEL,
        ClaimType.TRACTION,
    ]

    def get_system_prompt(self) -> str:
        return """You are an expert VC analyst extracting structured evidence claims from startup pitch decks.

Your task is to identify and extract factual claims that can be used for investment due diligence.

## Guidelines

1. **Extract only factual claims** - not opinions, aspirations, or marketing language
2. **Assign appropriate confidence scores**:
   - 0.9-1.0: Clear, verifiable facts (names, dates, funding amounts)
   - 0.7-0.9: Metrics with some uncertainty (reported revenue, user counts)
   - 0.5-0.7: Estimates or projections presented as facts
   - 0.3-0.5: Implied or inferred information
3. **Assign polarity** based on investment thesis impact:
   - "supportive": Evidence supporting investment (strong metrics, experienced team)
   - "risk": Red flags or concerns (unproven market, key person dependency)
   - "neutral": Informational without clear positive/negative signal
4. **Include source locators** when possible (page numbers, section names)
5. **Capture exact quotes** for key metrics and claims

## Claim Types to Extract

""" + self.get_claim_types_description() + """

## Output Format

Return a JSON array of claims:
```json
[
  {
    "claim_type": "traction",
    "field": "mrr",
    "value": 150000,
    "confidence": 0.85,
    "polarity": "supportive",
    "locator": "page 8, Traction slide",
    "quote": "MRR of $150K as of Q3 2024",
    "rationale": "Clear metric presented in traction section"
  }
]
```

Be thorough but focus on quality over quantity. Only extract claims you are confident about."""

    def get_extraction_prompt(self, document_text: str, doc_id: str) -> str:
        return f"""Extract evidence claims from the following pitch deck.

Document ID: {doc_id}

---
PITCH DECK CONTENT:
---

{document_text}

---

Extract all relevant claims following the guidelines. Focus on:
1. Company basics (name, founding date, location)
2. Funding terms (raise amount, valuation, instrument)
3. Team credentials and composition
4. Market size claims (TAM, SAM, SOM)
5. Traction metrics (revenue, users, growth)
6. Product stage and differentiation
7. Business model and unit economics

Return your findings as a JSON array of claims."""


class PitchDeckExtractorV2(PitchDeckExtractor):
    """
    Enhanced pitch deck extractor with section-aware prompting.

    Uses a two-pass approach:
    1. Identify sections in the deck
    2. Extract claims from each section with context
    """

    def get_system_prompt(self) -> str:
        return """You are an expert VC analyst extracting structured evidence claims from startup pitch decks.

Your task is to identify and extract factual claims that can be used for investment due diligence.

## Important Guidelines

1. **Be skeptical of marketing language** - Distinguish facts from aspirations
2. **Note uncertainty sources**:
   - "Projected" or "expected" = lower confidence
   - "As of [date]" = higher confidence if recent
   - Vague terms like "significant" or "leading" = low confidence
3. **Watch for red flags to mark as risks**:
   - Missing key metrics that should be present
   - Inconsistent numbers across slides
   - Unusually high growth claims without proof
   - Vague team backgrounds

## Confidence Calibration

- **0.95**: Official documents, legal names, registered information
- **0.85**: Clearly stated metrics with date references
- **0.75**: Metrics without clear date or methodology
- **0.60**: Estimates or "approximately" figures
- **0.45**: Projections or forward-looking statements
- **0.35**: Implied information requiring inference

## Polarity Assignment

**Supportive indicators:**
- Revenue growing >20% MoM/QoQ
- Strong retention metrics (>90%)
- Experienced founding team with relevant background
- Clear product-market fit signals
- Reasonable unit economics

**Risk indicators:**
- No revenue or very early stage
- High burn rate relative to traction
- Inexperienced team in the domain
- Crowded market without clear differentiation
- Regulatory uncertainty

## Output Format

Return a JSON array. Example:
```json
[
  {
    "claim_type": "company_identity",
    "field": "legal_name",
    "value": "TechCorp Inc.",
    "confidence": 0.95,
    "polarity": "neutral",
    "locator": "Title slide",
    "quote": "TechCorp Inc.",
    "rationale": "Company name clearly stated on title slide"
  },
  {
    "claim_type": "traction",
    "field": "mrr",
    "value": 150000,
    "confidence": 0.85,
    "polarity": "supportive",
    "locator": "Page 8 - Traction",
    "quote": "$150K MRR as of October 2024",
    "rationale": "Current MRR with date reference indicates real traction"
  },
  {
    "claim_type": "execution_risk",
    "field": "key_hire_gap",
    "value": "No CTO identified",
    "confidence": 0.70,
    "polarity": "risk",
    "locator": "Page 5 - Team",
    "quote": null,
    "rationale": "Team slide shows CEO and CMO but no technical leadership"
  }
]
```

""" + self.get_claim_types_description()

    def get_extraction_prompt(self, document_text: str, doc_id: str) -> str:
        return f"""Analyze this pitch deck and extract evidence claims for VC due diligence.

Document ID: {doc_id}

---
PITCH DECK CONTENT:
---

{document_text}

---

## Extraction Instructions

1. **First pass**: Identify all sections/slides in the deck
2. **Second pass**: Extract claims from each section

Focus areas by typical pitch deck structure:
- **Title/Overview**: Company name, tagline, sector
- **Problem/Solution**: Market context (be skeptical of problem framing)
- **Product**: Stage, features, screenshots
- **Market**: TAM/SAM/SOM claims (verify methodology if stated)
- **Business Model**: Revenue model, pricing, unit economics
- **Traction**: Metrics, customers, growth rates
- **Team**: Backgrounds, experience, gaps
- **Financials**: Burn, runway, projections
- **Ask**: Round terms, use of proceeds

Return comprehensive JSON array of claims. Include BOTH positive and negative observations (risks)."""
