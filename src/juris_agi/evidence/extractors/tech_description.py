"""
Technical Description Extractor.

Extracts evidence claims from technical documentation, architecture docs,
and technology descriptions.
"""

from typing import List, Optional

from .base import DocumentExtractor, ExtractionConfig, LLMCallFn
from ..ontology import ClaimType


class TechDescriptionExtractor(DocumentExtractor):
    """
    Extractor for technical documentation.

    Technical docs typically contain:
    - Product architecture and stack
    - Technical moat and IP
    - Development stage and roadmap
    - Technical team composition
    - Security and compliance details
    """

    DOC_TYPE = "tech_description"

    SUPPORTED_CLAIM_TYPES = [
        ClaimType.PRODUCT_READINESS,
        ClaimType.TECHNICAL_MOAT,
        ClaimType.DIFFERENTIATION,
        ClaimType.TEAM_QUALITY,
        ClaimType.TEAM_COMPOSITION,
        ClaimType.EXECUTION_RISK,
        ClaimType.REGULATORY_RISK,
    ]

    def get_system_prompt(self) -> str:
        return """You are a technical due diligence expert evaluating startup technology for VC investment.

Your task is to extract structured claims about technical capabilities, risks, and defensibility.

## Guidelines

1. **Assess technical credibility**:
   - Is the architecture appropriate for the problem?
   - Are technology choices reasonable?
   - Is there evidence of technical depth vs marketing fluff?

2. **Identify defensibility factors**:
   - Proprietary algorithms or data
   - Patents (filed vs granted)
   - Infrastructure or scale advantages
   - Unique technical expertise

3. **Spot technical risks**:
   - Technical debt indicators
   - Scalability concerns
   - Security vulnerabilities
   - Dependency risks (key libraries, APIs, vendors)

4. **Evaluate team technical capability**:
   - Engineering team composition
   - Relevant technical backgrounds
   - Ability to execute the roadmap

## Confidence Calibration for Technical Claims

- **0.90**: Verified technical details (open source code, patents)
- **0.75**: Detailed architecture with specific technologies
- **0.60**: General technical descriptions without specifics
- **0.45**: Marketing-style technical claims
- **0.30**: Vague or buzzword-heavy descriptions

## Polarity Guidelines

**Supportive indicators:**
- Granted patents or strong IP position
- Proprietary data assets
- Proven scalability (with evidence)
- Experienced technical team from relevant companies
- Modern, appropriate tech stack

**Risk indicators:**
- No clear technical moat
- Outdated technology stack
- Single points of failure
- Key person technical dependency
- Missing security/compliance basics
- Unrealistic technical roadmap

## Output Format

```json
[
  {
    "claim_type": "technical_moat",
    "field": "proprietary_data",
    "value": "10M+ labeled training examples",
    "confidence": 0.70,
    "polarity": "supportive",
    "locator": "Architecture doc, Section 3.2",
    "quote": "Our models are trained on 10M+ proprietary labeled examples",
    "rationale": "Significant data asset if accurate; would take competitors years to replicate"
  },
  {
    "claim_type": "execution_risk",
    "field": "technical_debt",
    "value": "Monolithic architecture needs refactoring",
    "confidence": 0.65,
    "polarity": "risk",
    "locator": "Architecture doc, Section 2.1",
    "quote": "Current monolith handles 1K req/sec",
    "rationale": "Monolithic architecture may struggle to scale; migration risk if growth accelerates"
  }
]
```

""" + self.get_claim_types_description()

    def get_extraction_prompt(self, document_text: str, doc_id: str) -> str:
        return f"""Analyze this technical documentation and extract evidence claims for VC due diligence.

Document ID: {doc_id}

---
TECHNICAL DOCUMENTATION:
---

{document_text}

---

## Technical Due Diligence Focus

### 1. Product & Architecture
- What stage is the product (prototype/MVP/production)?
- What's the technology stack?
- How is the system architected?
- What are the key technical components?

### 2. Technical Moat
- Any patents filed or granted?
- Proprietary algorithms or data?
- Unique technical capabilities?
- What would it take to replicate?

### 3. Scalability & Performance
- Current scale (users, requests, data volume)
- Evidence of scaling capability
- Performance benchmarks
- Infrastructure setup

### 4. Security & Compliance
- Security measures described
- Compliance certifications
- Data handling practices
- Vulnerability management

### 5. Technical Team
- Engineering team size and composition
- Relevant technical backgrounds
- Key technical hires needed

### 6. Technical Risks
- Technical debt indicators
- Dependency risks
- Scalability bottlenecks
- Bus factor / key person risks

Return comprehensive JSON array of technical claims, including both strengths and concerns."""


class TechDescriptionExtractorV2(TechDescriptionExtractor):
    """
    Enhanced technical extractor with deeper analysis.

    Evaluates architecture quality, tech debt, and execution capability.
    """

    def get_system_prompt(self) -> str:
        return """You are a senior technical advisor conducting technology due diligence for a VC firm.

Your task is to extract structured claims AND provide technical assessment.

## Technical Assessment Framework

### Architecture Quality
- **Appropriate complexity**: Is the architecture right-sized for the problem?
- **Separation of concerns**: Clean boundaries between components?
- **Extensibility**: Can it evolve with the business?
- **Operational maturity**: Monitoring, logging, deployment practices

### Technical Moat Evaluation
Rate the defensibility (in rationale):
- **Strong moat**: Patents + proprietary data + unique expertise
- **Medium moat**: One or two significant advantages
- **Weak moat**: Easily replicable technology

### Team Technical Capability
Assess ability to execute:
- Relevant domain expertise
- Experience at scale
- Full-stack capability vs gaps
- Technical leadership quality

### Risk Identification
Categorize risks:
- **Critical**: Could cause company failure
- **Significant**: Will slow growth or increase costs
- **Minor**: Should be addressed but manageable

## Red Flags to Surface

- No version control or CI/CD
- No testing mentioned
- Single database with no replication
- Hardcoded configurations
- No security measures described
- Outdated frameworks/libraries
- No technical documentation
- Unrealistic scaling claims
- "Blockchain/AI/ML" as buzzwords without substance

## Positive Signals

- Microservices or modular architecture (where appropriate)
- Comprehensive testing strategy
- Security-first design
- Clear technical roadmap with milestones
- Evidence of technical iteration based on learnings
- Active open source contributions
- Strong technical blog/content

## Output Format

```json
[
  {
    "claim_type": "product_readiness",
    "field": "stage",
    "value": "production",
    "confidence": 0.85,
    "polarity": "supportive",
    "locator": "Section 1 - Overview",
    "quote": "Production deployment serving 50K DAU",
    "rationale": "Product is live with meaningful scale; past MVP stage"
  },
  {
    "claim_type": "technical_moat",
    "field": "algo_advantage",
    "value": "Proprietary ML pipeline with 3x accuracy vs alternatives",
    "confidence": 0.55,
    "polarity": "supportive",
    "locator": "Section 4 - ML Architecture",
    "quote": "Our model achieves 94% accuracy vs 31% for GPT-4",
    "rationale": "Strong claim but needs verification; would be significant if true"
  }
]
```

""" + self.get_claim_types_description()

    def get_extraction_prompt(self, document_text: str, doc_id: str) -> str:
        return f"""Conduct comprehensive technical due diligence on this documentation.

Document ID: {doc_id}

---
TECHNICAL DOCUMENTATION:
---

{document_text}

---

## Analysis Required

### Phase 1: Inventory Technical Claims
Extract all stated technical facts and capabilities.

### Phase 2: Assess Credibility
For each claim, evaluate:
- Is this verifiable?
- Does it align with industry norms?
- Are there red flags in how it's described?

### Phase 3: Identify Gaps
What's NOT mentioned that should be?
- Security practices
- Testing approach
- Deployment/DevOps
- Data backup/recovery
- Incident response

### Phase 4: Rate Technical Risk
Overall assessment of technical execution risk.

### Phase 5: Evaluate Moat
How defensible is the technology?
What would it take to replicate?

Return comprehensive JSON array covering:
1. Product readiness facts
2. Technical moat evidence
3. Differentiation factors
4. Team technical capability
5. Technical risks and concerns

Include both positive findings and areas of concern."""
