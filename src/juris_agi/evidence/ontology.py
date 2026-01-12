"""
VC Investment Decision Ontology.

Defines the 15 claim types for structuring evidence in VC due diligence.
Each claim type represents a category of evidence that can be gathered
and evaluated when assessing an investment opportunity.
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Optional


class ClaimType(Enum):
    """
    The 15 claim types for VC investment evidence.

    Each type represents a distinct category of due diligence information.
    """

    # Identity & Structure
    COMPANY_IDENTITY = "company_identity"
    """Basic company information: name, legal entity, jurisdiction, founding date."""

    ROUND_TERMS = "round_terms"
    """Current round structure: valuation, amount, instrument type, lead investor."""

    USE_OF_PROCEEDS = "use_of_proceeds"
    """How raised capital will be deployed: hiring, R&D, marketing, runway."""

    # Team
    TEAM_QUALITY = "team_quality"
    """Founder/team credentials: background, domain expertise, track record."""

    TEAM_COMPOSITION = "team_composition"
    """Team structure: key hires, roles filled, gaps, vesting."""

    # Product & Technology
    PRODUCT_READINESS = "product_readiness"
    """Product stage: prototype, MVP, GA; development timeline."""

    TECHNICAL_MOAT = "technical_moat"
    """Defensible technology: IP, patents, proprietary algorithms, data assets."""

    DIFFERENTIATION = "differentiation"
    """Competitive positioning: unique value prop, switching costs, network effects."""

    # Market
    MARKET_SCOPE = "market_scope"
    """Market size and dynamics: TAM, SAM, SOM, growth rate, timing."""

    BUSINESS_MODEL = "business_model"
    """Revenue model: pricing, unit economics, LTV/CAC, margins."""

    TRACTION = "traction"
    """Evidence of progress: revenue, users, growth rate, retention, engagement."""

    # Risk Assessment
    EXECUTION_RISK = "execution_risk"
    """Operational risks: team capacity, technical challenges, go-to-market."""

    REGULATORY_RISK = "regulatory_risk"
    """Legal/regulatory exposure: compliance, licensing, pending litigation."""

    CAPITAL_INTENSITY = "capital_intensity"
    """Capital requirements: burn rate, runway, path to profitability."""

    # Exit
    EXIT_LOGIC = "exit_logic"
    """Exit potential: acquirer universe, IPO viability, comparable exits."""


# Human-readable descriptions for each claim type
CLAIM_TYPE_DESCRIPTIONS = {
    ClaimType.COMPANY_IDENTITY: (
        "Basic company information including legal name, entity type, "
        "jurisdiction of incorporation, founding date, and headquarters location."
    ),
    ClaimType.ROUND_TERMS: (
        "Details of the current funding round including target raise, "
        "pre/post-money valuation, instrument type (SAFE, convertible, priced), "
        "and lead investor status."
    ),
    ClaimType.USE_OF_PROCEEDS: (
        "Breakdown of how raised capital will be allocated across "
        "engineering, sales, marketing, operations, and expected runway extension."
    ),
    ClaimType.TEAM_QUALITY: (
        "Assessment of founder and key team member backgrounds, "
        "relevant experience, domain expertise, and prior entrepreneurial success."
    ),
    ClaimType.TEAM_COMPOSITION: (
        "Current team structure, key roles filled and open, "
        "equity distribution, vesting schedules, and advisory relationships."
    ),
    ClaimType.PRODUCT_READINESS: (
        "Current product development stage (idea, prototype, MVP, beta, GA), "
        "remaining milestones, and technical roadmap."
    ),
    ClaimType.TECHNICAL_MOAT: (
        "Defensibility of technology through patents, trade secrets, "
        "proprietary data, algorithms, or infrastructure advantages."
    ),
    ClaimType.DIFFERENTIATION: (
        "Competitive positioning including unique value proposition, "
        "switching costs, network effects, and barriers to entry."
    ),
    ClaimType.MARKET_SCOPE: (
        "Market size analysis (TAM, SAM, SOM), growth projections, "
        "market timing, and key industry dynamics."
    ),
    ClaimType.BUSINESS_MODEL: (
        "Revenue model details including pricing strategy, unit economics, "
        "gross margins, LTV/CAC ratios, and path to profitability."
    ),
    ClaimType.TRACTION: (
        "Quantitative evidence of progress including revenue, user growth, "
        "retention metrics, engagement, and key milestones achieved."
    ),
    ClaimType.EXECUTION_RISK: (
        "Assessment of operational risks including team capacity gaps, "
        "technical challenges, go-to-market complexity, and dependency risks."
    ),
    ClaimType.REGULATORY_RISK: (
        "Legal and regulatory considerations including compliance requirements, "
        "licensing needs, pending litigation, and regulatory uncertainty."
    ),
    ClaimType.CAPITAL_INTENSITY: (
        "Capital efficiency analysis including current burn rate, "
        "projected runway, capital requirements to key milestones."
    ),
    ClaimType.EXIT_LOGIC: (
        "Exit opportunity analysis including potential acquirers, "
        "IPO viability, comparable transactions, and expected timeframe."
    ),
}


@dataclass(frozen=True)
class ClaimTypeInfo:
    """Metadata about a claim type."""
    claim_type: ClaimType
    description: str
    typical_fields: List[str]
    risk_relevant: bool

    @property
    def name(self) -> str:
        return self.claim_type.value


# Define typical fields for each claim type (for validation guidance)
CLAIM_TYPE_METADATA = {
    ClaimType.COMPANY_IDENTITY: ClaimTypeInfo(
        claim_type=ClaimType.COMPANY_IDENTITY,
        description=CLAIM_TYPE_DESCRIPTIONS[ClaimType.COMPANY_IDENTITY],
        typical_fields=["legal_name", "entity_type", "jurisdiction", "founded_date", "hq_location"],
        risk_relevant=False,
    ),
    ClaimType.ROUND_TERMS: ClaimTypeInfo(
        claim_type=ClaimType.ROUND_TERMS,
        description=CLAIM_TYPE_DESCRIPTIONS[ClaimType.ROUND_TERMS],
        typical_fields=["raise_amount", "pre_money_valuation", "instrument_type", "lead_investor"],
        risk_relevant=False,
    ),
    ClaimType.USE_OF_PROCEEDS: ClaimTypeInfo(
        claim_type=ClaimType.USE_OF_PROCEEDS,
        description=CLAIM_TYPE_DESCRIPTIONS[ClaimType.USE_OF_PROCEEDS],
        typical_fields=["engineering_pct", "sales_pct", "marketing_pct", "runway_months"],
        risk_relevant=False,
    ),
    ClaimType.TEAM_QUALITY: ClaimTypeInfo(
        claim_type=ClaimType.TEAM_QUALITY,
        description=CLAIM_TYPE_DESCRIPTIONS[ClaimType.TEAM_QUALITY],
        typical_fields=["founder_background", "domain_expertise", "prior_exits", "years_experience"],
        risk_relevant=False,
    ),
    ClaimType.TEAM_COMPOSITION: ClaimTypeInfo(
        claim_type=ClaimType.TEAM_COMPOSITION,
        description=CLAIM_TYPE_DESCRIPTIONS[ClaimType.TEAM_COMPOSITION],
        typical_fields=["team_size", "key_roles_filled", "open_roles", "advisor_count"],
        risk_relevant=False,
    ),
    ClaimType.PRODUCT_READINESS: ClaimTypeInfo(
        claim_type=ClaimType.PRODUCT_READINESS,
        description=CLAIM_TYPE_DESCRIPTIONS[ClaimType.PRODUCT_READINESS],
        typical_fields=["stage", "months_to_ga", "tech_debt_level", "platform_maturity"],
        risk_relevant=False,
    ),
    ClaimType.TECHNICAL_MOAT: ClaimTypeInfo(
        claim_type=ClaimType.TECHNICAL_MOAT,
        description=CLAIM_TYPE_DESCRIPTIONS[ClaimType.TECHNICAL_MOAT],
        typical_fields=["patents_filed", "patents_granted", "proprietary_data", "algo_advantage"],
        risk_relevant=False,
    ),
    ClaimType.DIFFERENTIATION: ClaimTypeInfo(
        claim_type=ClaimType.DIFFERENTIATION,
        description=CLAIM_TYPE_DESCRIPTIONS[ClaimType.DIFFERENTIATION],
        typical_fields=["unique_value_prop", "switching_costs", "network_effects", "competitors"],
        risk_relevant=False,
    ),
    ClaimType.MARKET_SCOPE: ClaimTypeInfo(
        claim_type=ClaimType.MARKET_SCOPE,
        description=CLAIM_TYPE_DESCRIPTIONS[ClaimType.MARKET_SCOPE],
        typical_fields=["tam", "sam", "som", "market_growth_rate", "market_timing"],
        risk_relevant=False,
    ),
    ClaimType.BUSINESS_MODEL: ClaimTypeInfo(
        claim_type=ClaimType.BUSINESS_MODEL,
        description=CLAIM_TYPE_DESCRIPTIONS[ClaimType.BUSINESS_MODEL],
        typical_fields=["revenue_model", "arpu", "ltv", "cac", "gross_margin"],
        risk_relevant=False,
    ),
    ClaimType.TRACTION: ClaimTypeInfo(
        claim_type=ClaimType.TRACTION,
        description=CLAIM_TYPE_DESCRIPTIONS[ClaimType.TRACTION],
        typical_fields=["mrr", "arr", "users", "growth_rate", "retention", "nps"],
        risk_relevant=False,
    ),
    ClaimType.EXECUTION_RISK: ClaimTypeInfo(
        claim_type=ClaimType.EXECUTION_RISK,
        description=CLAIM_TYPE_DESCRIPTIONS[ClaimType.EXECUTION_RISK],
        typical_fields=["risk_category", "likelihood", "impact", "mitigation"],
        risk_relevant=True,
    ),
    ClaimType.REGULATORY_RISK: ClaimTypeInfo(
        claim_type=ClaimType.REGULATORY_RISK,
        description=CLAIM_TYPE_DESCRIPTIONS[ClaimType.REGULATORY_RISK],
        typical_fields=["jurisdiction", "compliance_status", "pending_issues", "regulatory_clarity"],
        risk_relevant=True,
    ),
    ClaimType.CAPITAL_INTENSITY: ClaimTypeInfo(
        claim_type=ClaimType.CAPITAL_INTENSITY,
        description=CLAIM_TYPE_DESCRIPTIONS[ClaimType.CAPITAL_INTENSITY],
        typical_fields=["monthly_burn", "runway_months", "capital_to_breakeven", "capex_needs"],
        risk_relevant=True,
    ),
    ClaimType.EXIT_LOGIC: ClaimTypeInfo(
        claim_type=ClaimType.EXIT_LOGIC,
        description=CLAIM_TYPE_DESCRIPTIONS[ClaimType.EXIT_LOGIC],
        typical_fields=["potential_acquirers", "ipo_viability", "comparable_exits", "expected_timeline"],
        risk_relevant=False,
    ),
}


def get_claim_type(name: str) -> Optional[ClaimType]:
    """
    Get ClaimType from string name.

    Args:
        name: Claim type name (case-insensitive, underscore or hyphen separated)

    Returns:
        ClaimType if found, None otherwise
    """
    normalized = name.lower().replace("-", "_")
    for ct in ClaimType:
        if ct.value == normalized:
            return ct
    return None


def get_all_claim_types() -> List[ClaimType]:
    """Return all claim types in ontology order."""
    return list(ClaimType)


def get_risk_claim_types() -> List[ClaimType]:
    """Return claim types that are risk-relevant."""
    return [
        ct for ct, info in CLAIM_TYPE_METADATA.items()
        if info.risk_relevant
    ]


def get_claim_type_info(claim_type: ClaimType) -> ClaimTypeInfo:
    """Get metadata for a claim type."""
    return CLAIM_TYPE_METADATA[claim_type]
