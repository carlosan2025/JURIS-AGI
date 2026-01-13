"""
Seed demo data for JURIS-AGI VC mode.

Provides:
- 3 historical deals with clear outcomes (invest/pass)
- 1 ambiguous new deal for live demo

All data is fictional and for demonstration purposes only.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Demo data directory
DEMO_DIR = Path(__file__).parent
DATA_DIR = DEMO_DIR / "data"


def create_historical_deals() -> List[Dict[str, Any]]:
    """
    Create 3 historical deals with clear investment decisions.

    These demonstrate how the system works on "known" cases.
    """
    deals = []

    # ==========================================================================
    # Deal 1: Clear INVEST - Strong SaaS metrics
    # ==========================================================================
    deals.append({
        "company_id": "cloudmetrics-2023",
        "company_name": "CloudMetrics Inc.",
        "sector": "Enterprise SaaS",
        "stage": "Series A",
        "outcome": "invest",
        "outcome_rationale": "Strong unit economics, experienced team, clear product-market fit",
        "evidence_graph": {
            "company_id": "cloudmetrics-2023",
            "claims": [
                {
                    "claim_type": "company_identity",
                    "field": "legal_name",
                    "value": "CloudMetrics Inc.",
                    "confidence": 1.0,
                    "polarity": "neutral",
                },
                {
                    "claim_type": "round_terms",
                    "field": "pre_money_valuation",
                    "value": 25000000,
                    "confidence": 0.95,
                    "polarity": "neutral",
                    "unit": "USD",
                },
                {
                    "claim_type": "round_terms",
                    "field": "raise_amount",
                    "value": 5000000,
                    "confidence": 0.95,
                    "polarity": "neutral",
                    "unit": "USD",
                },
                {
                    "claim_type": "traction",
                    "field": "arr",
                    "value": 2400000,
                    "confidence": 0.92,
                    "polarity": "supportive",
                    "unit": "USD",
                },
                {
                    "claim_type": "traction",
                    "field": "growth_rate",
                    "value": 0.15,
                    "confidence": 0.88,
                    "polarity": "supportive",
                    "unit": "monthly",
                },
                {
                    "claim_type": "traction",
                    "field": "net_revenue_retention",
                    "value": 1.25,
                    "confidence": 0.85,
                    "polarity": "supportive",
                },
                {
                    "claim_type": "business_model",
                    "field": "gross_margin",
                    "value": 0.82,
                    "confidence": 0.90,
                    "polarity": "supportive",
                },
                {
                    "claim_type": "business_model",
                    "field": "ltv_cac_ratio",
                    "value": 4.2,
                    "confidence": 0.85,
                    "polarity": "supportive",
                },
                {
                    "claim_type": "team_quality",
                    "field": "founder_background",
                    "value": "Ex-Datadog engineering lead, 12 years observability",
                    "confidence": 0.95,
                    "polarity": "supportive",
                },
                {
                    "claim_type": "team_composition",
                    "field": "team_size",
                    "value": 18,
                    "confidence": 0.90,
                    "polarity": "neutral",
                },
                {
                    "claim_type": "market_scope",
                    "field": "tam",
                    "value": 15000000000,
                    "confidence": 0.70,
                    "polarity": "supportive",
                    "unit": "USD",
                },
                {
                    "claim_type": "differentiation",
                    "field": "unique_value_prop",
                    "value": "AI-powered anomaly detection with 10x fewer false positives",
                    "confidence": 0.75,
                    "polarity": "supportive",
                },
                {
                    "claim_type": "capital_intensity",
                    "field": "runway_months",
                    "value": 24,
                    "confidence": 0.85,
                    "polarity": "supportive",
                },
            ],
        },
    })

    # ==========================================================================
    # Deal 2: Clear PASS - Weak fundamentals, regulatory risk
    # ==========================================================================
    deals.append({
        "company_id": "cryptolend-2023",
        "company_name": "CryptoLend Finance",
        "sector": "Crypto/DeFi",
        "stage": "Seed",
        "outcome": "pass",
        "outcome_rationale": "Regulatory uncertainty, unproven business model, inexperienced team",
        "evidence_graph": {
            "company_id": "cryptolend-2023",
            "claims": [
                {
                    "claim_type": "company_identity",
                    "field": "legal_name",
                    "value": "CryptoLend Finance Ltd.",
                    "confidence": 1.0,
                    "polarity": "neutral",
                },
                {
                    "claim_type": "round_terms",
                    "field": "pre_money_valuation",
                    "value": 12000000,
                    "confidence": 0.90,
                    "polarity": "neutral",
                    "unit": "USD",
                },
                {
                    "claim_type": "traction",
                    "field": "tvl",
                    "value": 500000,
                    "confidence": 0.75,
                    "polarity": "neutral",
                    "unit": "USD",
                },
                {
                    "claim_type": "traction",
                    "field": "monthly_volume",
                    "value": 2000000,
                    "confidence": 0.70,
                    "polarity": "neutral",
                    "unit": "USD",
                },
                {
                    "claim_type": "business_model",
                    "field": "revenue_model",
                    "value": "0.3% transaction fee + spread on lending",
                    "confidence": 0.80,
                    "polarity": "neutral",
                },
                {
                    "claim_type": "business_model",
                    "field": "gross_margin",
                    "value": 0.45,
                    "confidence": 0.65,
                    "polarity": "risk",
                },
                {
                    "claim_type": "team_quality",
                    "field": "founder_background",
                    "value": "2 years crypto trading, no financial services experience",
                    "confidence": 0.85,
                    "polarity": "risk",
                },
                {
                    "claim_type": "team_composition",
                    "field": "team_size",
                    "value": 4,
                    "confidence": 0.90,
                    "polarity": "risk",
                },
                {
                    "claim_type": "regulatory_risk",
                    "field": "compliance_status",
                    "value": "No money transmitter license, SEC guidance unclear",
                    "confidence": 0.85,
                    "polarity": "risk",
                },
                {
                    "claim_type": "regulatory_risk",
                    "field": "jurisdiction",
                    "value": "Cayman Islands incorporation, US customers",
                    "confidence": 0.90,
                    "polarity": "risk",
                },
                {
                    "claim_type": "execution_risk",
                    "field": "smart_contract_audit",
                    "value": "No third-party audit completed",
                    "confidence": 0.80,
                    "polarity": "risk",
                },
                {
                    "claim_type": "market_scope",
                    "field": "market_timing",
                    "value": "Post-FTX collapse, crypto winter",
                    "confidence": 0.90,
                    "polarity": "risk",
                },
                {
                    "claim_type": "capital_intensity",
                    "field": "monthly_burn",
                    "value": 80000,
                    "confidence": 0.75,
                    "polarity": "risk",
                    "unit": "USD",
                },
            ],
        },
    })

    # ==========================================================================
    # Deal 3: Clear INVEST - Deep tech with strong IP
    # ==========================================================================
    deals.append({
        "company_id": "quantumsense-2024",
        "company_name": "QuantumSense AI",
        "sector": "Deep Tech / AI",
        "stage": "Series A",
        "outcome": "invest",
        "outcome_rationale": "Defensible technology, strong team, clear enterprise demand",
        "evidence_graph": {
            "company_id": "quantumsense-2024",
            "claims": [
                {
                    "claim_type": "company_identity",
                    "field": "legal_name",
                    "value": "QuantumSense AI Inc.",
                    "confidence": 1.0,
                    "polarity": "neutral",
                },
                {
                    "claim_type": "round_terms",
                    "field": "pre_money_valuation",
                    "value": 40000000,
                    "confidence": 0.95,
                    "polarity": "neutral",
                    "unit": "USD",
                },
                {
                    "claim_type": "round_terms",
                    "field": "raise_amount",
                    "value": 10000000,
                    "confidence": 0.95,
                    "polarity": "neutral",
                    "unit": "USD",
                },
                {
                    "claim_type": "traction",
                    "field": "arr",
                    "value": 800000,
                    "confidence": 0.88,
                    "polarity": "supportive",
                    "unit": "USD",
                },
                {
                    "claim_type": "traction",
                    "field": "pilot_customers",
                    "value": 8,
                    "confidence": 0.90,
                    "polarity": "supportive",
                },
                {
                    "claim_type": "traction",
                    "field": "enterprise_pipeline",
                    "value": 5000000,
                    "confidence": 0.70,
                    "polarity": "supportive",
                    "unit": "USD",
                },
                {
                    "claim_type": "team_quality",
                    "field": "founder_background",
                    "value": "Stanford ML PhD, ex-Google Brain research scientist",
                    "confidence": 0.95,
                    "polarity": "supportive",
                },
                {
                    "claim_type": "team_quality",
                    "field": "publications",
                    "value": "15 peer-reviewed papers, 3000+ citations",
                    "confidence": 0.90,
                    "polarity": "supportive",
                },
                {
                    "claim_type": "technical_moat",
                    "field": "patents_filed",
                    "value": 4,
                    "confidence": 0.95,
                    "polarity": "supportive",
                },
                {
                    "claim_type": "technical_moat",
                    "field": "proprietary_data",
                    "value": "50M labeled industrial sensor readings",
                    "confidence": 0.80,
                    "polarity": "supportive",
                },
                {
                    "claim_type": "differentiation",
                    "field": "benchmark_performance",
                    "value": "3x accuracy vs GPT-4 on industrial anomaly detection",
                    "confidence": 0.75,
                    "polarity": "supportive",
                },
                {
                    "claim_type": "market_scope",
                    "field": "tam",
                    "value": 8000000000,
                    "confidence": 0.70,
                    "polarity": "supportive",
                    "unit": "USD",
                },
                {
                    "claim_type": "exit_logic",
                    "field": "potential_acquirers",
                    "value": ["Siemens", "Honeywell", "Rockwell Automation", "Microsoft"],
                    "confidence": 0.65,
                    "polarity": "supportive",
                },
            ],
        },
    })

    return deals


def create_ambiguous_deal() -> Dict[str, Any]:
    """
    Create an ambiguous deal for live demo.

    This deal has mixed signals - good in some areas, concerning in others.
    Decision depends on weighting and risk tolerance.
    """
    return {
        "company_id": "healthbridge-2024",
        "company_name": "HealthBridge AI",
        "sector": "Healthcare / AI",
        "stage": "Series A",
        "outcome": None,  # Undetermined - for live analysis
        "outcome_rationale": None,
        "description": """
HealthBridge AI is building an AI-powered clinical decision support system
for primary care physicians. The platform analyzes patient history, symptoms,
and test results to suggest differential diagnoses and care pathways.

Key observations:
- Strong technical team but limited healthcare domain experience
- Early traction with 3 pilot clinics, promising engagement metrics
- Large TAM but highly regulated market with long sales cycles
- Reasonable valuation for the space
- Some regulatory clarity via FDA's AI/ML guidance, but path to clearance unclear
- Competitive landscape includes well-funded incumbents
""",
        "evidence_graph": {
            "company_id": "healthbridge-2024",
            "claims": [
                # Identity
                {
                    "claim_type": "company_identity",
                    "field": "legal_name",
                    "value": "HealthBridge AI Inc.",
                    "confidence": 1.0,
                    "polarity": "neutral",
                },
                {
                    "claim_type": "company_identity",
                    "field": "founded_date",
                    "value": "2022-03",
                    "confidence": 0.95,
                    "polarity": "neutral",
                },

                # Round terms
                {
                    "claim_type": "round_terms",
                    "field": "pre_money_valuation",
                    "value": 18000000,
                    "confidence": 0.95,
                    "polarity": "neutral",
                    "unit": "USD",
                },
                {
                    "claim_type": "round_terms",
                    "field": "raise_amount",
                    "value": 4000000,
                    "confidence": 0.95,
                    "polarity": "neutral",
                    "unit": "USD",
                },
                {
                    "claim_type": "round_terms",
                    "field": "instrument_type",
                    "value": "Series A Preferred",
                    "confidence": 0.95,
                    "polarity": "neutral",
                },

                # Traction - mixed signals
                {
                    "claim_type": "traction",
                    "field": "arr",
                    "value": 180000,
                    "confidence": 0.85,
                    "polarity": "neutral",  # Low for Series A, but healthcare is slow
                    "unit": "USD",
                },
                {
                    "claim_type": "traction",
                    "field": "pilot_clinics",
                    "value": 3,
                    "confidence": 0.90,
                    "polarity": "supportive",
                },
                {
                    "claim_type": "traction",
                    "field": "physician_nps",
                    "value": 72,
                    "confidence": 0.80,
                    "polarity": "supportive",
                },
                {
                    "claim_type": "traction",
                    "field": "diagnoses_assisted",
                    "value": 15000,
                    "confidence": 0.85,
                    "polarity": "supportive",
                },

                # Team - strong tech, weak domain
                {
                    "claim_type": "team_quality",
                    "field": "founder_background",
                    "value": "MIT CS PhD, 8 years ML at Apple Health",
                    "confidence": 0.92,
                    "polarity": "supportive",
                },
                {
                    "claim_type": "team_composition",
                    "field": "team_size",
                    "value": 12,
                    "confidence": 0.90,
                    "polarity": "neutral",
                },
                {
                    "claim_type": "team_composition",
                    "field": "clinical_advisors",
                    "value": 2,
                    "confidence": 0.85,
                    "polarity": "neutral",
                },
                {
                    "claim_type": "execution_risk",
                    "field": "domain_expertise_gap",
                    "value": "No full-time clinical leadership; advisors part-time",
                    "confidence": 0.80,
                    "polarity": "risk",
                },

                # Technical
                {
                    "claim_type": "product_readiness",
                    "field": "stage",
                    "value": "production",
                    "confidence": 0.90,
                    "polarity": "supportive",
                },
                {
                    "claim_type": "technical_moat",
                    "field": "proprietary_data",
                    "value": "500K de-identified patient encounters from pilots",
                    "confidence": 0.75,
                    "polarity": "supportive",
                },
                {
                    "claim_type": "differentiation",
                    "field": "accuracy",
                    "value": "92% concordance with specialist diagnoses in pilots",
                    "confidence": 0.70,
                    "polarity": "supportive",
                },

                # Market
                {
                    "claim_type": "market_scope",
                    "field": "tam",
                    "value": 12000000000,
                    "confidence": 0.65,
                    "polarity": "supportive",
                    "unit": "USD",
                },
                {
                    "claim_type": "market_scope",
                    "field": "market_timing",
                    "value": "Post-COVID telehealth adoption accelerating AI interest",
                    "confidence": 0.75,
                    "polarity": "supportive",
                },

                # Regulatory - key risk area
                {
                    "claim_type": "regulatory_risk",
                    "field": "fda_status",
                    "value": "Pre-submission meeting scheduled; Class II 510(k) pathway",
                    "confidence": 0.85,
                    "polarity": "neutral",
                },
                {
                    "claim_type": "regulatory_risk",
                    "field": "clearance_timeline",
                    "value": "12-18 months estimated",
                    "confidence": 0.60,
                    "polarity": "risk",
                },
                {
                    "claim_type": "regulatory_risk",
                    "field": "hipaa_compliance",
                    "value": "SOC 2 Type II certified, HIPAA BAAs in place",
                    "confidence": 0.90,
                    "polarity": "supportive",
                },

                # Business model
                {
                    "claim_type": "business_model",
                    "field": "revenue_model",
                    "value": "$500/physician/month SaaS",
                    "confidence": 0.85,
                    "polarity": "neutral",
                },
                {
                    "claim_type": "business_model",
                    "field": "sales_cycle",
                    "value": "6-9 months average",
                    "confidence": 0.75,
                    "polarity": "risk",
                },
                {
                    "claim_type": "business_model",
                    "field": "gross_margin",
                    "value": 0.75,
                    "confidence": 0.80,
                    "polarity": "supportive",
                },

                # Capital
                {
                    "claim_type": "capital_intensity",
                    "field": "monthly_burn",
                    "value": 120000,
                    "confidence": 0.85,
                    "polarity": "neutral",
                    "unit": "USD",
                },
                {
                    "claim_type": "capital_intensity",
                    "field": "runway_months",
                    "value": 18,
                    "confidence": 0.80,
                    "polarity": "neutral",
                },
                {
                    "claim_type": "use_of_proceeds",
                    "field": "allocation",
                    "value": "40% R&D, 35% Sales, 15% Regulatory, 10% G&A",
                    "confidence": 0.85,
                    "polarity": "neutral",
                },

                # Competitive
                {
                    "claim_type": "differentiation",
                    "field": "competitors",
                    "value": "Nuance (Microsoft), Isabel Healthcare, Infermedica",
                    "confidence": 0.85,
                    "polarity": "risk",
                },
                {
                    "claim_type": "execution_risk",
                    "field": "competitive_threat",
                    "value": "Microsoft Nuance has 500x resources; unclear differentiation path",
                    "confidence": 0.70,
                    "polarity": "risk",
                },

                # Exit
                {
                    "claim_type": "exit_logic",
                    "field": "potential_acquirers",
                    "value": ["Epic", "Cerner (Oracle)", "Veeva", "Teladoc"],
                    "confidence": 0.60,
                    "polarity": "supportive",
                },
                {
                    "claim_type": "exit_logic",
                    "field": "comparable_exits",
                    "value": "Buoy Health ($75M), K Health ($132M valuation)",
                    "confidence": 0.65,
                    "polarity": "neutral",
                },
            ],
        },
    }


def save_demo_data():
    """Save all demo data to JSON files."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save historical deals
    historical = create_historical_deals()
    for deal in historical:
        filename = DATA_DIR / f"{deal['company_id']}.json"
        with open(filename, "w") as f:
            json.dump(deal, f, indent=2)
        print(f"Saved: {filename}")

    # Save historical index
    index = {
        "historical_deals": [
            {
                "company_id": d["company_id"],
                "company_name": d["company_name"],
                "sector": d["sector"],
                "stage": d["stage"],
                "outcome": d["outcome"],
            }
            for d in historical
        ]
    }
    with open(DATA_DIR / "historical_index.json", "w") as f:
        json.dump(index, f, indent=2)

    # Save ambiguous deal
    ambiguous = create_ambiguous_deal()
    with open(DATA_DIR / f"{ambiguous['company_id']}.json", "w") as f:
        json.dump(ambiguous, f, indent=2)
    print(f"Saved: {DATA_DIR / ambiguous['company_id']}.json")

    # Save ambiguous deal summary
    with open(DATA_DIR / "demo_deal.json", "w") as f:
        json.dump({
            "company_id": ambiguous["company_id"],
            "company_name": ambiguous["company_name"],
            "sector": ambiguous["sector"],
            "stage": ambiguous["stage"],
            "description": ambiguous["description"],
        }, f, indent=2)

    print(f"\nDemo data saved to {DATA_DIR}")
    return DATA_DIR


def load_deal(company_id: str) -> Dict[str, Any]:
    """Load a deal by company ID."""
    filepath = DATA_DIR / f"{company_id}.json"
    if not filepath.exists():
        raise FileNotFoundError(f"Deal not found: {company_id}")
    with open(filepath) as f:
        return json.load(f)


def load_all_historical_deals() -> List[Dict[str, Any]]:
    """Load all historical deals."""
    index_path = DATA_DIR / "historical_index.json"
    if not index_path.exists():
        save_demo_data()

    with open(index_path) as f:
        index = json.load(f)

    deals = []
    for deal_info in index["historical_deals"]:
        deals.append(load_deal(deal_info["company_id"]))
    return deals


def get_demo_deal() -> Dict[str, Any]:
    """Get the ambiguous demo deal for live analysis."""
    demo_path = DATA_DIR / "healthbridge-2024.json"
    if not demo_path.exists():
        save_demo_data()
    return load_deal("healthbridge-2024")


if __name__ == "__main__":
    save_demo_data()
