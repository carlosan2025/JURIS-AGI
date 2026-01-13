"""
Generate example decision reports for demo data.

Creates HTML and Markdown reports for the demo deals.
"""

import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.juris_agi.report import generate_report, render_html, render_markdown
from src.juris_agi.vc.decision_analysis import DecisionAnalyzer, DecisionOutcome
from src.juris_agi.vc.trace import VCDecisionTrace
from src.juris_agi.evidence.schema import EvidenceGraph, Claim, Polarity
from src.juris_agi.evidence.ontology import ClaimType
from src.juris_agi.config import DEFAULT_SEED

from demo.seed_data import get_demo_deal, load_all_historical_deals


REPORTS_DIR = Path(__file__).parent / "reports"


def parse_evidence_graph(deal_data: dict) -> EvidenceGraph:
    """Convert deal data to internal EvidenceGraph format."""
    graph_data = deal_data["evidence_graph"]
    company_id = graph_data["company_id"]

    claims = []
    for c in graph_data.get("claims", []):
        try:
            claim_type = ClaimType(c.get("claim_type", "company_identity"))
        except ValueError:
            claim_type = ClaimType.COMPANY_IDENTITY

        try:
            polarity = Polarity(c.get("polarity", "neutral"))
        except ValueError:
            polarity = Polarity.NEUTRAL

        claims.append(Claim(
            claim_type=claim_type,
            field=c.get("field", ""),
            value=c.get("value"),
            confidence=c.get("confidence", 0.5),
            polarity=polarity,
        ))

    return EvidenceGraph(company_id=company_id, claims=claims)


def create_decision_function():
    """Create the decision function for analysis."""
    def decision_fn(g: EvidenceGraph):
        supportive = sum(1 for c in g.claims if c.polarity == Polarity.SUPPORTIVE)
        risk = sum(1 for c in g.claims if c.polarity == Polarity.RISK)
        total = len(g.claims) or 1

        support_ratio = supportive / total
        risk_ratio = risk / total

        if support_ratio > 0.5 and risk_ratio < 0.3:
            decision = DecisionOutcome.INVEST
            confidence = 0.75 + support_ratio * 0.2
        elif risk_ratio > 0.4:
            decision = DecisionOutcome.PASS
            confidence = 0.6 + risk_ratio * 0.3
        else:
            decision = DecisionOutcome.DEFER
            confidence = 0.5

        return decision, min(confidence, 0.95)

    return decision_fn


def generate_deal_report(deal_data: dict, output_dir: Path):
    """Generate reports for a single deal."""
    company_id = deal_data["company_id"]
    company_name = deal_data.get("company_name", company_id)

    print(f"\nGenerating report for: {company_name}")

    # Parse evidence graph
    graph = parse_evidence_graph(deal_data)

    # Run analysis
    decision_fn = create_decision_function()
    analyzer = DecisionAnalyzer(
        decision_fn=decision_fn,
        seed=DEFAULT_SEED,
        num_counterfactuals=20,
    )
    result = analyzer.analyze(graph)

    # Create trace
    trace = VCDecisionTrace.from_analysis_result(company_id, result)
    trace_dict = trace.to_dict()

    # Generate report
    report = generate_report(
        evidence_graph=deal_data["evidence_graph"],
        trace=trace_dict,
        final_decision=result.decision.value,
        seed=DEFAULT_SEED,
    )

    # Render HTML
    html_content = render_html(report)
    html_path = output_dir / f"{company_id}-report.html"
    with open(html_path, "w") as f:
        f.write(html_content)
    print(f"  HTML: {html_path}")

    # Render Markdown
    md_content = render_markdown(report)
    md_path = output_dir / f"{company_id}-report.md"
    with open(md_path, "w") as f:
        f.write(md_content)
    print(f"  Markdown: {md_path}")

    # Also save trace for reference
    trace_path = output_dir / f"{company_id}-trace.json"
    with open(trace_path, "w") as f:
        json.dump(trace_dict, f, indent=2, default=str)
    print(f"  Trace: {trace_path}")

    return result.decision.value, result.confidence


def main():
    """Generate example reports for all demo deals."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("JURIS-AGI Example Report Generator")
    print("=" * 60)

    # Generate for demo deal (HealthBridge AI)
    print("\n--- Demo Deal (Ambiguous) ---")
    demo_deal = get_demo_deal()
    decision, confidence = generate_deal_report(demo_deal, REPORTS_DIR)
    print(f"  Decision: {decision.upper()} ({confidence:.0%} confidence)")

    # Generate for historical deals
    print("\n--- Historical Deals ---")
    historical_deals = load_all_historical_deals()

    for deal in historical_deals:
        expected = deal.get("outcome", "unknown")
        decision, confidence = generate_deal_report(deal, REPORTS_DIR)
        match = "✓" if decision == expected else "✗"
        print(f"  Decision: {decision.upper()} (expected: {expected.upper()}) {match}")

    print("\n" + "=" * 60)
    print(f"Reports saved to: {REPORTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
