"""
Report Renderers for JURIS-AGI.

Provides HTML, Markdown, and PDF rendering of decision reports.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional
import html

from .schema import (
    DecisionReport,
    ExecutiveSummary,
    InferredDecisionLogic,
    EvidenceBasis,
    DealEvaluation,
    CounterfactualAnalysis,
    UncertaintyLimitations,
    AuditMetadata,
    DecisionOutcome,
    Polarity,
    ConfidenceLevel,
)


class ReportRenderer(ABC):
    """Base class for report renderers."""

    @abstractmethod
    def render(self, report: DecisionReport) -> str:
        """Render the report to string format."""
        pass

    @abstractmethod
    def content_type(self) -> str:
        """Return the MIME content type."""
        pass

    @abstractmethod
    def file_extension(self) -> str:
        """Return the file extension."""
        pass


class HTMLRenderer(ReportRenderer):
    """Renders reports to HTML format."""

    def content_type(self) -> str:
        return "text/html"

    def file_extension(self) -> str:
        return "html"

    def render(self, report: DecisionReport) -> str:
        """Render the report to HTML."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(report.title)}</title>
    <style>
        {self._get_styles()}
    </style>
</head>
<body>
    <div class="container">
        <header class="report-header">
            <h1>{html.escape(report.title)}</h1>
            <div class="report-meta">
                Generated: {report.audit_metadata.generated_at.strftime('%Y-%m-%d %H:%M UTC')}
                | Report ID: {report.audit_metadata.report_id}
            </div>
        </header>

        <nav class="toc">
            <h2>Contents</h2>
            <ul>
                <li><a href="#executive-summary">Executive Summary</a></li>
                <li><a href="#decision-logic">Inferred Decision Logic</a></li>
                <li><a href="#evidence-basis">Evidence Basis</a></li>
                <li><a href="#deal-evaluation">Deal Evaluation</a></li>
                <li><a href="#counterfactual-analysis">Counterfactual & Sensitivity Analysis</a></li>
                <li><a href="#uncertainty">Uncertainty & Limitations</a></li>
                <li><a href="#audit-metadata">Audit Metadata</a></li>
            </ul>
        </nav>

        <main>
            {self._render_executive_summary(report.executive_summary)}
            {self._render_decision_logic(report.decision_logic)}
            {self._render_evidence_basis(report.evidence_basis)}
            {self._render_deal_evaluation(report.deal_evaluation)}
            {self._render_counterfactual_analysis(report.counterfactual_analysis)}
            {self._render_uncertainty(report.uncertainty_limitations)}
            {self._render_audit_metadata(report.audit_metadata)}
        </main>

        <footer class="report-footer">
            <div class="disclaimer">
                <strong>DISCLAIMER:</strong> This report is decision support only, not investment advice.
                All investment decisions remain the sole responsibility of the investment committee.
                Confidence scores reflect model uncertainty, not probability of investment success.
            </div>
            <div class="footer-meta">
                JURIS-AGI VC Decision Intelligence | Model Version: {report.audit_metadata.model_version}
            </div>
        </footer>
    </div>
</body>
</html>"""

    def _get_styles(self) -> str:
        return """
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            color: #1a1a1a;
            background: #f5f5f5;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        .report-header {
            background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        .report-header h1 { font-size: 28px; margin-bottom: 10px; }
        .report-meta { font-size: 14px; opacity: 0.9; }
        .toc {
            background: #f8f9fa;
            padding: 20px 40px;
            border-bottom: 1px solid #e0e0e0;
        }
        .toc h2 { font-size: 16px; margin-bottom: 10px; color: #666; }
        .toc ul { list-style: none; display: flex; flex-wrap: wrap; gap: 15px; }
        .toc a { color: #2d5a87; text-decoration: none; font-size: 14px; }
        .toc a:hover { text-decoration: underline; }
        main { padding: 40px; }
        section { margin-bottom: 40px; }
        section h2 {
            font-size: 22px;
            color: #1e3a5f;
            border-bottom: 2px solid #2d5a87;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        section h3 {
            font-size: 18px;
            color: #333;
            margin: 20px 0 10px;
        }
        .decision-box {
            display: flex;
            align-items: center;
            gap: 20px;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .decision-invest { background: #e8f5e9; border: 2px solid #4caf50; }
        .decision-pass { background: #ffebee; border: 2px solid #f44336; }
        .decision-defer { background: #fff8e1; border: 2px solid #ff9800; }
        .decision-badge {
            font-size: 24px;
            font-weight: bold;
            padding: 10px 20px;
            border-radius: 4px;
            color: white;
        }
        .decision-invest .decision-badge { background: #4caf50; }
        .decision-pass .decision-badge { background: #f44336; }
        .decision-defer .decision-badge { background: #ff9800; }
        .confidence-bar {
            flex: 1;
            background: #e0e0e0;
            height: 20px;
            border-radius: 10px;
            overflow: hidden;
        }
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #2d5a87 0%, #4caf50 100%);
            transition: width 0.3s ease;
        }
        .key-points { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }
        .key-points-box {
            padding: 15px;
            border-radius: 8px;
        }
        .strengths { background: #e8f5e9; border-left: 4px solid #4caf50; }
        .risks { background: #ffebee; border-left: 4px solid #f44336; }
        .key-points-box h4 { font-size: 14px; color: #666; margin-bottom: 10px; }
        .key-points-box ul { margin-left: 20px; }
        .key-points-box li { font-size: 14px; margin-bottom: 5px; }
        .evidence-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
            margin: 15px 0;
        }
        .evidence-table th, .evidence-table td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }
        .evidence-table th { background: #f5f5f5; font-weight: 600; }
        .polarity-supportive { color: #2e7d32; }
        .polarity-risk { color: #c62828; }
        .polarity-neutral { color: #666; }
        .stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }
        .stat-box {
            text-align: center;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        .stat-value { font-size: 28px; font-weight: bold; color: #1e3a5f; }
        .stat-label { font-size: 12px; color: #666; margin-top: 5px; }
        .rule-eval {
            padding: 15px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            margin-bottom: 10px;
        }
        .rule-header { display: flex; justify-content: space-between; align-items: center; }
        .rule-name { font-weight: 600; }
        .rule-result {
            padding: 4px 12px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
        }
        .rule-pass { background: #e8f5e9; color: #2e7d32; }
        .rule-fail { background: #ffebee; color: #c62828; }
        .rule-partial { background: #fff8e1; color: #f57c00; }
        .sensitivity-item {
            padding: 15px;
            border-left: 4px solid #ff9800;
            background: #fff8e1;
            margin-bottom: 10px;
            border-radius: 0 8px 8px 0;
        }
        .sensitivity-header { font-weight: 600; margin-bottom: 5px; }
        .sensitivity-explanation { font-size: 14px; color: #666; }
        .uncertainty-meter {
            display: flex;
            gap: 20px;
            margin: 20px 0;
        }
        .uncertainty-item {
            flex: 1;
            text-align: center;
            padding: 20px;
            border-radius: 8px;
            background: #f8f9fa;
        }
        .uncertainty-label { font-size: 12px; color: #666; margin-bottom: 5px; }
        .uncertainty-value { font-size: 24px; font-weight: bold; color: #1e3a5f; }
        .data-gaps {
            background: #fff3e0;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
        }
        .data-gaps h4 { color: #e65100; margin-bottom: 10px; }
        .data-gaps ul { margin-left: 20px; }
        .recommendations {
            background: #e3f2fd;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
        }
        .recommendations h4 { color: #1565c0; margin-bottom: 10px; }
        .metadata-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
        .metadata-item { font-size: 14px; padding: 8px 0; border-bottom: 1px solid #f0f0f0; }
        .metadata-label { color: #666; }
        .metadata-value { font-weight: 500; }
        .report-footer {
            background: #1e3a5f;
            color: white;
            padding: 30px 40px;
        }
        .disclaimer {
            background: rgba(255,255,255,0.1);
            padding: 15px;
            border-radius: 8px;
            font-size: 13px;
            margin-bottom: 15px;
        }
        .footer-meta { font-size: 12px; opacity: 0.8; text-align: center; }
        @media print {
            .container { box-shadow: none; }
            .toc { break-after: page; }
            section { break-inside: avoid; }
        }
        """

    def _render_executive_summary(self, summary: ExecutiveSummary) -> str:
        decision_class = f"decision-{summary.decision.value}"
        return f"""
        <section id="executive-summary">
            <h2>1. Executive Summary</h2>

            <div class="decision-box {decision_class}">
                <span class="decision-badge">{summary.decision.value.upper()}</span>
                <div style="flex: 1;">
                    <div style="font-size: 14px; color: #666; margin-bottom: 5px;">
                        Confidence: {summary.confidence:.0%} ({summary.confidence_level.value})
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {summary.confidence * 100}%;"></div>
                    </div>
                </div>
            </div>

            <p style="font-size: 18px; margin-bottom: 20px; font-weight: 500;">
                {html.escape(summary.one_line_summary)}
            </p>

            <div class="key-points">
                <div class="key-points-box strengths">
                    <h4>Key Strengths</h4>
                    <ul>
                        {''.join(f'<li>{html.escape(s)}</li>' for s in summary.key_strengths[:5])}
                    </ul>
                </div>
                <div class="key-points-box risks">
                    <h4>Key Risks</h4>
                    <ul>
                        {''.join(f'<li>{html.escape(r)}</li>' for r in summary.key_risks[:5])}
                    </ul>
                </div>
            </div>

            {self._render_critical_uncertainties(summary.critical_uncertainties)}

            <div style="background: #f5f5f5; padding: 15px; border-radius: 8px; margin-top: 20px;">
                <p style="font-size: 14px; line-height: 1.7;">
                    {html.escape(summary.recommendation_text)}
                </p>
            </div>
        </section>
        """

    def _render_critical_uncertainties(self, uncertainties: list) -> str:
        if not uncertainties:
            return ""
        return f"""
        <div style="margin: 20px 0;">
            <h4 style="font-size: 14px; color: #666; margin-bottom: 10px;">Critical Uncertainties</h4>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                {''.join(f'<span style="background: #fff8e1; padding: 5px 10px; border-radius: 4px; font-size: 13px; border: 1px solid #ff9800;">{html.escape(u)}</span>' for u in uncertainties)}
            </div>
        </div>
        """

    def _render_decision_logic(self, logic: InferredDecisionLogic) -> str:
        rule_html = ""
        for rule in logic.rule_evaluations:
            result_class = f"rule-{rule.result}"
            rule_html += f"""
            <div class="rule-eval">
                <div class="rule-header">
                    <span class="rule-name">{html.escape(rule.rule_name.replace('_', ' ').title())}</span>
                    <span class="rule-result {result_class}">{rule.result.upper()}</span>
                </div>
                <p style="font-size: 13px; color: #666; margin-top: 5px;">{html.escape(rule.rule_description)}</p>
                <p style="font-size: 12px; color: #888; margin-top: 5px;">
                    Evidence: {', '.join(rule.contributing_evidence[:3])}
                </p>
            </div>
            """

        return f"""
        <section id="decision-logic">
            <h2>2. Inferred Decision Logic</h2>

            <div class="stat-grid">
                <div class="stat-box">
                    <div class="stat-value">{logic.net_signal_score:+.2f}</div>
                    <div class="stat-label">Net Signal Score</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{len(logic.primary_factors)}</div>
                    <div class="stat-label">Primary Factors</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{len(logic.rule_evaluations)}</div>
                    <div class="stat-label">Rules Evaluated</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{len([r for r in logic.rule_evaluations if r.result == 'pass'])}</div>
                    <div class="stat-label">Rules Passed</div>
                </div>
            </div>

            <h3>Decision Threshold</h3>
            <p style="font-size: 14px; color: #666; margin-bottom: 20px;">
                {html.escape(logic.decision_threshold_description)}
            </p>

            <h3>Rule Evaluations</h3>
            {rule_html}

            <h3>Decision Rationale</h3>
            <p style="font-size: 14px; background: #f5f5f5; padding: 15px; border-radius: 8px;">
                {html.escape(logic.decision_rationale)}
            </p>
        </section>
        """

    def _render_evidence_basis(self, evidence: EvidenceBasis) -> str:
        def render_claims_table(claims, title):
            if not claims:
                return f"<p style='color: #666; font-style: italic;'>No {title.lower()} claims.</p>"
            rows = ""
            for c in claims[:10]:
                polarity_class = f"polarity-{c.polarity.value}"
                rows += f"""
                <tr>
                    <td>{html.escape(c.claim_type)}</td>
                    <td>{html.escape(c.field)}</td>
                    <td>{html.escape(str(c.value)[:50])}</td>
                    <td>{c.confidence:.0%}</td>
                    <td class="{polarity_class}">{c.polarity.value}</td>
                </tr>
                """
            return f"""
            <table class="evidence-table">
                <thead>
                    <tr>
                        <th>Type</th>
                        <th>Field</th>
                        <th>Value</th>
                        <th>Confidence</th>
                        <th>Polarity</th>
                    </tr>
                </thead>
                <tbody>{rows}</tbody>
            </table>
            """

        return f"""
        <section id="evidence-basis">
            <h2>3. Evidence Basis</h2>

            <div class="stat-grid">
                <div class="stat-box">
                    <div class="stat-value">{evidence.total_claims}</div>
                    <div class="stat-label">Total Claims</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value" style="color: #2e7d32;">{len(evidence.supportive_claims)}</div>
                    <div class="stat-label">Supportive</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value" style="color: #c62828;">{len(evidence.risk_claims)}</div>
                    <div class="stat-label">Risk</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{evidence.average_confidence:.0%}</div>
                    <div class="stat-label">Avg Confidence</div>
                </div>
            </div>

            <h3>Supportive Evidence</h3>
            {render_claims_table(evidence.supportive_claims, "Supportive")}

            <h3>Risk Evidence</h3>
            {render_claims_table(evidence.risk_claims, "Risk")}

            <h3>Neutral Evidence</h3>
            {render_claims_table(evidence.neutral_claims, "Neutral")}

            {self._render_low_confidence_warning(evidence.low_confidence_claims)}
        </section>
        """

    def _render_low_confidence_warning(self, claims: list) -> str:
        if not claims:
            return ""
        return f"""
        <div class="data-gaps">
            <h4>Low Confidence Claims ({len(claims)})</h4>
            <ul>
                {''.join(f'<li>{html.escape(c.claim_type)}.{html.escape(c.field)} ({c.confidence:.0%})</li>' for c in claims[:5])}
            </ul>
        </div>
        """

    def _render_deal_evaluation(self, deal: DealEvaluation) -> str:
        summaries_html = ""
        summaries = [
            ("Traction", deal.traction_summary),
            ("Team", deal.team_summary),
            ("Market", deal.market_summary),
            ("Risks", deal.risk_summary),
        ]
        for label, summary in summaries:
            if summary:
                summaries_html += f"""
                <div style="margin-bottom: 15px;">
                    <h4 style="font-size: 14px; color: #666; margin-bottom: 5px;">{label}</h4>
                    <p style="font-size: 14px; background: #f8f9fa; padding: 10px; border-radius: 4px;">
                        {html.escape(summary)}
                    </p>
                </div>
                """

        return f"""
        <section id="deal-evaluation">
            <h2>4. Deal Evaluation</h2>

            <div style="background: #f5f5f5; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                    <div>
                        <span style="color: #666; font-size: 12px;">Company</span>
                        <p style="font-weight: 600; font-size: 18px;">{html.escape(deal.company_name or deal.company_id)}</p>
                    </div>
                    <div>
                        <span style="color: #666; font-size: 12px;">Company ID</span>
                        <p style="font-family: monospace;">{html.escape(deal.company_id)}</p>
                    </div>
                    <div>
                        <span style="color: #666; font-size: 12px;">Sector</span>
                        <p>{html.escape(deal.sector or 'Not specified')}</p>
                    </div>
                    <div>
                        <span style="color: #666; font-size: 12px;">Stage</span>
                        <p>{html.escape(deal.stage or 'Not specified')}</p>
                    </div>
                </div>
            </div>

            <h3>Category Summaries</h3>
            {summaries_html if summaries_html else '<p style="color: #666;">No category summaries available.</p>'}
        </section>
        """

    def _render_counterfactual_analysis(self, cf: CounterfactualAnalysis) -> str:
        sensitivity_html = ""
        for item in cf.critical_claims[:5]:
            sensitivity_html += f"""
            <div class="sensitivity-item">
                <div class="sensitivity-header">
                    {html.escape(item.claim_type)}.{html.escape(item.claim_field)}
                    <span style="float: right; font-size: 12px; color: #666;">
                        Criticality: {item.criticality_score:.0%}
                    </span>
                </div>
                <div class="sensitivity-explanation">{html.escape(item.explanation)}</div>
            </div>
            """

        flip_html = ""
        for flip in cf.decision_flip_scenarios[:3]:
            flip_html += f"""
            <div style="padding: 15px; border: 1px solid #e0e0e0; border-radius: 8px; margin-bottom: 10px;">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                    <span style="background: #e0e0e0; padding: 4px 8px; border-radius: 4px; font-size: 12px;">
                        {flip.original_decision.value.upper()}
                    </span>
                    <span>→</span>
                    <span style="background: #ff9800; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px;">
                        {flip.new_decision.value.upper()}
                    </span>
                </div>
                <p style="font-size: 13px; color: #666;">{html.escape(flip.explanation)}</p>
            </div>
            """

        return f"""
        <section id="counterfactual-analysis">
            <h2>5. Counterfactual & Sensitivity Analysis</h2>

            <div class="stat-grid">
                <div class="stat-box">
                    <div class="stat-value">{cf.robustness_score:.0%}</div>
                    <div class="stat-label">Robustness Score</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{cf.stability_margin:.0%}</div>
                    <div class="stat-label">Stability Margin</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{cf.total_counterfactuals_tested}</div>
                    <div class="stat-label">CFs Tested</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{cf.flips_found}</div>
                    <div class="stat-label">Flips Found</div>
                </div>
            </div>

            <div style="background: #f5f5f5; padding: 15px; border-radius: 8px; margin: 20px 0;">
                <p style="font-size: 14px;">{html.escape(cf.robustness_interpretation)}</p>
            </div>

            <h3>Critical Claims</h3>
            <p style="font-size: 14px; color: #666; margin-bottom: 15px;">
                Claims most likely to flip the decision if changed:
            </p>
            {sensitivity_html if sensitivity_html else '<p style="color: #666;">No critical claims identified.</p>'}

            <h3>Decision Flip Scenarios</h3>
            {flip_html if flip_html else '<p style="color: #666;">No decision flip scenarios found.</p>'}
        </section>
        """

    def _render_uncertainty(self, uncertainty: UncertaintyLimitations) -> str:
        gaps_html = ""
        if uncertainty.data_gaps:
            gaps_html = f"""
            <div class="data-gaps">
                <h4>Data Gaps</h4>
                <ul>
                    {''.join(f'<li>{html.escape(gap)}</li>' for gap in uncertainty.data_gaps[:5])}
                </ul>
            </div>
            """

        recommendations_html = ""
        if uncertainty.recommendations_for_diligence:
            recommendations_html = f"""
            <div class="recommendations">
                <h4>Recommendations for Due Diligence</h4>
                <ul>
                    {''.join(f'<li>{html.escape(rec)}</li>' for rec in uncertainty.recommendations_for_diligence)}
                </ul>
            </div>
            """

        return f"""
        <section id="uncertainty">
            <h2>6. Uncertainty & Limitations</h2>

            <div class="uncertainty-meter">
                <div class="uncertainty-item">
                    <div class="uncertainty-label">Epistemic Uncertainty</div>
                    <div class="uncertainty-value">{uncertainty.epistemic_uncertainty:.0%}</div>
                    <div style="font-size: 11px; color: #888;">Reducible with more data</div>
                </div>
                <div class="uncertainty-item">
                    <div class="uncertainty-label">Aleatoric Uncertainty</div>
                    <div class="uncertainty-value">{uncertainty.aleatoric_uncertainty:.0%}</div>
                    <div style="font-size: 11px; color: #888;">Inherent randomness</div>
                </div>
                <div class="uncertainty-item">
                    <div class="uncertainty-label">Confidence Level</div>
                    <div class="uncertainty-value">{uncertainty.confidence_level.value.upper()}</div>
                    <div style="font-size: 11px; color: #888;">Overall assessment</div>
                </div>
            </div>

            {gaps_html}
            {recommendations_html}

            <h3>Key Assumptions</h3>
            <ul style="margin-left: 20px; margin-bottom: 20px;">
                {''.join(f'<li style="font-size: 14px; margin-bottom: 5px;">{html.escape(a)}</li>' for a in uncertainty.assumptions_made)}
            </ul>

            <h3>Limitations</h3>
            <ul style="margin-left: 20px;">
                {''.join(f'<li style="font-size: 14px; margin-bottom: 5px;">{html.escape(l)}</li>' for l in uncertainty.limitations)}
            </ul>
        </section>
        """

    def _render_audit_metadata(self, meta: AuditMetadata) -> str:
        return f"""
        <section id="audit-metadata">
            <h2>7. Audit Metadata</h2>

            <div class="metadata-grid">
                <div class="metadata-item">
                    <span class="metadata-label">Report ID:</span>
                    <span class="metadata-value" style="font-family: monospace;">{html.escape(meta.report_id)}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Generated At:</span>
                    <span class="metadata-value">{meta.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Model Version:</span>
                    <span class="metadata-value">{html.escape(meta.model_version)}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Trace ID:</span>
                    <span class="metadata-value" style="font-family: monospace;">{html.escape(meta.trace_id or 'N/A')}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Analyst ID:</span>
                    <span class="metadata-value">{html.escape(meta.analyst_id or 'N/A')}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Claims Analyzed:</span>
                    <span class="metadata-value">{meta.num_claims_analyzed}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Counterfactuals Tested:</span>
                    <span class="metadata-value">{meta.num_counterfactuals_tested}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Reproducibility Seed:</span>
                    <span class="metadata-value">{meta.reproducibility_seed or 'N/A'}</span>
                </div>
            </div>
        </section>
        """


class MarkdownRenderer(ReportRenderer):
    """Renders reports to Markdown format."""

    def content_type(self) -> str:
        return "text/markdown"

    def file_extension(self) -> str:
        return "md"

    def render(self, report: DecisionReport) -> str:
        """Render the report to Markdown."""
        sections = [
            self._render_header(report),
            self._render_executive_summary(report.executive_summary),
            self._render_decision_logic(report.decision_logic),
            self._render_evidence_basis(report.evidence_basis),
            self._render_deal_evaluation(report.deal_evaluation),
            self._render_counterfactual_analysis(report.counterfactual_analysis),
            self._render_uncertainty(report.uncertainty_limitations),
            self._render_audit_metadata(report.audit_metadata),
            self._render_disclaimer(),
        ]
        return "\n\n".join(sections)

    def _render_header(self, report: DecisionReport) -> str:
        return f"""# {report.title}

**Generated:** {report.audit_metadata.generated_at.strftime('%Y-%m-%d %H:%M UTC')}
**Report ID:** `{report.audit_metadata.report_id}`

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Inferred Decision Logic](#2-inferred-decision-logic)
3. [Evidence Basis](#3-evidence-basis)
4. [Deal Evaluation](#4-deal-evaluation)
5. [Counterfactual & Sensitivity Analysis](#5-counterfactual--sensitivity-analysis)
6. [Uncertainty & Limitations](#6-uncertainty--limitations)
7. [Audit Metadata](#7-audit-metadata)"""

    def _render_executive_summary(self, summary: ExecutiveSummary) -> str:
        strengths = "\n".join(f"- {s}" for s in summary.key_strengths[:5])
        risks = "\n".join(f"- {r}" for r in summary.key_risks[:5])
        uncertainties = ", ".join(summary.critical_uncertainties[:3])

        return f"""## 1. Executive Summary

### Decision: **{summary.decision.value.upper()}**

**Confidence:** {summary.confidence:.0%} ({summary.confidence_level.value})

> {summary.one_line_summary}

### Key Strengths
{strengths}

### Key Risks
{risks}

### Critical Uncertainties
{uncertainties if uncertainties else "None identified"}

### Recommendation

{summary.recommendation_text}"""

    def _render_decision_logic(self, logic: InferredDecisionLogic) -> str:
        rules = "\n".join(
            f"| {r.rule_name.replace('_', ' ').title()} | {r.result.upper()} | {r.weight:.0%} |"
            for r in logic.rule_evaluations
        )

        return f"""## 2. Inferred Decision Logic

**Net Signal Score:** {logic.net_signal_score:+.2f}

### Decision Threshold
{logic.decision_threshold_description}

### Rule Evaluations

| Rule | Result | Weight |
|------|--------|--------|
{rules}

### Decision Rationale
{logic.decision_rationale}"""

    def _render_evidence_basis(self, evidence: EvidenceBasis) -> str:
        supportive = "\n".join(
            f"| {c.claim_type} | {c.field} | {str(c.value)[:30]} | {c.confidence:.0%} |"
            for c in evidence.supportive_claims[:5]
        )
        risks = "\n".join(
            f"| {c.claim_type} | {c.field} | {str(c.value)[:30]} | {c.confidence:.0%} |"
            for c in evidence.risk_claims[:5]
        )

        return f"""## 3. Evidence Basis

- **Total Claims:** {evidence.total_claims}
- **Supportive:** {len(evidence.supportive_claims)}
- **Risk:** {len(evidence.risk_claims)}
- **Neutral:** {len(evidence.neutral_claims)}
- **Average Confidence:** {evidence.average_confidence:.0%}

### Supportive Evidence

| Type | Field | Value | Confidence |
|------|-------|-------|------------|
{supportive if supportive else "| - | - | No supportive claims | - |"}

### Risk Evidence

| Type | Field | Value | Confidence |
|------|-------|-------|------------|
{risks if risks else "| - | - | No risk claims | - |"}"""

    def _render_deal_evaluation(self, deal: DealEvaluation) -> str:
        return f"""## 4. Deal Evaluation

- **Company:** {deal.company_name or deal.company_id}
- **Company ID:** `{deal.company_id}`
- **Sector:** {deal.sector or 'Not specified'}
- **Stage:** {deal.stage or 'Not specified'}

### Traction
{deal.traction_summary or 'Not available'}

### Team
{deal.team_summary or 'Not available'}

### Market
{deal.market_summary or 'Not available'}

### Risks
{deal.risk_summary or 'Not available'}"""

    def _render_counterfactual_analysis(self, cf: CounterfactualAnalysis) -> str:
        critical = "\n".join(
            f"- **{c.claim_type}.{c.claim_field}** (Criticality: {c.criticality_score:.0%}): {c.explanation}"
            for c in cf.critical_claims[:5]
        )

        return f"""## 5. Counterfactual & Sensitivity Analysis

- **Robustness Score:** {cf.robustness_score:.0%}
- **Stability Margin:** {cf.stability_margin:.0%}
- **Counterfactuals Tested:** {cf.total_counterfactuals_tested}
- **Flips Found:** {cf.flips_found}

### Interpretation
{cf.robustness_interpretation}

### Critical Claims
{critical if critical else "No critical claims identified."}"""

    def _render_uncertainty(self, uncertainty: UncertaintyLimitations) -> str:
        gaps = "\n".join(f"- {g}" for g in uncertainty.data_gaps[:5])
        recs = "\n".join(f"- {r}" for r in uncertainty.recommendations_for_diligence)
        limitations = "\n".join(f"- {l}" for l in uncertainty.limitations)

        return f"""## 6. Uncertainty & Limitations

- **Epistemic Uncertainty:** {uncertainty.epistemic_uncertainty:.0%} (reducible)
- **Aleatoric Uncertainty:** {uncertainty.aleatoric_uncertainty:.0%} (irreducible)
- **Confidence Level:** {uncertainty.confidence_level.value.upper()}

### Data Gaps
{gaps if gaps else "No significant data gaps identified."}

### Recommendations for Due Diligence
{recs}

### Limitations
{limitations}"""

    def _render_audit_metadata(self, meta: AuditMetadata) -> str:
        return f"""## 7. Audit Metadata

| Field | Value |
|-------|-------|
| Report ID | `{meta.report_id}` |
| Generated At | {meta.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')} |
| Model Version | {meta.model_version} |
| Trace ID | `{meta.trace_id or 'N/A'}` |
| Analyst ID | {meta.analyst_id or 'N/A'} |
| Claims Analyzed | {meta.num_claims_analyzed} |
| Counterfactuals Tested | {meta.num_counterfactuals_tested} |
| Reproducibility Seed | {meta.reproducibility_seed or 'N/A'} |"""

    def _render_disclaimer(self) -> str:
        return """---

## Disclaimer

**IMPORTANT:** This report is decision support only, not investment advice.
All investment decisions remain the sole responsibility of the investment committee.
Confidence scores reflect model uncertainty, not probability of investment success.

---

*Generated by JURIS-AGI VC Decision Intelligence*"""


class PDFRenderer(ReportRenderer):
    """
    Renders reports to PDF format.

    Uses HTML-to-PDF conversion for simplicity.
    For production, consider using reportlab or weasyprint.
    """

    def content_type(self) -> str:
        return "application/pdf"

    def file_extension(self) -> str:
        return "pdf"

    def render(self, report: DecisionReport) -> bytes:
        """
        Render the report to PDF.

        Returns bytes (PDF content) instead of string.
        Falls back to HTML if PDF libraries not available.
        """
        # Try weasyprint first
        try:
            from weasyprint import HTML
            html_content = HTMLRenderer().render(report)
            return HTML(string=html_content).write_pdf()
        except ImportError:
            pass

        # Try pdfkit
        try:
            import pdfkit
            html_content = HTMLRenderer().render(report)
            return pdfkit.from_string(html_content, False)
        except ImportError:
            pass

        # Fallback: return HTML with PDF-like styling note
        html_content = HTMLRenderer().render(report)
        return html_content.encode('utf-8')


# Convenience functions
def render_html(report: DecisionReport) -> str:
    """Render report to HTML."""
    return HTMLRenderer().render(report)


def render_markdown(report: DecisionReport) -> str:
    """Render report to Markdown."""
    return MarkdownRenderer().render(report)


def render_pdf(report: DecisionReport) -> bytes:
    """Render report to PDF."""
    return PDFRenderer().render(report)
