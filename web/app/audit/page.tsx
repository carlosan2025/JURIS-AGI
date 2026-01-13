"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import {
  AlertCircle,
  CheckCircle2,
  XCircle,
  Clock,
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  Minus,
  ArrowLeft,
  FileText,
  Brain,
  Scale,
  RefreshCw,
  Download,
  ExternalLink,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import type { AnalysisResult, CriticalClaim, CounterfactualExplanation } from "@/types/analysis";
import type { EvidenceGraph } from "@/types/evidence";
import { Disclaimer } from "@/components/Disclaimer";
import { simulateReportHTML } from "@/lib/api";

export default function AuditPage() {
  const router = useRouter();
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [evidenceGraph, setEvidenceGraph] = useState<EvidenceGraph | null>(null);
  const [showReportModal, setShowReportModal] = useState(false);
  const [reportHTML, setReportHTML] = useState<string | null>(null);

  useEffect(() => {
    const stored = sessionStorage.getItem("analysisResult");
    if (stored) {
      setResult(JSON.parse(stored));
    }
    const storedGraph = sessionStorage.getItem("evidenceGraph");
    if (storedGraph) {
      setEvidenceGraph(JSON.parse(storedGraph));
    }
  }, []);

  const generateReport = () => {
    if (result && evidenceGraph) {
      const html = simulateReportHTML(evidenceGraph, result);
      setReportHTML(html);
      setShowReportModal(true);
    }
  };

  const downloadReport = (format: "html" | "pdf") => {
    if (!reportHTML) return;

    const blob = new Blob([reportHTML], { type: "text/html" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `decision-report-${result?.company_id || "report"}.${format === "pdf" ? "html" : format}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const openReportInNewTab = () => {
    if (!reportHTML) return;
    const blob = new Blob([reportHTML], { type: "text/html" });
    const url = URL.createObjectURL(blob);
    window.open(url, "_blank");
  };

  if (!result) {
    return (
      <div className="flex flex-col items-center justify-center py-20">
        <AlertCircle className="h-12 w-12 text-muted-foreground mb-4" />
        <h2 className="text-xl font-semibold mb-2">No Analysis Result</h2>
        <p className="text-muted-foreground mb-4">
          Please run an analysis first.
        </p>
        <Button onClick={() => router.push("/analyze")}>Go to Analysis</Button>
      </div>
    );
  }

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Audit Trail</h1>
          <p className="text-muted-foreground mt-1">
            Complete decision reasoning and counterfactual analysis
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button onClick={generateReport} disabled={!evidenceGraph}>
            <FileText className="mr-2 h-4 w-4" />
            View Decision Report
          </Button>
          <Button variant="outline" onClick={() => router.push("/")}>
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Workspace
          </Button>
        </div>
      </div>

      {/* Report Modal */}
      {showReportModal && reportHTML && (
        <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] flex flex-col">
            <div className="flex items-center justify-between p-4 border-b">
              <h2 className="text-lg font-semibold">Decision Report</h2>
              <div className="flex items-center gap-2">
                <Button variant="outline" size="sm" onClick={openReportInNewTab}>
                  <ExternalLink className="h-4 w-4 mr-1" />
                  Open in Tab
                </Button>
                <Button variant="outline" size="sm" onClick={() => downloadReport("html")}>
                  <Download className="h-4 w-4 mr-1" />
                  Download HTML
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowReportModal(false)}
                >
                  Close
                </Button>
              </div>
            </div>
            <div className="flex-1 overflow-auto">
              <iframe
                srcDoc={reportHTML}
                className="w-full h-full min-h-[600px]"
                title="Decision Report"
              />
            </div>
          </div>
        </div>
      )}

      {/* Decision Summary */}
      <Card>
        <CardContent className="py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div
                className={`p-3 rounded-full ${
                  result.decision === "invest"
                    ? "bg-green-100"
                    : result.decision === "pass"
                    ? "bg-red-100"
                    : "bg-yellow-100"
                }`}
              >
                {result.decision === "invest" ? (
                  <CheckCircle2 className="h-8 w-8 text-green-600" />
                ) : result.decision === "pass" ? (
                  <XCircle className="h-8 w-8 text-red-600" />
                ) : (
                  <Clock className="h-8 w-8 text-yellow-600" />
                )}
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Final Decision</p>
                <div className="flex items-center gap-3 mt-1">
                  <Badge
                    variant={
                      result.decision === "invest"
                        ? "invest"
                        : result.decision === "pass"
                        ? "pass"
                        : "defer"
                    }
                    className="text-xl px-4 py-1"
                  >
                    {result.decision.toUpperCase()}
                  </Badge>
                  <span className="text-lg">
                    {(result.confidence * 100).toFixed(0)}% confidence
                  </span>
                </div>
              </div>
            </div>
            <div className="text-right">
              <p className="text-sm text-muted-foreground">Robustness Score</p>
              <p className="text-3xl font-bold mt-1">
                {(result.robustness.overall_score * 100).toFixed(0)}%
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Tabs */}
      <Tabs defaultValue="rules" className="space-y-4">
        <TabsList className="grid grid-cols-4 w-full">
          <TabsTrigger value="rules" className="flex items-center gap-2">
            <Scale className="h-4 w-4" />
            Decision Rules
          </TabsTrigger>
          <TabsTrigger value="uncertainty" className="flex items-center gap-2">
            <Brain className="h-4 w-4" />
            Uncertainty
          </TabsTrigger>
          <TabsTrigger value="counterfactuals" className="flex items-center gap-2">
            <RefreshCw className="h-4 w-4" />
            Counterfactuals
          </TabsTrigger>
          <TabsTrigger value="timeline" className="flex items-center gap-2">
            <Clock className="h-4 w-4" />
            Timeline
          </TabsTrigger>
        </TabsList>

        {/* Decision Rules Tab */}
        <TabsContent value="rules" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Inferred Decision Rules</CardTitle>
              <CardDescription>
                The key factors that influenced this investment decision
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Decision Logic */}
              <div className="p-4 rounded-lg bg-muted/30 border">
                <h4 className="font-medium mb-3">Decision Logic</h4>
                <div className="space-y-2 text-sm">
                  <p>The decision engine evaluated the evidence graph using the following criteria:</p>
                  <ul className="list-disc list-inside space-y-1 ml-2 mt-2">
                    <li>
                      <span className="font-medium">Net Signal Score:</span> Weighted sum of supportive vs risk claims
                    </li>
                    <li>
                      <span className="font-medium">Confidence Threshold:</span> Claims below 0.5 confidence are down-weighted
                    </li>
                    <li>
                      <span className="font-medium">Critical Claim Impact:</span> High-criticality claims have outsized influence
                    </li>
                    <li>
                      <span className="font-medium">Robustness Check:</span> Decision stability under perturbations
                    </li>
                  </ul>
                </div>
              </div>

              {/* Critical Claims */}
              <div>
                <h4 className="font-medium mb-3">Critical Claims ({result.critical_claims.length})</h4>
                <div className="space-y-3">
                  {result.critical_claims.map((cc, i) => (
                    <CriticalClaimCard key={i} claim={cc} index={i} />
                  ))}
                </div>
              </div>

              {/* Decision Threshold */}
              <div className="p-4 rounded-lg border">
                <h4 className="font-medium mb-2">Decision Threshold</h4>
                <div className="flex items-center gap-4">
                  <div className="flex-1">
                    <div className="flex justify-between text-sm mb-1">
                      <span>Pass</span>
                      <span>Defer</span>
                      <span>Invest</span>
                    </div>
                    <div className="relative h-3 bg-gradient-to-r from-red-200 via-yellow-200 to-green-200 rounded-full">
                      <div
                        className="absolute top-1/2 -translate-y-1/2 w-4 h-4 bg-foreground rounded-full border-2 border-background shadow-md"
                        style={{ left: `${result.confidence * 100}%`, transform: 'translate(-50%, -50%)' }}
                      />
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Uncertainty Tab */}
        <TabsContent value="uncertainty" className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            {/* Epistemic Uncertainty */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <Brain className="h-5 w-5" />
                  Epistemic Uncertainty
                </CardTitle>
                <CardDescription>
                  Uncertainty due to lack of knowledge (reducible)
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="text-center p-4">
                  <p className="text-4xl font-bold">
                    {(result.robustness.epistemic_uncertainty * 100).toFixed(0)}%
                  </p>
                  <p className="text-sm text-muted-foreground mt-1">
                    Could be reduced with more data
                  </p>
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Low</span>
                    <span>High</span>
                  </div>
                  <Progress value={result.robustness.epistemic_uncertainty * 100} className="h-2" />
                </div>
                <div className="p-3 rounded-lg bg-muted/30 text-sm">
                  <p className="font-medium mb-1">Sources of epistemic uncertainty:</p>
                  <ul className="list-disc list-inside space-y-1 text-muted-foreground">
                    <li>Claims with confidence below 0.8</li>
                    <li>Missing due diligence data</li>
                    <li>Unverified founder claims</li>
                  </ul>
                </div>
              </CardContent>
            </Card>

            {/* Aleatoric Uncertainty */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <AlertTriangle className="h-5 w-5" />
                  Aleatoric Uncertainty
                </CardTitle>
                <CardDescription>
                  Inherent randomness in the system (irreducible)
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="text-center p-4">
                  <p className="text-4xl font-bold">
                    {(result.robustness.aleatoric_uncertainty * 100).toFixed(0)}%
                  </p>
                  <p className="text-sm text-muted-foreground mt-1">
                    Cannot be reduced with more data
                  </p>
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Low</span>
                    <span>High</span>
                  </div>
                  <Progress value={result.robustness.aleatoric_uncertainty * 100} className="h-2" />
                </div>
                <div className="p-3 rounded-lg bg-muted/30 text-sm">
                  <p className="font-medium mb-1">Sources of aleatoric uncertainty:</p>
                  <ul className="list-disc list-inside space-y-1 text-muted-foreground">
                    <li>Market volatility</li>
                    <li>Competitive dynamics</li>
                    <li>Macroeconomic factors</li>
                  </ul>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Stability Margin */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Decision Stability</CardTitle>
              <CardDescription>
                How much the evidence would need to change to flip the decision
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-3 gap-4">
                <div className="p-4 rounded-lg border text-center">
                  <p className="text-sm text-muted-foreground">Stability Margin</p>
                  <p className="text-2xl font-semibold mt-1">
                    {(result.robustness.stability_margin * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="p-4 rounded-lg border text-center">
                  <p className="text-sm text-muted-foreground">Flips Found</p>
                  <p className="text-2xl font-semibold mt-1">{result.robustness.flips_found}</p>
                </div>
                <div className="p-4 rounded-lg border text-center">
                  <p className="text-sm text-muted-foreground">Total Counterfactuals</p>
                  <p className="text-2xl font-semibold mt-1">
                    {result.robustness.counterfactuals_tested}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Counterfactuals Tab */}
        <TabsContent value="counterfactuals" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Counterfactual Explanations</CardTitle>
              <CardDescription>
                What would need to change for the decision to flip?
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {result.counterfactual_explanations.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  <RefreshCw className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>No decision-flipping counterfactuals found.</p>
                  <p className="text-sm mt-1">
                    The current decision is highly robust to perturbations.
                  </p>
                </div>
              ) : (
                result.counterfactual_explanations.map((cf, i) => (
                  <CounterfactualCard key={i} cf={cf} index={i} />
                ))
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Timeline Tab */}
        <TabsContent value="timeline" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Reasoning Timeline</CardTitle>
              <CardDescription>
                Step-by-step trace of the decision process
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="relative">
                {/* Timeline line */}
                <div className="absolute left-4 top-0 bottom-0 w-0.5 bg-border" />

                <div className="space-y-4">
                  {result.trace_entries.map((entry, i) => (
                    <div key={i} className="relative pl-10">
                      {/* Timeline dot */}
                      <div
                        className={`absolute left-2.5 w-3 h-3 rounded-full border-2 ${
                          entry.type === "analysis_complete"
                            ? "bg-green-500 border-green-500"
                            : entry.type === "critical_claim"
                            ? "bg-yellow-500 border-yellow-500"
                            : entry.type === "counterfactual_flip"
                            ? "bg-red-500 border-red-500"
                            : "bg-background border-primary"
                        }`}
                      />

                      <div className="p-3 rounded-lg border bg-card">
                        <div className="flex items-center justify-between mb-1">
                          <Badge variant="outline" className="text-xs">
                            {entry.type.replace(/_/g, " ")}
                          </Badge>
                          <span className="text-xs text-muted-foreground">
                            {entry.timestamp}
                          </span>
                        </div>
                        <p className="text-sm">{entry.message}</p>
                        {entry.details && (
                          <details className="mt-2">
                            <summary className="text-xs text-muted-foreground cursor-pointer hover:text-foreground">
                              View details
                            </summary>
                            <pre className="mt-2 p-2 rounded bg-muted/50 text-xs overflow-x-auto">
                              {JSON.stringify(entry.details, null, 2)}
                            </pre>
                          </details>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Disclaimer */}
      <Disclaimer />
    </div>
  );
}

function CriticalClaimCard({ claim, index }: { claim: CriticalClaim; index: number }) {
  const polarityIcon =
    claim.polarity === "supportive" ? (
      <TrendingUp className="h-4 w-4 text-green-600" />
    ) : claim.polarity === "risk" ? (
      <TrendingDown className="h-4 w-4 text-red-600" />
    ) : (
      <Minus className="h-4 w-4 text-gray-500" />
    );

  return (
    <Card
      className="border-l-4"
      style={{
        borderLeftColor:
          claim.polarity === "supportive"
            ? "rgb(22 163 74)"
            : claim.polarity === "risk"
            ? "rgb(220 38 38)"
            : "rgb(156 163 175)",
      }}
    >
      <CardContent className="py-3">
        <div className="flex items-start justify-between">
          <div className="flex items-start gap-3">
            <div className="mt-0.5">{polarityIcon}</div>
            <div>
              <p className="font-medium">{claim.claim_type}</p>
              <p className="text-sm text-muted-foreground">
                {claim.field}: {String(claim.value)}
              </p>
            </div>
          </div>
          <div className="text-right">
            <Badge
              variant={
                claim.criticality_score > 0.7
                  ? "destructive"
                  : claim.criticality_score > 0.4
                  ? "secondary"
                  : "outline"
              }
            >
              Criticality: {(claim.criticality_score * 100).toFixed(0)}%
            </Badge>
            <p className="text-xs text-muted-foreground mt-1">
              Confidence: {(claim.confidence * 100).toFixed(0)}%
            </p>
          </div>
        </div>
        {claim.flip_description && (
          <p className="text-sm mt-2 p-2 rounded bg-muted/30">
            <span className="font-medium">Impact: </span>
            {claim.flip_description}
          </p>
        )}
      </CardContent>
    </Card>
  );
}

function CounterfactualCard({
  cf,
  index,
}: {
  cf: CounterfactualExplanation;
  index: number;
}) {
  return (
    <Card className="border-l-4 border-l-yellow-500">
      <CardContent className="py-4">
        <div className="flex items-start gap-3">
          <AlertTriangle className="h-5 w-5 text-yellow-600 mt-0.5" />
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-2">
              <Badge variant="outline">{cf.original_decision}</Badge>
              <span className="text-muted-foreground">→</span>
              <Badge
                variant={
                  cf.flipped_decision === "invest"
                    ? "invest"
                    : cf.flipped_decision === "pass"
                    ? "pass"
                    : "defer"
                }
              >
                {cf.flipped_decision}
              </Badge>
            </div>
            <p className="text-sm">{cf.explanation}</p>
            {cf.key_changes && cf.key_changes.length > 0 && (
              <div className="mt-3">
                <p className="text-xs font-medium text-muted-foreground mb-1">
                  Key Changes:
                </p>
                <ul className="text-sm space-y-1">
                  {cf.key_changes.map((change, i) => (
                    <li key={i} className="flex items-center gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-yellow-500" />
                      {change}
                    </li>
                  ))}
                </ul>
              </div>
            )}
            <div className="mt-3 flex items-center gap-4 text-xs text-muted-foreground">
              <span>Perturbation magnitude: {(cf.perturbation_magnitude * 100).toFixed(1)}%</span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
