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
  Lightbulb,
  HelpCircle,
  Target,
  Layers,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import type {
  AnalysisResult,
  CriticalClaim,
  CounterfactualExplanation,
} from "@/types/analysis";
import type { EvidenceGraph } from "@/types/evidence";
import { Disclaimer } from "@/components/Disclaimer";
import { simulateReportHTML, type VCJobResult, type PolicyVariant } from "@/lib/api";

export default function AuditPage() {
  const router = useRouter();
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [vcResult, setVcResult] = useState<VCJobResult | null>(null);
  const [evidenceGraph, setEvidenceGraph] = useState<EvidenceGraph | null>(null);
  const [showReportModal, setShowReportModal] = useState(false);
  const [reportHTML, setReportHTML] = useState<string | null>(null);
  const [selectedPolicyIndex, setSelectedPolicyIndex] = useState(0);

  useEffect(() => {
    // Load legacy analysis result
    const stored = sessionStorage.getItem("analysisResult");
    if (stored) {
      setResult(JSON.parse(stored));
    }

    // Load new VC job result
    const vcStored = sessionStorage.getItem("vcJobResult");
    if (vcStored) {
      setVcResult(JSON.parse(vcStored));
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
    a.download = `decision-report-${result?.company_id || vcResult?.job_id || "report"}.${format === "pdf" ? "html" : format}`;
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

  // If we have VC result, show the new enhanced audit page
  if (vcResult && vcResult.decision) {
    return (
      <EnhancedAuditPage
        vcResult={vcResult}
        evidenceGraph={evidenceGraph}
        selectedPolicyIndex={selectedPolicyIndex}
        setSelectedPolicyIndex={setSelectedPolicyIndex}
        router={router}
      />
    );
  }

  // Legacy audit page for old analysis results
  if (!result) {
    return (
      <div className="flex flex-col items-center justify-center py-20">
        <AlertCircle className="h-12 w-12 text-muted-foreground mb-4" />
        <h2 className="text-xl font-semibold mb-2">No Analysis Result</h2>
        <p className="text-muted-foreground mb-4">Please run an analysis first.</p>
        <div className="flex gap-2">
          <Button onClick={() => router.push("/reasoning")}>
            Go to VC Reasoning
          </Button>
          <Button variant="outline" onClick={() => router.push("/analyze")}>
            Go to Legacy Analysis
          </Button>
        </div>
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
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => downloadReport("html")}
                >
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
                  <p>
                    The decision engine evaluated the evidence graph using the
                    following criteria:
                  </p>
                  <ul className="list-disc list-inside space-y-1 ml-2 mt-2">
                    <li>
                      <span className="font-medium">Net Signal Score:</span>{" "}
                      Weighted sum of supportive vs risk claims
                    </li>
                    <li>
                      <span className="font-medium">Confidence Threshold:</span>{" "}
                      Claims below 0.5 confidence are down-weighted
                    </li>
                    <li>
                      <span className="font-medium">Critical Claim Impact:</span>{" "}
                      High-criticality claims have outsized influence
                    </li>
                    <li>
                      <span className="font-medium">Robustness Check:</span>{" "}
                      Decision stability under perturbations
                    </li>
                  </ul>
                </div>
              </div>

              {/* Critical Claims */}
              <div>
                <h4 className="font-medium mb-3">
                  Critical Claims ({result.critical_claims.length})
                </h4>
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
                        style={{
                          left: `${result.confidence * 100}%`,
                          transform: "translate(-50%, -50%)",
                        }}
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
                  <Progress
                    value={result.robustness.epistemic_uncertainty * 100}
                    className="h-2"
                  />
                </div>
                <div className="p-3 rounded-lg bg-muted/30 text-sm">
                  <p className="font-medium mb-1">
                    Sources of epistemic uncertainty:
                  </p>
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
                  <Progress
                    value={result.robustness.aleatoric_uncertainty * 100}
                    className="h-2"
                  />
                </div>
                <div className="p-3 rounded-lg bg-muted/30 text-sm">
                  <p className="font-medium mb-1">
                    Sources of aleatoric uncertainty:
                  </p>
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
                  <p className="text-2xl font-semibold mt-1">
                    {result.robustness.flips_found}
                  </p>
                </div>
                <div className="p-4 rounded-lg border text-center">
                  <p className="text-sm text-muted-foreground">
                    Total Counterfactuals
                  </p>
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

// Enhanced Audit Page for VC Job Results
function EnhancedAuditPage({
  vcResult,
  evidenceGraph,
  selectedPolicyIndex,
  setSelectedPolicyIndex,
  router,
}: {
  vcResult: VCJobResult;
  evidenceGraph: EvidenceGraph | null;
  selectedPolicyIndex: number;
  setSelectedPolicyIndex: (index: number) => void;
  router: ReturnType<typeof useRouter>;
}) {
  const decision = vcResult.decision!;
  const policies = vcResult.policies || [];
  const uncertainty = vcResult.uncertainty;
  const counterfactuals = vcResult.counterfactuals || [];

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Full Audit</h1>
          <p className="text-muted-foreground mt-1">
            Policy variants, claims analysis, and uncertainty insights
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" onClick={() => router.push("/reasoning")}>
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Reasoning
          </Button>
          <Button variant="outline" onClick={() => router.push("/")}>
            Back to Workspace
          </Button>
        </div>
      </div>

      {/* Primary Decision Summary */}
      <Card>
        <CardContent className="py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div
                className={`p-3 rounded-full ${
                  decision.primary === "invest"
                    ? "bg-green-100 dark:bg-green-900"
                    : decision.primary === "pass"
                    ? "bg-red-100 dark:bg-red-900"
                    : "bg-yellow-100 dark:bg-yellow-900"
                }`}
              >
                {decision.primary === "invest" ? (
                  <CheckCircle2 className="h-8 w-8 text-green-600" />
                ) : decision.primary === "pass" ? (
                  <XCircle className="h-8 w-8 text-red-600" />
                ) : (
                  <Clock className="h-8 w-8 text-yellow-600" />
                )}
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Primary Decision</p>
                <div className="flex items-center gap-3 mt-1">
                  <Badge
                    variant={
                      decision.primary === "invest"
                        ? "invest"
                        : decision.primary === "pass"
                        ? "pass"
                        : "defer"
                    }
                    className="text-xl px-4 py-1"
                  >
                    {decision.primary.toUpperCase()}
                  </Badge>
                  <span className="text-lg">
                    {(decision.confidence * 100).toFixed(0)}% confidence
                  </span>
                </div>
              </div>
            </div>
            <div className="text-right">
              <p className="text-sm text-muted-foreground">Policy Variants</p>
              <p className="text-3xl font-bold mt-1">{policies.length}</p>
            </div>
          </div>
          {decision.explanation && (
            <p className="mt-4 text-sm text-muted-foreground bg-muted/30 p-3 rounded-lg">
              {decision.explanation}
            </p>
          )}
        </CardContent>
      </Card>

      {/* Tabs */}
      <Tabs defaultValue="policies" className="space-y-4">
        <TabsList className="grid grid-cols-4 w-full">
          <TabsTrigger value="policies" className="flex items-center gap-2">
            <Layers className="h-4 w-4" />
            Policy Variants ({policies.length})
          </TabsTrigger>
          <TabsTrigger value="uncertainty" className="flex items-center gap-2">
            <HelpCircle className="h-4 w-4" />
            Uncertainty
          </TabsTrigger>
          <TabsTrigger value="counterfactuals" className="flex items-center gap-2">
            <RefreshCw className="h-4 w-4" />
            Counterfactuals
          </TabsTrigger>
          <TabsTrigger value="improve" className="flex items-center gap-2">
            <Lightbulb className="h-4 w-4" />
            Reduce Uncertainty
          </TabsTrigger>
        </TabsList>

        {/* Policy Variants Tab */}
        <TabsContent value="policies" className="space-y-4">
          {policies.length === 0 ? (
            <Card>
              <CardContent className="py-8 text-center text-muted-foreground">
                <Layers className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>No policy variants generated</p>
              </CardContent>
            </Card>
          ) : (
            <>
              {/* Policy Selector */}
              <div className="flex gap-2 overflow-x-auto pb-2">
                {policies.map((policy, index) => (
                  <Button
                    key={policy.policy_id}
                    variant={selectedPolicyIndex === index ? "default" : "outline"}
                    onClick={() => setSelectedPolicyIndex(index)}
                    className="shrink-0"
                  >
                    <Badge
                      variant={
                        policy.decision === "invest"
                          ? "invest"
                          : policy.decision === "pass"
                          ? "pass"
                          : "defer"
                      }
                      className="mr-2"
                    >
                      {policy.decision}
                    </Badge>
                    Policy {index + 1}
                    {index === 0 && (
                      <Badge variant="secondary" className="ml-2">
                        Best
                      </Badge>
                    )}
                  </Button>
                ))}
              </div>

              {/* Side by Side Comparison */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                {policies.slice(0, 3).map((policy, index) => (
                  <PolicyVariantCard
                    key={policy.policy_id}
                    policy={policy}
                    index={index}
                    isSelected={selectedPolicyIndex === index}
                    onSelect={() => setSelectedPolicyIndex(index)}
                  />
                ))}
              </div>

              {/* Selected Policy Details */}
              {policies[selectedPolicyIndex] && (
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg flex items-center gap-2">
                      <Target className="h-5 w-5" />
                      Policy {selectedPolicyIndex + 1} Rules Detail
                    </CardTitle>
                    <CardDescription>
                      Predicates and claims used by this policy variant
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {policies[selectedPolicyIndex].rules.map((rule, ruleIndex) => (
                      <div
                        key={ruleIndex}
                        className="p-4 rounded-lg border bg-muted/20"
                      >
                        <div className="flex items-center justify-between mb-2">
                          <code className="text-sm font-mono bg-background px-2 py-1 rounded">
                            {rule.predicate}
                          </code>
                          <Badge variant="outline">
                            weight: {(rule.weight * 100).toFixed(0)}%
                          </Badge>
                        </div>
                        <div className="flex flex-wrap gap-1">
                          {rule.claim_ids.map((claimId) => (
                            <Badge key={claimId} variant="secondary" className="text-xs">
                              {claimId}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    ))}
                  </CardContent>
                </Card>
              )}
            </>
          )}
        </TabsContent>

        {/* Uncertainty Tab */}
        <TabsContent value="uncertainty" className="space-y-4">
          {uncertainty ? (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {/* Uncertainty Level */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Overall Uncertainty</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="text-center p-6">
                    <Badge
                      variant={
                        uncertainty.level === "low"
                          ? "default"
                          : uncertainty.level === "high" ||
                            uncertainty.level === "very_high"
                          ? "destructive"
                          : "secondary"
                      }
                      className="text-2xl px-6 py-2"
                    >
                      {uncertainty.level.toUpperCase()}
                    </Badge>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-4 rounded-lg border text-center">
                      <p className="text-sm text-muted-foreground">Epistemic</p>
                      <p className="text-2xl font-bold mt-1">
                        {(uncertainty.epistemic * 100).toFixed(0)}%
                      </p>
                      <p className="text-xs text-muted-foreground">Reducible</p>
                    </div>
                    <div className="p-4 rounded-lg border text-center">
                      <p className="text-sm text-muted-foreground">Aleatoric</p>
                      <p className="text-2xl font-bold mt-1">
                        {(uncertainty.aleatoric * 100).toFixed(0)}%
                      </p>
                      <p className="text-xs text-muted-foreground">Irreducible</p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Missing Information */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg flex items-center gap-2">
                    <AlertTriangle className="h-5 w-5 text-amber-500" />
                    Missing Information
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-2">
                    {uncertainty.missing_info.map((info, i) => (
                      <li
                        key={i}
                        className="flex items-start gap-2 p-2 rounded bg-muted/30"
                      >
                        <AlertCircle className="h-4 w-4 text-amber-500 mt-0.5 shrink-0" />
                        <span className="text-sm">{info}</span>
                      </li>
                    ))}
                  </ul>
                </CardContent>
              </Card>
            </div>
          ) : (
            <Card>
              <CardContent className="py-8 text-center text-muted-foreground">
                <HelpCircle className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>No uncertainty analysis available</p>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* Counterfactuals Tab */}
        <TabsContent value="counterfactuals" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Counterfactual Analysis</CardTitle>
              <CardDescription>
                What changes would flip the decision?
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {counterfactuals.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  <RefreshCw className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>No counterfactuals generated</p>
                  <p className="text-sm mt-1">
                    The decision appears robust to perturbations
                  </p>
                </div>
              ) : (
                counterfactuals.map((cf, i) => (
                  <div
                    key={i}
                    className="p-4 rounded-lg border border-l-4 border-l-amber-500"
                  >
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
                      <Badge variant="secondary" className="ml-auto">
                        {(cf.magnitude * 100).toFixed(0)}% change
                      </Badge>
                    </div>
                    <p className="text-sm">{cf.description}</p>
                    {cf.key_changes.length > 0 && (
                      <div className="flex flex-wrap gap-1 mt-2">
                        {cf.key_changes.map((change, j) => (
                          <Badge key={j} variant="outline" className="text-xs">
                            {change}
                          </Badge>
                        ))}
                      </div>
                    )}
                  </div>
                ))
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Reduce Uncertainty Tab */}
        <TabsContent value="improve" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Lightbulb className="h-5 w-5 text-amber-500" />
                How to Reduce Uncertainty
              </CardTitle>
              <CardDescription>
                Suggested questions and data to gather for higher confidence
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {uncertainty?.suggested_questions &&
              uncertainty.suggested_questions.length > 0 ? (
                <div>
                  <h4 className="font-medium mb-3">Suggested Questions</h4>
                  <div className="space-y-2">
                    {uncertainty.suggested_questions.map((question, i) => (
                      <div
                        key={i}
                        className="flex items-start gap-3 p-3 rounded-lg bg-blue-50 dark:bg-blue-950 border border-blue-200 dark:border-blue-800"
                      >
                        <HelpCircle className="h-5 w-5 text-blue-500 mt-0.5 shrink-0" />
                        <span className="text-sm">{question}</span>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  <Lightbulb className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>No specific suggestions available</p>
                  <p className="text-sm mt-1">
                    The model has sufficient confidence in the current evidence
                  </p>
                </div>
              )}

              {uncertainty?.missing_info &&
                uncertainty.missing_info.length > 0 && (
                  <div>
                    <h4 className="font-medium mb-3">Data Gaps to Address</h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                      {uncertainty.missing_info.map((info, i) => (
                        <div
                          key={i}
                          className="flex items-start gap-2 p-3 rounded-lg border"
                        >
                          <Target className="h-4 w-4 text-amber-500 mt-0.5 shrink-0" />
                          <span className="text-sm">{info}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

              <div className="p-4 rounded-lg bg-muted/30 border">
                <h4 className="font-medium mb-2">Next Steps</h4>
                <ol className="list-decimal list-inside space-y-1 text-sm text-muted-foreground">
                  <li>Gather answers to the suggested questions above</li>
                  <li>Add new claims to the evidence graph</li>
                  <li>Re-run context preview with updated constraints</li>
                  <li>Execute reasoning again for improved confidence</li>
                </ol>
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

// Policy Variant Card Component
function PolicyVariantCard({
  policy,
  index,
  isSelected,
  onSelect,
}: {
  policy: PolicyVariant;
  index: number;
  isSelected: boolean;
  onSelect: () => void;
}) {
  return (
    <Card
      className={`cursor-pointer transition-all ${
        isSelected ? "ring-2 ring-primary" : "hover:border-primary/50"
      }`}
      onClick={onSelect}
    >
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Policy {index + 1}</CardTitle>
          {index === 0 && (
            <Badge variant="secondary" className="text-xs">
              Best MDL
            </Badge>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {/* Decision */}
        <div className="text-center p-3 rounded-lg bg-muted/30">
          <Badge
            variant={
              policy.decision === "invest"
                ? "invest"
                : policy.decision === "pass"
                ? "pass"
                : "defer"
            }
            className="text-lg px-3 py-0.5"
          >
            {policy.decision.toUpperCase()}
          </Badge>
          <p className="text-sm text-muted-foreground mt-1">
            {(policy.confidence * 100).toFixed(0)}% confidence
          </p>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 gap-2 text-center">
          <div className="p-2 rounded border">
            <p className="text-lg font-bold">{policy.mdl_score.toFixed(1)}</p>
            <p className="text-xs text-muted-foreground">MDL Score</p>
          </div>
          <div className="p-2 rounded border">
            <p className="text-lg font-bold">{policy.rules.length}</p>
            <p className="text-xs text-muted-foreground">Rules</p>
          </div>
        </div>

        {/* Coverage */}
        <div className="p-2 rounded bg-muted/30">
          <div className="flex justify-between text-xs text-muted-foreground mb-1">
            <span>Coverage</span>
            <span>
              {policy.coverage.covered}/{policy.coverage.total}
            </span>
          </div>
          <Progress
            value={(policy.coverage.covered / policy.coverage.total) * 100}
            className="h-1.5"
          />
        </div>
      </CardContent>
    </Card>
  );
}

function CriticalClaimCard({
  claim,
  index,
}: {
  claim: CriticalClaim;
  index: number;
}) {
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
              <span>
                Perturbation magnitude: {(cf.perturbation_magnitude * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
