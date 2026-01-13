"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { Loader2, CheckCircle2, XCircle, AlertCircle, ArrowRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import type { EvidenceGraph } from "@/types/evidence";
import type { AnalysisResult } from "@/types/analysis";
import { simulateAnalysis } from "@/lib/api";
import {
  Disclaimer,
  UnderdeterminedWarning,
  LowConfidenceWarning,
  checkEvidenceRequirements,
  shouldRefuseAnalysis,
} from "@/components/Disclaimer";

type AnalysisState = "idle" | "running" | "completed" | "error";

interface AnalysisEvent {
  timestamp: Date;
  type: string;
  message: string;
}

export default function AnalyzePage() {
  const router = useRouter();
  const [evidenceGraph, setEvidenceGraph] = useState<EvidenceGraph | null>(null);
  const [state, setState] = useState<AnalysisState>("idle");
  const [progress, setProgress] = useState(0);
  const [events, setEvents] = useState<AnalysisEvent[]>([]);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [evidenceIssues, setEvidenceIssues] = useState<string[]>([]);
  const [refusalReason, setRefusalReason] = useState<string | null>(null);

  useEffect(() => {
    // Load evidence graph from sessionStorage
    const stored = sessionStorage.getItem("evidenceGraph");
    if (stored) {
      const graph = JSON.parse(stored);
      setEvidenceGraph(graph);

      // Check evidence requirements
      const requirements = checkEvidenceRequirements(graph.claims);
      if (!requirements.meetsRequirements) {
        setEvidenceIssues(requirements.issues);
      }
    }
  }, []);

  const addEvent = (type: string, message: string) => {
    setEvents((prev) => [
      ...prev,
      { timestamp: new Date(), type, message },
    ]);
  };

  const runAnalysis = async () => {
    if (!evidenceGraph) return;

    setState("running");
    setProgress(0);
    setEvents([]);
    setError(null);

    try {
      // Simulate progress events
      addEvent("start", "Starting VC decision analysis...");
      setProgress(10);

      await new Promise((r) => setTimeout(r, 500));
      addEvent("loading", `Loading ${evidenceGraph.claims.length} claims...`);
      setProgress(20);

      await new Promise((r) => setTimeout(r, 500));
      addEvent("validation", "Validating evidence graph against ontology...");
      setProgress(30);

      await new Promise((r) => setTimeout(r, 500));
      addEvent("processing", "Analyzing claim relationships...");
      setProgress(45);

      await new Promise((r) => setTimeout(r, 500));
      addEvent("counterfactual", "Generating counterfactual scenarios...");
      setProgress(60);

      await new Promise((r) => setTimeout(r, 500));
      addEvent("robustness", "Computing decision robustness...");
      setProgress(75);

      await new Promise((r) => setTimeout(r, 500));
      addEvent("criticality", "Identifying critical claims...");
      setProgress(85);

      // Run actual analysis (simulated for now)
      const analysisResult = await simulateAnalysis(evidenceGraph);
      setProgress(95);

      // Check if we should refuse the analysis due to insufficient confidence
      const refusal = shouldRefuseAnalysis(analysisResult);
      if (refusal.refuse) {
        setRefusalReason(refusal.reason);
        addEvent("refused", `Analysis refused: ${refusal.reason}`);
      } else {
        addEvent("complete", `Decision: ${analysisResult.decision.toUpperCase()}`);
      }
      setProgress(100);

      setResult(analysisResult);
      setState("completed");

      // Store result for audit page
      sessionStorage.setItem("analysisResult", JSON.stringify(analysisResult));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed");
      setState("error");
      addEvent("error", "Analysis failed");
    }
  };

  const viewAudit = () => {
    router.push("/audit");
  };

  if (!evidenceGraph) {
    return (
      <div className="flex flex-col items-center justify-center py-20">
        <AlertCircle className="h-12 w-12 text-muted-foreground mb-4" />
        <h2 className="text-xl font-semibold mb-2">No Evidence Graph</h2>
        <p className="text-muted-foreground mb-4">
          Please create an evidence graph first in the Deal Workspace.
        </p>
        <Button onClick={() => router.push("/")}>Go to Workspace</Button>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold">Run Analysis</h1>
        <p className="text-muted-foreground mt-1">
          Analyze your evidence graph for investment decision
        </p>
      </div>

      {/* Evidence Graph Summary */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center justify-between">
            <span>Evidence Graph: {evidenceGraph.company_id}</span>
            <Badge variant="outline">{evidenceGraph.claims.length} claims</Badge>
          </CardTitle>
          <CardDescription>
            {evidenceGraph.claims.filter((c) => c.polarity === "supportive").length} supportive,{" "}
            {evidenceGraph.claims.filter((c) => c.polarity === "risk").length} risk,{" "}
            {evidenceGraph.claims.filter((c) => c.polarity === "neutral").length} neutral
          </CardDescription>
        </CardHeader>
      </Card>

      {/* Analysis Control */}
      {state === "idle" && (
        <Card>
          <CardContent className="py-8 flex flex-col items-center">
            <p className="text-muted-foreground mb-4">
              Ready to analyze {evidenceGraph.claims.length} claims
            </p>
            <Button size="lg" onClick={runAnalysis}>
              Start Analysis
            </Button>
          </CardContent>
        </Card>
      )}

      {/* Progress */}
      {state === "running" && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <Loader2 className="h-5 w-5 animate-spin" />
              Analyzing...
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <Progress value={progress} className="h-2" />
            <p className="text-sm text-muted-foreground">{progress}% complete</p>

            {/* Event Stream */}
            <div className="border rounded-lg p-4 bg-muted/30 max-h-60 overflow-y-auto">
              {events.map((event, i) => (
                <div key={i} className="flex items-start gap-2 text-sm py-1">
                  <span className="text-muted-foreground text-xs">
                    {event.timestamp.toLocaleTimeString()}
                  </span>
                  <Badge variant="outline" className="text-xs">
                    {event.type}
                  </Badge>
                  <span>{event.message}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Evidence Issues Warning */}
      {evidenceIssues.length > 0 && (
        <UnderdeterminedWarning
          reason="Evidence graph does not meet minimum requirements for reliable analysis."
          missingClaims={evidenceIssues}
        />
      )}

      {/* Result */}
      {state === "completed" && result && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              {refusalReason ? (
                <>
                  <AlertCircle className="h-5 w-5 text-amber-600" />
                  Analysis Complete - Recommendation Withheld
                </>
              ) : (
                <>
                  <CheckCircle2 className="h-5 w-5 text-green-600" />
                  Analysis Complete
                </>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Refusal Warning */}
            {refusalReason && (
              <UnderdeterminedWarning reason={refusalReason} />
            )}

            {/* Low Confidence Warning */}
            {!refusalReason && <LowConfidenceWarning confidence={result.confidence} />}

            {/* Decision */}
            <div className="flex items-center justify-between p-4 rounded-lg bg-muted/30">
              <div>
                <p className="text-sm text-muted-foreground">
                  {refusalReason ? "Preliminary Assessment (Not a Recommendation)" : "Decision Support Output"}
                </p>
                <div className="flex items-center gap-2 mt-1">
                  <Badge
                    variant={
                      refusalReason
                        ? "outline"
                        : result.decision === "invest"
                        ? "invest"
                        : result.decision === "pass"
                        ? "pass"
                        : "defer"
                    }
                    className="text-lg px-3 py-1"
                  >
                    {refusalReason ? "UNDERDETERMINED" : result.decision.toUpperCase()}
                  </Badge>
                  <span className="text-muted-foreground">
                    {(result.confidence * 100).toFixed(0)}% confidence
                  </span>
                </div>
              </div>
              <Button onClick={viewAudit}>
                View Audit Trail
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </div>

            {/* Summary Stats */}
            <div className="grid grid-cols-3 gap-4">
              <div className="p-4 rounded-lg border">
                <p className="text-sm text-muted-foreground">Robustness Score</p>
                <p className="text-2xl font-semibold mt-1">
                  {(result.robustness.overall_score * 100).toFixed(0)}%
                </p>
              </div>
              <div className="p-4 rounded-lg border">
                <p className="text-sm text-muted-foreground">Critical Claims</p>
                <p className="text-2xl font-semibold mt-1">
                  {result.critical_claims.length}
                </p>
              </div>
              <div className="p-4 rounded-lg border">
                <p className="text-sm text-muted-foreground">Decision Flips Found</p>
                <p className="text-2xl font-semibold mt-1">
                  {result.robustness.flips_found}
                </p>
              </div>
            </div>

            {/* Key Counterfactuals */}
            {result.counterfactual_explanations.length > 0 && (
              <div>
                <h3 className="font-medium mb-2">Key Counterfactuals</h3>
                <div className="space-y-2">
                  {result.counterfactual_explanations.slice(0, 2).map((cf, i) => (
                    <div key={i} className="p-3 rounded-lg border bg-muted/20 text-sm">
                      {cf.explanation}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Event Stream */}
            <div>
              <h3 className="font-medium mb-2">Analysis Events</h3>
              <div className="border rounded-lg p-4 bg-muted/30 max-h-40 overflow-y-auto">
                {events.map((event, i) => (
                  <div key={i} className="flex items-start gap-2 text-sm py-1">
                    <span className="text-muted-foreground text-xs">
                      {event.timestamp.toLocaleTimeString()}
                    </span>
                    <Badge variant="outline" className="text-xs">
                      {event.type}
                    </Badge>
                    <span>{event.message}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Disclaimer */}
            <Disclaimer />
          </CardContent>
        </Card>
      )}

      {/* Error */}
      {state === "error" && (
        <Card>
          <CardContent className="py-8 flex flex-col items-center">
            <XCircle className="h-12 w-12 text-red-600 mb-4" />
            <h2 className="text-xl font-semibold mb-2">Analysis Failed</h2>
            <p className="text-muted-foreground mb-4">{error}</p>
            <Button onClick={runAnalysis}>Retry Analysis</Button>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
