"use client";

import { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import {
  AlertCircle,
  ArrowRight,
  Loader2,
  CheckCircle2,
  XCircle,
  Clock,
  Zap,
  Brain,
  Target,
  Shield,
  HelpCircle,
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
import { Progress } from "@/components/ui/progress";
import type { EvidenceGraph } from "@/types/evidence";
import {
  simulateVCJob,
  type VCConstraints,
  type VCJobEvent,
  type VCJobResult,
} from "@/lib/api";
import { Disclaimer } from "@/components/Disclaimer";

type ReasoningState = "idle" | "running" | "completed" | "error";

const EVENT_ICONS: Record<string, React.ReactNode> = {
  job_started: <Zap className="h-4 w-4 text-blue-500" />,
  context_built: <Target className="h-4 w-4 text-green-500" />,
  working_set_built: <Brain className="h-4 w-4 text-purple-500" />,
  thresholds_proposed: <Shield className="h-4 w-4 text-amber-500" />,
  policies_learning: <Loader2 className="h-4 w-4 animate-spin text-blue-500" />,
  policies_learned: <CheckCircle2 className="h-4 w-4 text-green-500" />,
  evaluation_complete: <Target className="h-4 w-4 text-green-500" />,
  uncertainty_analyzed: <HelpCircle className="h-4 w-4 text-amber-500" />,
  job_completed: <CheckCircle2 className="h-4 w-4 text-green-600" />,
  job_failed: <XCircle className="h-4 w-4 text-red-600" />,
};

export default function ReasoningPage() {
  const router = useRouter();
  const [evidenceGraph, setEvidenceGraph] = useState<EvidenceGraph | null>(null);
  const [constraints, setConstraints] = useState<VCConstraints | null>(null);
  const [state, setState] = useState<ReasoningState>("idle");
  const [events, setEvents] = useState<VCJobEvent[]>([]);
  const [result, setResult] = useState<VCJobResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const eventsEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Load evidence graph and constraints from sessionStorage
    const storedGraph = sessionStorage.getItem("evidenceGraph");
    const storedConstraints = sessionStorage.getItem("vcConstraints");

    if (storedGraph) {
      setEvidenceGraph(JSON.parse(storedGraph));
    }
    if (storedConstraints) {
      setConstraints(JSON.parse(storedConstraints));
    }
  }, []);

  // Auto-scroll to bottom of events
  useEffect(() => {
    eventsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [events]);

  const handleEvent = (event: VCJobEvent) => {
    setEvents((prev) => [...prev, event]);

    // Update progress based on event type
    const progressMap: Record<string, number> = {
      job_started: 5,
      context_built: 15,
      working_set_built: 30,
      thresholds_proposed: 45,
      policies_learning: 55,
      policies_learned: 70,
      evaluation_complete: 85,
      uncertainty_analyzed: 95,
      job_completed: 100,
    };

    if (progressMap[event.event_type]) {
      setProgress(progressMap[event.event_type]);
    }
  };

  const runReasoning = async () => {
    if (!evidenceGraph) return;

    setState("running");
    setProgress(0);
    setEvents([]);
    setError(null);
    setResult(null);

    try {
      // Use simulation for demo mode
      const jobResult = await simulateVCJob(
        evidenceGraph,
        constraints || undefined,
        handleEvent
      );

      setResult(jobResult);
      setState("completed");

      // Store result for audit page
      sessionStorage.setItem("vcJobResult", JSON.stringify(jobResult));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Reasoning failed");
      setState("error");
    }
  };

  const viewAudit = () => {
    router.push("/audit");
  };

  const goToContext = () => {
    router.push("/context");
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
    <div className="max-w-5xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Run Reasoning</h1>
          <p className="text-muted-foreground mt-1">
            Execute VC decision reasoning with live progress tracking
          </p>
        </div>
        <Badge variant="outline" className="text-lg px-3 py-1">
          {evidenceGraph.company_id}
        </Badge>
      </div>

      {/* Constraints Summary */}
      {constraints && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Applied Constraints</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex gap-4 text-sm">
              <div>
                <span className="text-muted-foreground">Max Claims:</span>{" "}
                <span className="font-medium">{constraints.max_claims || 50}</span>
              </div>
              <div>
                <span className="text-muted-foreground">Min Confidence:</span>{" "}
                <span className="font-medium">
                  {((constraints.min_confidence || 0.5) * 100).toFixed(0)}%
                </span>
              </div>
              <div>
                <span className="text-muted-foreground">Per-Bucket Cap:</span>{" "}
                <span className="font-medium">{constraints.per_bucket_cap || 10}</span>
              </div>
              <Button variant="link" size="sm" onClick={goToContext} className="ml-auto">
                Modify Constraints
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left Column - Events Timeline */}
        <Card className="lg:row-span-2">
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <Clock className="h-5 w-5" />
              Events Timeline
            </CardTitle>
            <CardDescription>
              Real-time progress of the reasoning pipeline
            </CardDescription>
          </CardHeader>
          <CardContent>
            {state === "running" && (
              <div className="mb-4">
                <Progress value={progress} className="h-2" />
                <p className="text-xs text-muted-foreground mt-1">
                  {progress}% complete
                </p>
              </div>
            )}

            <div className="relative">
              {/* Timeline line */}
              {events.length > 0 && (
                <div className="absolute left-[11px] top-0 bottom-0 w-0.5 bg-border" />
              )}

              {/* Events */}
              <div className="space-y-4 max-h-[500px] overflow-y-auto pr-2">
                {events.length === 0 && state === "idle" && (
                  <p className="text-muted-foreground text-sm text-center py-8">
                    Click "Start Reasoning" to begin
                  </p>
                )}

                {events.map((event, index) => (
                  <div key={index} className="flex gap-3 relative">
                    {/* Icon */}
                    <div className="relative z-10 flex items-center justify-center w-6 h-6 rounded-full bg-background border">
                      {EVENT_ICONS[event.event_type] || (
                        <div className="w-2 h-2 rounded-full bg-muted-foreground" />
                      )}
                    </div>

                    {/* Content */}
                    <div className="flex-1 pb-4">
                      <div className="flex items-center gap-2">
                        <Badge variant="outline" className="text-xs">
                          {event.event_type.replace(/_/g, " ")}
                        </Badge>
                        <span className="text-xs text-muted-foreground">
                          {new Date(event.timestamp).toLocaleTimeString()}
                        </span>
                      </div>
                      <p className="text-sm mt-1">{event.message}</p>
                      {event.details && Object.keys(event.details).length > 0 && (
                        <div className="mt-2 text-xs text-muted-foreground bg-muted/30 rounded p-2">
                          {Object.entries(event.details).map(([key, value]) => (
                            <div key={key}>
                              <span className="font-medium">{key}:</span>{" "}
                              {typeof value === "object"
                                ? JSON.stringify(value)
                                : String(value)}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                ))}

                {state === "running" && (
                  <div className="flex gap-3 relative">
                    <div className="relative z-10 flex items-center justify-center w-6 h-6 rounded-full bg-background border">
                      <Loader2 className="h-4 w-4 animate-spin text-blue-500" />
                    </div>
                    <div className="flex-1">
                      <p className="text-sm text-muted-foreground">Processing...</p>
                    </div>
                  </div>
                )}

                <div ref={eventsEndRef} />
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Right Column - Controls and Results */}
        <div className="space-y-6">
          {/* Start Button */}
          {state === "idle" && (
            <Card>
              <CardContent className="py-8 flex flex-col items-center">
                <Brain className="h-16 w-16 text-muted-foreground mb-4" />
                <p className="text-muted-foreground mb-4 text-center">
                  Ready to analyze {evidenceGraph.claims.length} claims using
                  neuro-symbolic reasoning
                </p>
                <Button size="lg" onClick={runReasoning}>
                  <Zap className="h-5 w-5 mr-2" />
                  Start Reasoning
                </Button>
              </CardContent>
            </Card>
          )}

          {/* Running Status */}
          {state === "running" && (
            <Card>
              <CardContent className="py-8 flex flex-col items-center">
                <Loader2 className="h-12 w-12 animate-spin text-blue-500 mb-4" />
                <p className="text-lg font-medium">Running VC Reasoning</p>
                <p className="text-muted-foreground text-sm mt-1">
                  This may take a few moments...
                </p>
              </CardContent>
            </Card>
          )}

          {/* Error */}
          {state === "error" && (
            <Card>
              <CardContent className="py-8 flex flex-col items-center">
                <XCircle className="h-12 w-12 text-red-600 mb-4" />
                <h2 className="text-xl font-semibold mb-2">Reasoning Failed</h2>
                <p className="text-muted-foreground mb-4">{error}</p>
                <Button onClick={runReasoning}>Retry</Button>
              </CardContent>
            </Card>
          )}

          {/* Result Summary */}
          {state === "completed" && result && result.decision && (
            <>
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg flex items-center gap-2">
                    <CheckCircle2 className="h-5 w-5 text-green-600" />
                    Reasoning Complete
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  {/* Decision */}
                  <div className="p-4 rounded-lg bg-muted/30 text-center">
                    <p className="text-sm text-muted-foreground mb-2">
                      Primary Decision
                    </p>
                    <Badge
                      variant={
                        result.decision.primary === "invest"
                          ? "invest"
                          : result.decision.primary === "pass"
                          ? "pass"
                          : "defer"
                      }
                      className="text-xl px-4 py-1"
                    >
                      {result.decision.primary.toUpperCase()}
                    </Badge>
                    <p className="text-sm text-muted-foreground mt-2">
                      {(result.decision.confidence * 100).toFixed(0)}% confidence
                    </p>
                  </div>

                  {/* Stats */}
                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-3 rounded-lg border text-center">
                      <p className="text-2xl font-bold">
                        {result.policies?.length || 0}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        Policy Variants
                      </p>
                    </div>
                    <div className="p-3 rounded-lg border text-center">
                      <p className="text-2xl font-bold">
                        {result.context_summary?.claims_used || 0}
                      </p>
                      <p className="text-xs text-muted-foreground">Claims Used</p>
                    </div>
                  </div>

                  {/* Uncertainty */}
                  {result.uncertainty && (
                    <div className="p-3 rounded-lg border">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium">
                          Uncertainty Level
                        </span>
                        <Badge
                          variant={
                            result.uncertainty.level === "low"
                              ? "default"
                              : result.uncertainty.level === "high" ||
                                result.uncertainty.level === "very_high"
                              ? "destructive"
                              : "secondary"
                          }
                        >
                          {result.uncertainty.level}
                        </Badge>
                      </div>
                      <div className="flex gap-4 text-xs text-muted-foreground">
                        <span>
                          Epistemic:{" "}
                          {(result.uncertainty.epistemic * 100).toFixed(0)}%
                        </span>
                        <span>
                          Aleatoric:{" "}
                          {(result.uncertainty.aleatoric * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                  )}

                  <Button onClick={viewAudit} className="w-full">
                    View Full Audit
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </Button>
                </CardContent>
              </Card>

              <Disclaimer />
            </>
          )}
        </div>
      </div>
    </div>
  );
}
