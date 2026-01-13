"use client";

import { useState, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import {
  AlertCircle,
  ArrowRight,
  ChevronDown,
  ChevronRight,
  Filter,
  Layers,
  AlertTriangle,
  Trash2,
  RefreshCw,
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
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import type { EvidenceGraph } from "@/types/evidence";
import {
  simulateContextPreview,
  type VCConstraints,
  type ContextPreview,
  type ContextClaim,
  type ContextConflict,
} from "@/lib/api";

export default function ContextPage() {
  const router = useRouter();
  const [evidenceGraph, setEvidenceGraph] = useState<EvidenceGraph | null>(null);
  const [preview, setPreview] = useState<ContextPreview | null>(null);
  const [expandedBuckets, setExpandedBuckets] = useState<Set<string>>(new Set());

  // Constraint controls
  const [maxClaims, setMaxClaims] = useState(50);
  const [minConfidence, setMinConfidence] = useState(0.5);
  const [perBucketCap, setPerBucketCap] = useState(10);

  useEffect(() => {
    // Load evidence graph from sessionStorage
    const stored = sessionStorage.getItem("evidenceGraph");
    if (stored) {
      const graph = JSON.parse(stored);
      setEvidenceGraph(graph);
    }
  }, []);

  const buildPreview = useCallback(() => {
    if (!evidenceGraph) return;

    const constraints: VCConstraints = {
      max_claims: maxClaims,
      min_confidence: minConfidence,
      per_bucket_cap: perBucketCap,
    };

    // Use simulation for demo mode
    const contextPreview = simulateContextPreview(evidenceGraph, constraints);
    setPreview(contextPreview);

    // Auto-expand buckets with conflicts or dropped claims
    const bucketsToExpand = new Set<string>();
    Object.entries(contextPreview.claims_by_bucket).forEach(([bucket, claims]) => {
      if (claims.some((c) => c.dropped)) {
        bucketsToExpand.add(bucket);
      }
    });
    setExpandedBuckets(bucketsToExpand);
  }, [evidenceGraph, maxClaims, minConfidence, perBucketCap]);

  useEffect(() => {
    if (evidenceGraph) {
      buildPreview();
    }
  }, [evidenceGraph, buildPreview]);

  const toggleBucket = (bucket: string) => {
    setExpandedBuckets((prev) => {
      const next = new Set(prev);
      if (next.has(bucket)) {
        next.delete(bucket);
      } else {
        next.add(bucket);
      }
      return next;
    });
  };

  const proceedToReasoning = () => {
    // Store constraints for use in reasoning
    sessionStorage.setItem(
      "vcConstraints",
      JSON.stringify({
        max_claims: maxClaims,
        min_confidence: minConfidence,
        per_bucket_cap: perBucketCap,
      })
    );
    router.push("/reasoning");
  };

  const getPolarityColor = (polarity: string) => {
    switch (polarity) {
      case "supportive":
        return "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200";
      case "risk":
        return "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200";
      default:
        return "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200";
    }
  };

  const getDropReasonLabel = (reason: string) => {
    switch (reason) {
      case "low_confidence":
        return "Low confidence";
      case "bucket_cap_exceeded":
        return "Bucket cap exceeded";
      case "max_claims_exceeded":
        return "Max claims exceeded";
      default:
        return reason;
    }
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
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Context Preview</h1>
          <p className="text-muted-foreground mt-1">
            Review and adjust claims before running VC reasoning
          </p>
        </div>
        <Badge variant="outline" className="text-lg px-3 py-1">
          {evidenceGraph.company_id}
        </Badge>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column - Constraints */}
        <div className="lg:col-span-1 space-y-6">
          {/* Constraint Controls */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Filter className="h-5 w-5" />
                Context Constraints
              </CardTitle>
              <CardDescription>
                Adjust parameters to control which claims are included
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Max Claims */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label htmlFor="maxClaims">Max Claims</Label>
                  <span className="text-sm text-muted-foreground">{maxClaims}</span>
                </div>
                <Slider
                  id="maxClaims"
                  min={10}
                  max={200}
                  step={5}
                  value={[maxClaims]}
                  onValueChange={([v]) => setMaxClaims(v)}
                />
                <p className="text-xs text-muted-foreground">
                  Maximum total claims to include in reasoning
                </p>
              </div>

              {/* Min Confidence */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label htmlFor="minConfidence">Min Confidence</Label>
                  <span className="text-sm text-muted-foreground">
                    {(minConfidence * 100).toFixed(0)}%
                  </span>
                </div>
                <Slider
                  id="minConfidence"
                  min={0}
                  max={1}
                  step={0.05}
                  value={[minConfidence]}
                  onValueChange={([v]) => setMinConfidence(v)}
                />
                <p className="text-xs text-muted-foreground">
                  Minimum confidence threshold for claims
                </p>
              </div>

              {/* Per-Bucket Cap */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label htmlFor="perBucketCap">Per-Bucket Cap</Label>
                  <span className="text-sm text-muted-foreground">{perBucketCap}</span>
                </div>
                <Slider
                  id="perBucketCap"
                  min={3}
                  max={30}
                  step={1}
                  value={[perBucketCap]}
                  onValueChange={([v]) => setPerBucketCap(v)}
                />
                <p className="text-xs text-muted-foreground">
                  Maximum claims per ontology bucket
                </p>
              </div>

              <Button onClick={buildPreview} variant="outline" className="w-full">
                <RefreshCw className="h-4 w-4 mr-2" />
                Rebuild Preview
              </Button>
            </CardContent>
          </Card>

          {/* Summary Stats */}
          {preview && (
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Context Summary</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="p-3 rounded-lg bg-muted/30 text-center">
                    <p className="text-2xl font-bold text-green-600">
                      {preview.selected_claims}
                    </p>
                    <p className="text-xs text-muted-foreground">Selected</p>
                  </div>
                  <div className="p-3 rounded-lg bg-muted/30 text-center">
                    <p className="text-2xl font-bold text-amber-600">
                      {preview.dropped_claims}
                    </p>
                    <p className="text-xs text-muted-foreground">Dropped</p>
                  </div>
                </div>

                {/* Dropped Reasons */}
                {Object.keys(preview.dropped_by_reason).length > 0 && (
                  <div>
                    <p className="text-sm font-medium mb-2">Dropped Reasons</p>
                    <div className="space-y-1">
                      {Object.entries(preview.dropped_by_reason).map(
                        ([reason, count]) => (
                          <div
                            key={reason}
                            className="flex items-center justify-between text-sm"
                          >
                            <span className="text-muted-foreground">
                              {getDropReasonLabel(reason)}
                            </span>
                            <Badge variant="secondary">{count}</Badge>
                          </div>
                        )
                      )}
                    </div>
                  </div>
                )}

                {/* Conflicts */}
                {preview.conflicts.length > 0 && (
                  <div>
                    <p className="text-sm font-medium mb-2 flex items-center gap-1">
                      <AlertTriangle className="h-4 w-4 text-amber-500" />
                      Conflicts Detected
                    </p>
                    <div className="space-y-2">
                      {preview.conflicts.map((conflict, i) => (
                        <div
                          key={i}
                          className="p-2 rounded border border-amber-200 bg-amber-50 dark:bg-amber-950 dark:border-amber-800 text-sm"
                        >
                          <p className="font-medium">{conflict.description}</p>
                          {conflict.resolution && (
                            <p className="text-xs text-muted-foreground mt-1">
                              Resolution: {conflict.resolution}
                            </p>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          {/* Proceed Button */}
          <Button onClick={proceedToReasoning} className="w-full" size="lg">
            Proceed to Reasoning
            <ArrowRight className="ml-2 h-5 w-5" />
          </Button>
        </div>

        {/* Right Column - Claims by Bucket */}
        <div className="lg:col-span-2">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Layers className="h-5 w-5" />
                Claims by Ontology Bucket
              </CardTitle>
              <CardDescription>
                {preview
                  ? `${Object.keys(preview.claims_by_bucket).length} buckets with ${preview.total_claims} total claims`
                  : "Loading..."}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {preview && (
                <div className="space-y-2">
                  {Object.entries(preview.claims_by_bucket)
                    .sort(([a], [b]) => a.localeCompare(b))
                    .map(([bucket, claims]) => {
                      const selectedCount = claims.filter((c) => !c.dropped).length;
                      const droppedCount = claims.filter((c) => c.dropped).length;
                      const isExpanded = expandedBuckets.has(bucket);

                      return (
                        <Collapsible
                          key={bucket}
                          open={isExpanded}
                          onOpenChange={() => toggleBucket(bucket)}
                        >
                          <CollapsibleTrigger className="w-full">
                            <div className="flex items-center justify-between p-3 rounded-lg border hover:bg-muted/30 transition-colors">
                              <div className="flex items-center gap-2">
                                {isExpanded ? (
                                  <ChevronDown className="h-4 w-4" />
                                ) : (
                                  <ChevronRight className="h-4 w-4" />
                                )}
                                <span className="font-medium">{bucket}</span>
                              </div>
                              <div className="flex items-center gap-2">
                                <Badge variant="default">{selectedCount}</Badge>
                                {droppedCount > 0 && (
                                  <Badge variant="secondary" className="opacity-60">
                                    <Trash2 className="h-3 w-3 mr-1" />
                                    {droppedCount}
                                  </Badge>
                                )}
                              </div>
                            </div>
                          </CollapsibleTrigger>
                          <CollapsibleContent>
                            <div className="mt-2 ml-6 space-y-2">
                              {claims.map((claim) => (
                                <ClaimItem
                                  key={claim.claim_id}
                                  claim={claim}
                                  getPolarityColor={getPolarityColor}
                                  getDropReasonLabel={getDropReasonLabel}
                                />
                              ))}
                            </div>
                          </CollapsibleContent>
                        </Collapsible>
                      );
                    })}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}

function ClaimItem({
  claim,
  getPolarityColor,
  getDropReasonLabel,
}: {
  claim: ContextClaim;
  getPolarityColor: (polarity: string) => string;
  getDropReasonLabel: (reason: string) => string;
}) {
  return (
    <div
      className={`p-3 rounded-lg border text-sm ${
        claim.dropped
          ? "opacity-50 bg-muted/20 border-dashed"
          : "bg-background"
      }`}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="font-medium truncate">{claim.field}</span>
            <Badge className={getPolarityColor(claim.polarity)} variant="outline">
              {claim.polarity}
            </Badge>
            <Badge variant="outline" className="text-xs">
              {(claim.confidence * 100).toFixed(0)}%
            </Badge>
          </div>
          <p className="text-muted-foreground mt-1 truncate">
            {typeof claim.value === "object"
              ? JSON.stringify(claim.value)
              : String(claim.value)}
          </p>
        </div>
        {claim.dropped && claim.drop_reason && (
          <Badge variant="destructive" className="text-xs shrink-0">
            {getDropReasonLabel(claim.drop_reason)}
          </Badge>
        )}
      </div>
    </div>
  );
}
