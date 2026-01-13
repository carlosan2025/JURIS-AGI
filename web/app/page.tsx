"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Plus, Trash2, FileText, Play } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Slider } from "@/components/ui/slider";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import type { Claim, ClaimType, Polarity, EvidenceGraph } from "@/types/evidence";
import { CLAIM_TYPE_LABELS, CLAIM_TYPE_CATEGORIES, POLARITY_COLORS } from "@/types/evidence";

function generateId() {
  return Math.random().toString(36).substring(2, 9);
}

const SAMPLE_CLAIMS: Claim[] = [
  {
    id: generateId(),
    claim_type: "company_identity",
    field: "legal_name",
    value: "TechCorp Inc.",
    confidence: 1.0,
    polarity: "neutral",
  },
  {
    id: generateId(),
    claim_type: "traction",
    field: "mrr",
    value: 150000,
    confidence: 0.9,
    polarity: "supportive",
    unit: "USD",
  },
  {
    id: generateId(),
    claim_type: "team_quality",
    field: "founder_experience",
    value: "15 years in enterprise SaaS",
    confidence: 0.85,
    polarity: "supportive",
  },
  {
    id: generateId(),
    claim_type: "execution_risk",
    field: "key_hire_gap",
    value: "CTO position unfilled",
    confidence: 0.8,
    polarity: "risk",
  },
];

export default function DealWorkspace() {
  const router = useRouter();
  const [companyId, setCompanyId] = useState("demo-startup");
  const [claims, setClaims] = useState<Claim[]>(SAMPLE_CLAIMS);
  const [expandedCategories, setExpandedCategories] = useState<string[]>(
    Object.keys(CLAIM_TYPE_CATEGORIES)
  );

  const addClaim = (claimType: ClaimType) => {
    const newClaim: Claim = {
      id: generateId(),
      claim_type: claimType,
      field: "",
      value: "",
      confidence: 0.8,
      polarity: "neutral",
    };
    setClaims([...claims, newClaim]);
  };

  const updateClaim = (id: string, updates: Partial<Claim>) => {
    setClaims(claims.map((c) => (c.id === id ? { ...c, ...updates } : c)));
  };

  const removeClaim = (id: string) => {
    setClaims(claims.filter((c) => c.id !== id));
  };

  const getClaimsByType = (types: ClaimType[]) => {
    return claims.filter((c) => types.includes(c.claim_type));
  };

  const handleRunAnalysis = () => {
    const evidenceGraph: EvidenceGraph = {
      company_id: companyId,
      claims: claims.filter((c) => c.field && c.value),
    };
    // Store in sessionStorage for the analyze page
    sessionStorage.setItem("evidenceGraph", JSON.stringify(evidenceGraph));
    router.push("/analyze");
  };

  const claimCount = claims.filter((c) => c.field && c.value).length;
  const supportiveCount = claims.filter((c) => c.polarity === "supportive").length;
  const riskCount = claims.filter((c) => c.polarity === "risk").length;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Deal Workspace</h1>
          <p className="text-muted-foreground mt-1">
            Build your evidence graph for VC investment analysis
          </p>
        </div>
        <Button onClick={handleRunAnalysis} size="lg" disabled={claimCount === 0}>
          <Play className="mr-2 h-4 w-4" />
          Run Analysis
        </Button>
      </div>

      {/* Company Info */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Company Information</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <Label htmlFor="companyId">Company ID</Label>
              <Input
                id="companyId"
                value={companyId}
                onChange={(e) => setCompanyId(e.target.value)}
                placeholder="e.g., acme-corp"
              />
            </div>
            <div className="flex items-end gap-4">
              <div className="flex gap-2">
                <Badge variant="secondary">{claimCount} claims</Badge>
                <Badge variant="supportive">{supportiveCount} supportive</Badge>
                <Badge variant="risk">{riskCount} risks</Badge>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Claims by Category */}
      <Accordion type="multiple" value={expandedCategories} onValueChange={setExpandedCategories}>
        {Object.entries(CLAIM_TYPE_CATEGORIES).map(([category, types]) => (
          <AccordionItem key={category} value={category}>
            <AccordionTrigger className="text-lg font-semibold">
              <div className="flex items-center gap-2">
                {category}
                <Badge variant="outline" className="ml-2">
                  {getClaimsByType(types).length}
                </Badge>
              </div>
            </AccordionTrigger>
            <AccordionContent>
              <div className="space-y-4 pt-2">
                {types.map((claimType) => (
                  <div key={claimType} className="space-y-2">
                    <div className="flex items-center justify-between">
                      <Label className="text-sm font-medium">
                        {CLAIM_TYPE_LABELS[claimType]}
                      </Label>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => addClaim(claimType)}
                      >
                        <Plus className="h-4 w-4 mr-1" />
                        Add
                      </Button>
                    </div>
                    {getClaimsByType([claimType]).map((claim) => (
                      <ClaimEditor
                        key={claim.id}
                        claim={claim}
                        onUpdate={(updates) => updateClaim(claim.id, updates)}
                        onRemove={() => removeClaim(claim.id)}
                      />
                    ))}
                  </div>
                ))}
              </div>
            </AccordionContent>
          </AccordionItem>
        ))}
      </Accordion>
    </div>
  );
}

function ClaimEditor({
  claim,
  onUpdate,
  onRemove,
}: {
  claim: Claim;
  onUpdate: (updates: Partial<Claim>) => void;
  onRemove: () => void;
}) {
  return (
    <Card className="border-l-4" style={{
      borderLeftColor: claim.polarity === "supportive"
        ? "rgb(22 163 74)"
        : claim.polarity === "risk"
        ? "rgb(220 38 38)"
        : "rgb(156 163 175)"
    }}>
      <CardContent className="pt-4">
        <div className="grid grid-cols-12 gap-4">
          {/* Field */}
          <div className="col-span-3">
            <Label className="text-xs text-muted-foreground">Field</Label>
            <Input
              value={claim.field}
              onChange={(e) => onUpdate({ field: e.target.value })}
              placeholder="e.g., mrr, team_size"
              className="mt-1"
            />
          </div>

          {/* Value */}
          <div className="col-span-3">
            <Label className="text-xs text-muted-foreground">Value</Label>
            <Input
              value={String(claim.value)}
              onChange={(e) => {
                const val = e.target.value;
                const numVal = Number(val);
                onUpdate({ value: isNaN(numVal) ? val : numVal });
              }}
              placeholder="Value"
              className="mt-1"
            />
          </div>

          {/* Polarity */}
          <div className="col-span-2">
            <Label className="text-xs text-muted-foreground">Polarity</Label>
            <Select
              value={claim.polarity}
              onValueChange={(v: Polarity) => onUpdate({ polarity: v })}
            >
              <SelectTrigger className="mt-1">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="supportive">Supportive</SelectItem>
                <SelectItem value="risk">Risk</SelectItem>
                <SelectItem value="neutral">Neutral</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Confidence */}
          <div className="col-span-3">
            <Label className="text-xs text-muted-foreground">
              Confidence: {(claim.confidence * 100).toFixed(0)}%
            </Label>
            <Slider
              value={[claim.confidence * 100]}
              onValueChange={([v]) => onUpdate({ confidence: v / 100 })}
              max={100}
              step={5}
              className="mt-3"
            />
          </div>

          {/* Actions */}
          <div className="col-span-1 flex items-end justify-end">
            <Button variant="ghost" size="icon" onClick={onRemove}>
              <Trash2 className="h-4 w-4 text-muted-foreground" />
            </Button>
          </div>
        </div>

        {/* Source (collapsible) */}
        <details className="mt-3">
          <summary className="text-xs text-muted-foreground cursor-pointer hover:text-foreground">
            <FileText className="h-3 w-3 inline mr-1" />
            Source & Notes
          </summary>
          <div className="grid grid-cols-2 gap-4 mt-2">
            <div>
              <Label className="text-xs text-muted-foreground">Source Document</Label>
              <Input
                value={claim.source?.doc_id || ""}
                onChange={(e) =>
                  onUpdate({
                    source: { ...claim.source, doc_id: e.target.value },
                  })
                }
                placeholder="e.g., pitch_deck_v2"
                className="mt-1"
              />
            </div>
            <div>
              <Label className="text-xs text-muted-foreground">Notes</Label>
              <Input
                value={claim.notes || ""}
                onChange={(e) => onUpdate({ notes: e.target.value })}
                placeholder="Optional notes..."
                className="mt-1"
              />
            </div>
          </div>
        </details>
      </CardContent>
    </Card>
  );
}
