"use client";

import { useState, useRef, useCallback } from "react";
import { useRouter } from "next/navigation";
import {
  Upload,
  FileText,
  Loader2,
  CheckCircle2,
  XCircle,
  AlertCircle,
  Check,
  X,
  Edit2,
  ArrowRight,
  TrendingUp,
  TrendingDown,
  Minus,
  File,
  FileUp,
  Sparkles,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import type {
  DocumentType,
  ProposedClaim,
  ExtractionResult,
  ClaimStatus,
} from "@/types/extraction";
import { DOCUMENT_TYPE_LABELS, DOCUMENT_TYPE_DESCRIPTIONS } from "@/types/extraction";
import { simulateExtraction } from "@/lib/api";

// Sample documents for demo mode
const SAMPLE_DOCUMENTS: Record<string, { type: DocumentType; content: string; companyId: string }> = {
  "HealthBridge AI - Pitch Deck": {
    type: "pitch_deck",
    companyId: "healthbridge-2024",
    content: `HEALTHBRIDGE AI
Clinical Decision Support for Primary Care

EXECUTIVE SUMMARY
HealthBridge AI is building an AI-powered clinical decision support system for primary care physicians. Our platform analyzes patient history, symptoms, and test results to suggest differential diagnoses and care pathways.

PROBLEM
- Primary care physicians face information overload
- 12 minutes average per patient visit
- Diagnostic errors affect 12M Americans annually
- 80% of serious medical errors involve miscommunication during patient handoffs

SOLUTION
AI-powered differential diagnosis assistant that integrates with EHR systems to provide real-time clinical decision support.

TRACTION
- 3 pilot clinics in the Bay Area
- $180K ARR (growing 20% MoM)
- 72 NPS from physician users
- 15,000+ diagnoses assisted
- 92% concordance with specialist diagnoses

TEAM
- CEO: Dr. Sarah Chen - MIT CS PhD, 8 years ML at Apple Health
- CTO: James Wu - Stanford ML, ex-Google Health
- Clinical Advisors: 2 board-certified physicians (part-time)
- Team Size: 12 employees

MARKET OPPORTUNITY
- $12B TAM in clinical decision support
- Post-COVID telehealth adoption accelerating AI interest
- FDA's AI/ML guidance provides regulatory pathway

PRODUCT
- Real-time EHR integration (Epic, Cerner)
- Natural language symptom input
- Evidence-based recommendations with citations
- HIPAA compliant, SOC 2 Type II certified

COMPETITIVE LANDSCAPE
- Nuance (Microsoft) - Large incumbent, enterprise focus
- Isabel Healthcare - Rule-based, legacy technology
- Infermedica - Consumer-facing, less clinical depth

BUSINESS MODEL
- SaaS: $500/physician/month
- Enterprise contracts: Custom pricing
- Gross margin: 75%
- Sales cycle: 6-9 months

REGULATORY STATUS
- Pre-submission meeting with FDA scheduled
- Targeting Class II 510(k) pathway
- Estimated 12-18 months to clearance
- HIPAA BAAs in place with all pilot partners

FINANCIALS
- Monthly burn: $120K
- Runway: 18 months post-raise
- Use of funds: 40% R&D, 35% Sales, 15% Regulatory, 10% G&A

THE ASK
Series A: $4M at $18M pre-money valuation
Instrument: Series A Preferred Stock

EXIT POTENTIAL
Strategic acquirers: Epic, Cerner (Oracle), Veeva, Teladoc
Comparable exits: Buoy Health ($75M), K Health ($132M)`,
  },
  "TechStartup - Financial Model": {
    type: "financial_model",
    companyId: "techstartup-2024",
    content: `FINANCIAL MODEL SUMMARY - TECHSTARTUP INC.

PROFIT & LOSS (Actuals)
=====================================
                    Q1 2024    Q2 2024    Q3 2024    Q4 2024
Revenue             $150K      $185K      $220K      $280K
COGS                $35K       $42K       $50K       $63K
Gross Profit        $115K      $143K      $170K      $217K
Gross Margin        76.7%      77.3%      77.3%      77.5%

Operating Expenses
- R&D               $80K       $85K       $90K       $95K
- Sales & Marketing $45K       $55K       $70K       $85K
- G&A               $25K       $28K       $30K       $32K
Total OpEx          $150K      $168K      $190K      $212K

EBITDA              ($35K)     ($25K)     ($20K)     $5K
Net Burn            $35K       $25K       $20K       ($5K)

KEY METRICS
=====================================
ARR (Dec 2024):     $1,120K (Annual Run Rate from Q4)
MRR (Dec 2024):     $93K
MoM Growth:         12% average
Customers:          45 (up from 18 at start of year)
Net Revenue Retention: 115%
CAC:                $8,500
LTV:                $42,000
LTV/CAC:            4.9x

CASH FLOW
=====================================
Beginning Cash (Jan 2024):  $1.2M
Net Operating Cash Flow:    ($75K)
Ending Cash (Dec 2024):     $1.125M
Monthly Burn (current):     $0K (breakeven as of Q4)

PROJECTIONS (2025)
=====================================
Revenue Target:     $2.5M ARR
Growth Rate:        120% YoY
Gross Margin:       78%
Path to Profitability: Q2 2025 (already achieved in Q4 2024)

UNIT ECONOMICS
=====================================
Average Contract Value: $25K/year
Gross Margin per Customer: $19,500/year
Payback Period: 5.2 months
Churn Rate: 8% annual`,
  },
  "AI Platform - Technical Overview": {
    type: "tech_description",
    companyId: "aiplatform-2024",
    content: `TECHNICAL ARCHITECTURE OVERVIEW
AI Platform Inc. - Confidential

1. SYSTEM OVERVIEW
==================
AI Platform is a machine learning operations (MLOps) platform that enables enterprise teams to deploy, monitor, and iterate on ML models at scale.

Current Status: Production deployment serving 25K daily active users
Uptime: 99.95% over past 12 months
Latency: p99 < 100ms for inference requests

2. ARCHITECTURE
===============
2.1 Core Infrastructure
- Kubernetes-based microservices architecture
- Multi-cloud deployment (AWS primary, GCP secondary)
- Current capacity: 500 req/sec (monolithic architecture handling well)
- Note: Planning migration to distributed architecture in 2025

2.2 Data Pipeline
- Real-time data ingestion via Kafka
- Feature store built on Redis + PostgreSQL
- Batch processing via Apache Spark
- Data lake on S3 with Delta Lake format

2.3 ML Infrastructure
- Model serving via TensorFlow Serving and Triton
- Auto-scaling based on request volume
- A/B testing framework for model deployment
- Automated retraining pipelines

3. ML CAPABILITIES
==================
3.1 Proprietary Technology
- Custom transformer architecture for time-series prediction
- Models trained on 5M+ proprietary labeled examples from enterprise customers
- Transfer learning framework reducing training time by 60%

3.2 Model Performance
- Average model accuracy: 94% (varies by use case)
- Training time: 2-4 hours for typical enterprise model
- Inference latency: 15ms average

4. SECURITY & COMPLIANCE
========================
- SOC 2 Type II certified
- GDPR compliant data handling
- End-to-end encryption (TLS 1.3)
- Role-based access control
- Audit logging for all operations

5. TECHNICAL TEAM
=================
- 8 ML engineers (Stanford, MIT, CMU backgrounds)
- 4 infrastructure engineers
- 2 security specialists
- Technical advisory board includes ex-Google Brain researcher

6. ROADMAP
==========
Q1 2025: Distributed architecture migration
Q2 2025: GPU cluster expansion (3x current capacity)
Q3 2025: AutoML feature release
Q4 2025: On-premise deployment option`,
  },
  "BioHealth - IC Memo": {
    type: "ic_memo",
    companyId: "biohealth-2024",
    content: `INVESTMENT COMMITTEE MEMO
BioHealth Diagnostics - Series A

CONFIDENTIAL - For IC Review Only
Date: January 2025
Deal Lead: Partner A
Second: Partner B

DEAL TERMS
==========
Pre-money Valuation: $15M
Raise Amount: $4M on Series A Preferred
Pro-forma Ownership: 21%
Board Seat: Yes (Observer initially)
Lead Investor: Our Fund (50% of round)
Co-investors: Health Ventures ($1M), Angels ($1M)

COMPANY OVERVIEW
================
BioHealth Diagnostics is developing AI-powered diagnostic tools for early cancer detection using liquid biopsy technology. The company has developed a proprietary biomarker panel that can detect 12 cancer types from a single blood draw.

TEAM ASSESSMENT
===============
CEO: Dr. Michael Roberts
- 20 years in diagnostics industry
- Former VP at Illumina (2010-2020)
- PhD in Molecular Biology, UCSF
- Strong domain expertise verified via extensive reference calls

CTO: Dr. Lisa Wang
- 15 years ML experience
- Former Google Health (ML Lead)
- Published 40+ papers in computational biology
- Note: Technical knowledge concentrated primarily in CTO role

Team Size: 18 employees
- 8 scientists (PhD level)
- 5 engineers
- 5 operations/business

KEY STRENGTHS
=============
1. Exceptional domain expertise in founding team
2. Proprietary biomarker panel with early clinical validation
3. Strong IP portfolio (12 patents filed, 4 granted)
4. Early FDA engagement showing positive signals
5. Strategic partnership discussions with major lab networks

KEY RISKS
=========
1. Regulatory timeline uncertain (FDA approval 18-24 months)
2. Key person risk - Technical knowledge concentrated in CTO
3. Capital intensive R&D (will need Series B within 18 months)
4. Competitive pressure from Grail (Illumina) and Guardant
5. Reimbursement pathway not yet established

TRACTION
========
- 500 samples processed in pilot studies
- 89% sensitivity, 95% specificity in initial validation
- 2 health system LOIs worth $2M potential ARR
- Research collaboration with Stanford Medicine
- Peer-reviewed publication pending in Nature Medicine

FINANCIAL SUMMARY
=================
- Current cash: $800K
- Monthly burn: $150K
- ARR: $0 (pre-revenue, R&D stage)
- Projected revenue: $2M in Year 2 post-FDA approval

USE OF PROCEEDS
===============
- 50% R&D (clinical validation studies)
- 25% Regulatory (FDA submission preparation)
- 15% Team expansion (hire VP Sales, 2 scientists)
- 10% G&A

EXIT ANALYSIS
=============
Potential Acquirers:
- Illumina (strategic fit, prior relationship)
- Roche Diagnostics
- Thermo Fisher
- Quest Diagnostics

Comparable Exits:
- Grail acquired by Illumina for $8B
- Guardant IPO at $3B valuation
- Foundation Medicine acquired by Roche for $5.3B

RECOMMENDATION
==============
We recommend PROCEEDING with this investment. While there are meaningful risks around regulatory timeline and capital intensity, the team quality, market opportunity, and early clinical data are compelling. The cancer diagnostics market is experiencing significant tailwinds, and BioHealth's multi-cancer approach is differentiated.

Key Conditions:
1. Board observer seat converting to full seat at Series B
2. Monthly financial reporting requirement
3. Key person insurance on CEO and CTO`,
  },
};

const SAMPLE_DOC_NAMES = Object.keys(SAMPLE_DOCUMENTS);

type ExtractionState = "idle" | "uploading" | "extracting" | "reviewing" | "merging" | "complete";

export default function ExtractPage() {
  const router = useRouter();
  const [state, setState] = useState<ExtractionState>("idle");
  const [docId, setDocId] = useState("");
  const [docType, setDocType] = useState<DocumentType>("pitch_deck");
  const [content, setContent] = useState("");
  const [companyId, setCompanyId] = useState("");
  const [result, setResult] = useState<ExtractionResult | null>(null);
  const [claims, setClaims] = useState<ProposedClaim[]>([]);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [editingClaim, setEditingClaim] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [fileName, setFileName] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Handle file upload
  const handleFileUpload = useCallback(async (file: File) => {
    setError(null);
    setFileName(file.name);

    // Determine document type from file name
    const name = file.name.toLowerCase();
    if (name.includes("pitch") || name.includes("deck")) {
      setDocType("pitch_deck");
    } else if (name.includes("financial") || name.includes("model") || name.includes("p&l")) {
      setDocType("financial_model");
    } else if (name.includes("tech") || name.includes("architecture")) {
      setDocType("tech_description");
    } else if (name.includes("memo") || name.includes("ic")) {
      setDocType("ic_memo");
    }

    // Set doc ID from file name
    const docName = file.name.replace(/\.[^/.]+$/, "").replace(/[^a-zA-Z0-9]/g, "_").toLowerCase();
    setDocId(docName);

    // Read file content
    try {
      const text = await file.text();
      setContent(text);
    } catch (err) {
      setError("Failed to read file. Please ensure it's a text-based document.");
    }
  }, []);

  // Handle drag and drop
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileUpload(files[0]);
    }
  }, [handleFileUpload]);

  // Handle file input change
  const handleFileInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFileUpload(files[0]);
    }
  }, [handleFileUpload]);

  // Load sample document
  const loadSampleDocument = useCallback((sampleName: string) => {
    const sample = SAMPLE_DOCUMENTS[sampleName];
    if (sample) {
      setContent(sample.content);
      setDocType(sample.type);
      setCompanyId(sample.companyId);
      setDocId(sample.companyId + "_" + sample.type);
      setFileName(sampleName + ".txt");
      setError(null);
    }
  }, []);

  const handleExtract = async () => {
    if (!content.trim()) {
      setError("Please paste document content");
      return;
    }

    setState("extracting");
    setProgress(0);
    setError(null);

    try {
      // Simulate progress
      const progressInterval = setInterval(() => {
        setProgress((p) => Math.min(p + 10, 90));
      }, 150);

      // Run extraction (simulated for demo)
      const extractionResult = await simulateExtraction(
        docId || `doc_${Date.now()}`,
        docType,
        content
      );

      clearInterval(progressInterval);
      setProgress(100);

      setResult(extractionResult);
      setClaims(extractionResult.proposed_claims);
      setState("reviewing");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Extraction failed");
      setState("idle");
    }
  };

  const handleClaimAction = (proposalId: string, action: "approve" | "reject") => {
    setClaims((prev) =>
      prev.map((c) =>
        c.proposal_id === proposalId
          ? { ...c, status: action === "approve" ? "approved" : "rejected" }
          : c
      )
    );
  };

  const handleClaimEdit = (proposalId: string, updates: Partial<ProposedClaim>) => {
    setClaims((prev) =>
      prev.map((c) =>
        c.proposal_id === proposalId
          ? { ...c, ...updates, status: "modified" }
          : c
      )
    );
    setEditingClaim(null);
  };

  const handleMerge = () => {
    const approvedClaims = claims.filter(
      (c) => c.status === "approved" || c.status === "modified"
    );

    if (approvedClaims.length === 0) {
      setError("No approved claims to merge");
      return;
    }

    // Store merged claims in sessionStorage for the workspace
    const existingGraph = sessionStorage.getItem("evidenceGraph");
    const graph = existingGraph ? JSON.parse(existingGraph) : { company_id: companyId || "extracted", claims: [] };

    // Convert proposed claims to evidence claims
    const newClaims = approvedClaims.map((pc) => ({
      id: pc.proposal_id,
      claim_type: pc.claim_type,
      field: pc.field,
      value: pc.modified_value ?? pc.value,
      confidence: pc.modified_confidence ?? pc.confidence,
      polarity: pc.modified_polarity ?? pc.polarity,
      source: {
        doc_id: result?.doc_id,
        locator: pc.locator,
        quote: pc.quote,
        doc_type: result?.doc_type,
      },
      notes: `Extracted from ${result?.doc_type}. Rationale: ${pc.rationale}`,
    }));

    graph.claims = [...graph.claims, ...newClaims];
    graph.company_id = companyId || graph.company_id;
    sessionStorage.setItem("evidenceGraph", JSON.stringify(graph));

    setState("complete");
  };

  const approvedCount = claims.filter((c) => c.status === "approved" || c.status === "modified").length;
  const rejectedCount = claims.filter((c) => c.status === "rejected").length;
  const pendingCount = claims.filter((c) => c.status === "pending").length;

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold">Extract Claims</h1>
        <p className="text-muted-foreground mt-1">
          Use LLM to extract evidence claims from documents
        </p>
      </div>

      {/* Upload/Input Section */}
      {state === "idle" && (
        <div className="space-y-6">
          {/* Demo Mode Banner */}
          <Card className="border-blue-200 bg-blue-50">
            <CardContent className="py-4">
              <div className="flex items-start gap-3">
                <Sparkles className="h-5 w-5 text-blue-600 mt-0.5" />
                <div>
                  <p className="font-medium text-blue-900">Demo Mode</p>
                  <p className="text-sm text-blue-700">
                    This demo uses simulated extraction. In production, documents would be processed by an LLM to extract claims with citations.
                    Try a sample document below to see how it works.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Sample Documents */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Quick Start - Sample Documents</CardTitle>
              <CardDescription>
                Load a sample document to try the extraction workflow
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-3">
                {SAMPLE_DOC_NAMES.map((name) => (
                  <Button
                    key={name}
                    variant="outline"
                    className="h-auto py-3 px-4 justify-start"
                    onClick={() => loadSampleDocument(name)}
                  >
                    <FileText className="h-4 w-4 mr-2 flex-shrink-0" />
                    <span className="text-left truncate">{name}</span>
                  </Button>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* File Upload */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Upload Document</CardTitle>
              <CardDescription>
                Drag and drop a file or click to browse. Supports .txt, .md, .pdf, and other text files.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Drag and Drop Zone */}
              <div
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                className={`
                  border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors
                  ${isDragging
                    ? "border-blue-500 bg-blue-50"
                    : fileName
                      ? "border-green-500 bg-green-50"
                      : "border-gray-300 hover:border-gray-400 hover:bg-gray-50"
                  }
                `}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  onChange={handleFileInputChange}
                  accept=".txt,.md,.pdf,.doc,.docx,.rtf"
                  className="hidden"
                />
                {fileName ? (
                  <div className="flex flex-col items-center gap-2">
                    <File className="h-10 w-10 text-green-600" />
                    <p className="font-medium text-green-700">{fileName}</p>
                    <p className="text-sm text-green-600">File loaded successfully</p>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={(e) => {
                        e.stopPropagation();
                        setFileName(null);
                        setContent("");
                      }}
                    >
                      Remove
                    </Button>
                  </div>
                ) : (
                  <div className="flex flex-col items-center gap-2">
                    <FileUp className="h-10 w-10 text-gray-400" />
                    <p className="font-medium text-gray-700">
                      {isDragging ? "Drop file here" : "Drag and drop your document"}
                    </p>
                    <p className="text-sm text-muted-foreground">
                      or click to browse
                    </p>
                  </div>
                )}
              </div>

              {/* Document Details */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="docId">Document ID (optional)</Label>
                  <Input
                    id="docId"
                    value={docId}
                    onChange={(e) => setDocId(e.target.value)}
                    placeholder="e.g., pitch_deck_v2"
                  />
                </div>
                <div>
                  <Label htmlFor="companyId">Company ID</Label>
                  <Input
                    id="companyId"
                    value={companyId}
                    onChange={(e) => setCompanyId(e.target.value)}
                    placeholder="e.g., acme-corp"
                  />
                </div>
              </div>

              <div>
                <Label htmlFor="docType">Document Type</Label>
                <Select value={docType} onValueChange={(v: DocumentType) => setDocType(v)}>
                  <SelectTrigger className="mt-1">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {(Object.keys(DOCUMENT_TYPE_LABELS) as DocumentType[]).map((type) => (
                      <SelectItem key={type} value={type}>
                        <div>
                          <div>{DOCUMENT_TYPE_LABELS[type]}</div>
                          <div className="text-xs text-muted-foreground">
                            {DOCUMENT_TYPE_DESCRIPTIONS[type]}
                          </div>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label htmlFor="content">Document Content</Label>
                <Textarea
                  id="content"
                  value={content}
                  onChange={(e) => setContent(e.target.value)}
                  placeholder="Paste the document text here, or upload/load a sample document above..."
                  className="mt-1 min-h-[200px] font-mono text-sm"
                />
                {content && (
                  <p className="text-xs text-muted-foreground mt-1">
                    {content.length.toLocaleString()} characters
                  </p>
                )}
              </div>

              {error && (
                <div className="flex items-center gap-2 text-red-600 text-sm">
                  <AlertCircle className="h-4 w-4" />
                  {error}
                </div>
              )}

              <Button onClick={handleExtract} size="lg" disabled={!content.trim()}>
                <FileText className="mr-2 h-4 w-4" />
                Extract Claims
              </Button>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Extraction Progress */}
      {state === "extracting" && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <Loader2 className="h-5 w-5 animate-spin" />
              Extracting Claims...
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <Progress value={progress} className="h-2" />
            <p className="text-sm text-muted-foreground">
              Analyzing {DOCUMENT_TYPE_LABELS[docType]} with LLM...
            </p>
          </CardContent>
        </Card>
      )}

      {/* Review Section */}
      {state === "reviewing" && result && (
        <>
          {/* Summary */}
          <Card>
            <CardContent className="py-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <FileText className="h-8 w-8 text-muted-foreground" />
                  <div>
                    <p className="font-medium">{result.doc_id}</p>
                    <p className="text-sm text-muted-foreground">
                      {DOCUMENT_TYPE_LABELS[docType as DocumentType]} • {claims.length} claims extracted
                    </p>
                  </div>
                </div>
                <div className="flex gap-2">
                  <Badge variant="supportive">{approvedCount} approved</Badge>
                  <Badge variant="risk">{rejectedCount} rejected</Badge>
                  <Badge variant="secondary">{pendingCount} pending</Badge>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Claims Review */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Review Proposed Claims</CardTitle>
              <CardDescription>
                Approve, reject, or modify each extracted claim before merging
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {claims.map((claim) => (
                <ClaimReviewCard
                  key={claim.proposal_id}
                  claim={claim}
                  isEditing={editingClaim === claim.proposal_id}
                  onApprove={() => handleClaimAction(claim.proposal_id, "approve")}
                  onReject={() => handleClaimAction(claim.proposal_id, "reject")}
                  onEdit={() => setEditingClaim(claim.proposal_id)}
                  onSaveEdit={(updates) => handleClaimEdit(claim.proposal_id, updates)}
                  onCancelEdit={() => setEditingClaim(null)}
                />
              ))}

              {claims.length === 0 && (
                <div className="text-center py-8 text-muted-foreground">
                  No claims extracted from this document.
                </div>
              )}
            </CardContent>
          </Card>

          {/* Merge Action */}
          <div className="flex justify-between items-center">
            <Button variant="outline" onClick={() => setState("idle")}>
              Start Over
            </Button>
            <Button
              onClick={handleMerge}
              disabled={approvedCount === 0}
              size="lg"
            >
              Merge {approvedCount} Claims
              <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
          </div>
        </>
      )}

      {/* Complete */}
      {state === "complete" && (
        <Card>
          <CardContent className="py-8 flex flex-col items-center">
            <CheckCircle2 className="h-12 w-12 text-green-600 mb-4" />
            <h2 className="text-xl font-semibold mb-2">Claims Merged Successfully</h2>
            <p className="text-muted-foreground mb-4">
              {approvedCount} claims have been added to your evidence graph.
            </p>
            <div className="flex gap-4">
              <Button variant="outline" onClick={() => setState("idle")}>
                Extract More
              </Button>
              <Button onClick={() => router.push("/")}>
                Go to Workspace
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function ClaimReviewCard({
  claim,
  isEditing,
  onApprove,
  onReject,
  onEdit,
  onSaveEdit,
  onCancelEdit,
}: {
  claim: ProposedClaim;
  isEditing: boolean;
  onApprove: () => void;
  onReject: () => void;
  onEdit: () => void;
  onSaveEdit: (updates: Partial<ProposedClaim>) => void;
  onCancelEdit: () => void;
}) {
  const [editValue, setEditValue] = useState(String(claim.value));
  const [editConfidence, setEditConfidence] = useState(claim.confidence);
  const [editPolarity, setEditPolarity] = useState(claim.polarity);

  const polarityIcon =
    claim.polarity === "supportive" ? (
      <TrendingUp className="h-4 w-4 text-green-600" />
    ) : claim.polarity === "risk" ? (
      <TrendingDown className="h-4 w-4 text-red-600" />
    ) : (
      <Minus className="h-4 w-4 text-gray-500" />
    );

  const statusBadge = {
    pending: <Badge variant="secondary">Pending</Badge>,
    approved: <Badge variant="supportive">Approved</Badge>,
    rejected: <Badge variant="risk">Rejected</Badge>,
    modified: <Badge variant="outline">Modified</Badge>,
    merged: <Badge variant="secondary">Merged</Badge>,
  };

  if (isEditing) {
    return (
      <Card className="border-l-4 border-l-yellow-500">
        <CardContent className="py-4 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Badge variant="outline">{claim.claim_type}</Badge>
              <span className="text-sm text-muted-foreground">{claim.field}</span>
            </div>
            <span className="text-sm text-muted-foreground">Editing...</span>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <Label>Value</Label>
              <Input
                value={editValue}
                onChange={(e) => setEditValue(e.target.value)}
                className="mt-1"
              />
            </div>
            <div>
              <Label>Polarity</Label>
              <Select value={editPolarity} onValueChange={(v: "supportive" | "risk" | "neutral") => setEditPolarity(v)}>
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
          </div>

          <div>
            <Label>Confidence: {(editConfidence * 100).toFixed(0)}%</Label>
            <Slider
              value={[editConfidence * 100]}
              onValueChange={([v]) => setEditConfidence(v / 100)}
              max={100}
              step={5}
              className="mt-2"
            />
          </div>

          <div className="flex justify-end gap-2">
            <Button variant="outline" size="sm" onClick={onCancelEdit}>
              Cancel
            </Button>
            <Button
              size="sm"
              onClick={() =>
                onSaveEdit({
                  modified_value: editValue,
                  modified_confidence: editConfidence,
                  modified_polarity: editPolarity,
                })
              }
            >
              Save Changes
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card
      className={`border-l-4 ${
        claim.status === "approved"
          ? "border-l-green-500 bg-green-50/50"
          : claim.status === "rejected"
          ? "border-l-red-500 bg-red-50/50 opacity-60"
          : claim.status === "modified"
          ? "border-l-yellow-500 bg-yellow-50/50"
          : "border-l-gray-300"
      }`}
    >
      <CardContent className="py-4">
        <div className="flex items-start justify-between gap-4">
          <div className="flex-1 space-y-2">
            {/* Header */}
            <div className="flex items-center gap-2">
              {polarityIcon}
              <Badge variant="outline">{claim.claim_type}</Badge>
              <span className="font-medium">{claim.field}</span>
              {statusBadge[claim.status]}
            </div>

            {/* Value */}
            <div className="text-lg">
              {claim.status === "modified" && claim.modified_value
                ? String(claim.modified_value)
                : String(claim.value)}
            </div>

            {/* Confidence */}
            <div className="flex items-center gap-2 text-sm">
              <span className="text-muted-foreground">Confidence:</span>
              <span className="font-medium">
                {(
                  (claim.status === "modified" && claim.modified_confidence
                    ? claim.modified_confidence
                    : claim.confidence) * 100
                ).toFixed(0)}
                %
              </span>
            </div>

            {/* Source */}
            {claim.locator && (
              <div className="text-sm text-muted-foreground">
                <span className="font-medium">Source:</span> {claim.locator}
              </div>
            )}

            {/* Quote */}
            {claim.quote && (
              <div className="text-sm italic border-l-2 pl-2 text-muted-foreground">
                &ldquo;{claim.quote}&rdquo;
              </div>
            )}

            {/* Rationale */}
            <div className="text-sm text-muted-foreground">
              <span className="font-medium">Rationale:</span> {claim.rationale}
            </div>
          </div>

          {/* Actions */}
          {claim.status === "pending" && (
            <div className="flex flex-col gap-2">
              <Button size="sm" variant="outline" onClick={onApprove}>
                <Check className="h-4 w-4 mr-1" />
                Approve
              </Button>
              <Button size="sm" variant="outline" onClick={onEdit}>
                <Edit2 className="h-4 w-4 mr-1" />
                Edit
              </Button>
              <Button size="sm" variant="outline" onClick={onReject}>
                <X className="h-4 w-4 mr-1" />
                Reject
              </Button>
            </div>
          )}

          {claim.status !== "pending" && (
            <Button size="sm" variant="ghost" onClick={onEdit}>
              <Edit2 className="h-4 w-4" />
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
