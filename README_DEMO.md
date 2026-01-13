# JURIS-AGI VC Demo Guide

**Decision Support Tool for Venture Capital Investment Analysis**

> **IMPORTANT DISCLAIMER**: JURIS-AGI is a decision *support* tool, not a decision *maker*.
> All outputs are intended to augment human judgment, not replace it. Investment decisions
> remain the sole responsibility of the investment committee.

---

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- (Optional) OpenAI API key for LLM extraction features

### Setup

```bash
# 1. Clone and enter directory
cd JURIS-AGI

# 2. Install Python dependencies
pip install -e ".[dev]"

# 3. Install web dependencies
cd web && npm install && cd ..

# 4. Generate demo data
python demo/seed_data.py
```

### Running the Demo

**Terminal 1 - Backend API:**
```bash
python -m juris_agi.api.local_server
# Server runs on http://localhost:8000
```

**Terminal 2 - Web UI:**
```bash
cd web && npm run dev
# UI runs on http://localhost:3000
```

---

## What This Demo Shows

JURIS-AGI demonstrates **evidence-based investment analysis with counterfactual reasoning**:

1. **Structured Evidence Graphs** - Claims about a company organized by type (traction, team, market, etc.) with confidence scores and polarity (supportive/risk/neutral)

2. **Counterfactual Analysis** - "What would need to be true for this decision to flip?" Identifies which claims are most critical to the investment thesis

3. **Robustness Scoring** - How stable is the decision? What's the margin of safety before the recommendation changes?

4. **Decision Transparency** - Full audit trail of how claims contribute to the final recommendation

5. **Human-in-the-Loop Extraction** - Optional LLM-based extraction from documents, with mandatory human review before claims enter the evidence graph

---

## 5-Minute IC Walkthrough Script

### Setup (30 seconds)

1. Ensure both backend and frontend are running
2. Open browser to `http://localhost:3000`
3. Have the HealthBridge deal JSON ready: `demo/data/healthbridge-2024.json`

### Part 1: Historical Validation (1 minute)

**Narration:** "Let me first show you how JURIS-AGI performs on deals where we already know the outcome."

1. Navigate to **Workspace** tab
2. Load `demo/data/cloudmetrics-2023.json` (paste evidence graph)
3. Click **Analyze**
4. Show the result: "This was CloudMetrics, a clear INVEST decision we made in 2023. Strong SaaS metrics, experienced team. Notice the system correctly identifies it as INVEST with high confidence."

5. Quickly load `demo/data/cryptolend-2023.json`
6. Show result: "CryptoLend was a clear PASS - regulatory risk, inexperienced team. The system flags these exact concerns."

**Key point:** "The system's reasoning aligns with our historical judgment."

### Part 2: Live Deal Analysis (2 minutes)

**Narration:** "Now let's look at a current deal in our pipeline."

1. Load `demo/data/healthbridge-2024.json`
2. Click **Analyze**
3. Wait for results

**Walk through the output:**

- **Decision & Confidence:** "The system recommends [INVEST/PASS/DEFER] with X% confidence. Note this is lower confidence than our historical examples - that's appropriate given the mixed signals."

- **Critical Claims:** "These are the claims that matter most to the decision. Notice [Team Quality] and [Regulatory Risk] are flagged as critical."

- **Robustness Score:** "This tells us how stable the decision is. A score of 0.65 means moderate sensitivity - some uncertainty is baked in."

### Part 3: Counterfactual Reasoning (1 minute)

**Narration:** "Here's where it gets interesting - what would need to change for the decision to flip?"

1. Scroll to **Decision Flips** section
2. Show example: "If regulatory clearance timeline extended from 12 to 24 months, the decision flips from INVEST to DEFER."

3. Show another: "If they hired a full-time Chief Medical Officer, the domain expertise gap closes and confidence increases."

**Key point:** "This isn't magic - it's structured sensitivity analysis. It tells us exactly what to dig into during diligence."

### Part 4: Audit Trail (30 seconds)

1. Navigate to **Audit** tab
2. Show trace entries: "Every claim, every inference is logged. Full transparency for LP reporting and internal review."

**Narration:** "This is the audit trail. When an LP asks 'why did you invest in HealthBridge?', we can show the exact evidence and reasoning."

### Wrap-up (30 seconds)

**Key messages:**

1. "This is decision *support*, not decision making. The IC makes the final call."

2. "The value is in structured thinking - forcing us to be explicit about our evidence and identifying where we're most uncertain."

3. "Counterfactual reasoning helps us focus diligence on what actually matters."

4. "Everything is auditable. No black boxes."

---

## Demo Data

### Historical Deals (Known Outcomes)

| Company | Sector | Stage | Outcome | Key Factors |
|---------|--------|-------|---------|-------------|
| CloudMetrics Inc. | Enterprise SaaS | Series A | INVEST | Strong unit economics, proven team |
| CryptoLend Finance | Crypto/DeFi | Seed | PASS | Regulatory risk, weak fundamentals |
| QuantumSense AI | Deep Tech / AI | Series A | INVEST | Defensible IP, enterprise traction |

### Ambiguous Deal (Live Demo)

| Company | Sector | Stage | Outcome |
|---------|--------|-------|---------|
| HealthBridge AI | Healthcare / AI | Series A | **TBD** |

**HealthBridge AI** has intentionally mixed signals:
- Strong technical team, but limited healthcare domain expertise
- Early traction with 3 pilot clinics, high NPS
- Large TAM ($12B) but long sales cycles (6-9 months)
- Regulatory path exists (510(k)) but timeline uncertain
- Faces well-funded competitors (Microsoft Nuance)

The system should produce a moderate-confidence recommendation with clear identification of the critical uncertainties.

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/vc/analyze` | POST | Submit evidence graph for analysis |
| `/jobs/{id}` | GET | Get analysis job status/result |
| `/extract` | POST | Extract claims from document |
| `/extract/review` | POST | Review proposed claim |
| `/extract/merge` | POST | Merge approved claims |
| `/health` | GET | Health check |

---

## Configuration

### Environment Variables

```bash
# Optional - for LLM-based extraction
OPENAI_API_KEY=sk-...

# API server port (default: 8000)
API_PORT=8000

# Fixed seed for reproducibility (default: 42)
JURIS_RANDOM_SEED=42
```

### Reproducibility

Demo runs use a fixed random seed (42) to ensure consistent results across presentations. The same evidence graph will always produce the same analysis output.

---

## Troubleshooting

**Backend won't start:**
```bash
# Check Python version
python --version  # Should be 3.10+

# Reinstall dependencies
pip install -e ".[dev]"
```

**Frontend won't start:**
```bash
# Check Node version
node --version  # Should be 18+

# Clear cache and reinstall
cd web && rm -rf node_modules .next && npm install
```

**Analysis returns error:**
- Check backend logs for stack trace
- Verify evidence graph JSON is valid
- Ensure all required claim fields are present

---

## Limitations & Caveats

1. **Not Financial Advice** - This tool does not constitute investment advice. All investment decisions require human judgment and professional due diligence.

2. **Confidence ≠ Certainty** - Confidence scores reflect model uncertainty, not probability of success.

3. **Garbage In, Garbage Out** - Analysis quality depends on evidence quality. Unverified claims will produce unreliable outputs.

4. **Demo Mode Limitations** - The demo uses simulated analysis. Production deployment requires additional validation.

5. **No Guarantee of Correctness** - The system can and will make mistakes. It is a tool to augment, not replace, human expertise.

---

## Contact

For issues or questions about this demo, please contact the JURIS-AGI development team.
