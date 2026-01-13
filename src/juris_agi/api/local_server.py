"""
Local PoC server for JURIS-AGI.

Supports two modes:
1. Sync mode (default): No Redis, direct in-process execution
2. Async mode: Redis queue with worker (docker-compose)

Run with:
    python -m juris_agi.api.local_server
"""

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .models import (
    SolveRequest,
    SolveResponse,
    JobStatus,
    JobResult,
    HealthResponse,
    ErrorResponse,
    PredictionResult,
    GridData,
    BudgetConfig,
)
from .local_config import get_local_config, LocalPoCConfig, is_local_poc_mode

# Evidence extraction models
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

# Optional Redis
try:
    import redis
    from rq import Queue
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None
    Queue = None

# Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# In-memory job storage for sync mode
_jobs: Dict[str, Dict[str, Any]] = {}

# Global state
_redis_client = None
_job_queue = None


def get_redis():
    """Get Redis client if enabled."""
    global _redis_client
    config = get_local_config()

    if not config.redis_enabled or not REDIS_AVAILABLE:
        return None

    if _redis_client is None:
        try:
            _redis_client = redis.from_url(config.redis_url, decode_responses=True)
            _redis_client.ping()
            logger.info("Connected to Redis")
        except Exception as e:
            logger.warning(f"Redis not available: {e}")
            _redis_client = None

    return _redis_client


def get_queue():
    """Get RQ queue if enabled."""
    global _job_queue
    config = get_local_config()

    if not config.redis_enabled:
        return None

    if _job_queue is None:
        redis_client = get_redis()
        if redis_client and REDIS_AVAILABLE:
            _job_queue = Queue("juris_default", connection=redis_client)

    return _job_queue


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    config = get_local_config()
    logger.info("Starting JURIS-AGI Local PoC Server...")
    logger.info(f"Mode: {'sync' if config.sync_mode else 'async'}")
    logger.info(f"Runs directory: {config.runs_dir}")
    logger.info(f"GPU enabled: {config.gpu_enabled}")

    # Ensure directories exist
    config.ensure_dirs()

    if config.redis_enabled:
        get_redis()

    yield

    logger.info("Shutting down JURIS-AGI Local PoC Server...")
    global _redis_client
    if _redis_client:
        _redis_client.close()
        _redis_client = None


def create_local_app() -> FastAPI:
    """Create the local PoC FastAPI application."""
    app = FastAPI(
        title="JURIS-AGI Local PoC",
        description="Local Proof-of-Concept for ARC-style abstract reasoning (CPU-only)",
        version="0.1.0-poc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app


app = create_local_app()


# =============================================================================
# Health Endpoint
# =============================================================================

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check with local PoC status."""
    config = get_local_config()
    redis_client = get_redis()

    redis_connected = False
    if redis_client:
        try:
            redis_client.ping()
            redis_connected = True
        except Exception:
            pass

    return {
        "status": "healthy",
        "version": "0.1.0-poc",
        "mode": "local_poc",
        "execution": "sync" if config.sync_mode else "async",
        "gpu_available": False,  # Always false in PoC
        "gpu_enabled": config.gpu_enabled,
        "redis_connected": redis_connected,
        "config": config.to_dict(),
    }


# =============================================================================
# Solve Endpoint
# =============================================================================

@app.post("/solve", response_model=SolveResponse, tags=["Solve"])
async def solve_task(request: SolveRequest):
    """
    Submit an ARC task for solving.

    In sync mode: Returns immediately with result.
    In async mode: Returns job_id for polling.
    """
    config = get_local_config()

    # Validate grids against limits
    for i, pair in enumerate(request.task.train):
        valid, error = config.validate_grid(pair.input.data)
        if not valid:
            raise HTTPException(400, f"Train pair {i} input: {error}")
        valid, error = config.validate_grid(pair.output.data)
        if not valid:
            raise HTTPException(400, f"Train pair {i} output: {error}")

    for i, pair in enumerate(request.task.test):
        valid, error = config.validate_grid(pair.input.data)
        if not valid:
            raise HTTPException(400, f"Test pair {i} input: {error}")

    # Generate job ID
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    task_id = request.task_id or f"task_{uuid.uuid4().hex[:8]}"
    created_at = datetime.utcnow()

    # Apply PoC limits to budget
    budget = BudgetConfig(
        max_time_seconds=min(request.budget.max_time_seconds, config.max_runtime_seconds),
        max_iterations=min(request.budget.max_iterations, config.max_search_expansions),
        beam_width=min(request.budget.beam_width, 50),
        max_depth=min(request.budget.max_depth, config.max_program_depth),
    )

    job_data = {
        "job_id": job_id,
        "task_id": task_id,
        "status": JobStatus.PENDING.value,
        "created_at": created_at.isoformat(),
        "task": request.task.model_dump(),
        "budget": budget.model_dump(),
        "use_neural": False,  # Always false in CPU PoC
        "return_trace": request.return_trace,
    }

    if config.sync_mode:
        # Synchronous execution - run immediately
        logger.info(f"Running job {job_id} synchronously...")
        result_data = run_solve_sync(job_id, job_data, config)
        _jobs[job_id] = result_data

        return SolveResponse(
            job_id=job_id,
            status=JobStatus(result_data["status"]),
            created_at=created_at,
            estimated_time_seconds=result_data.get("runtime_seconds", 0),
        )
    else:
        # Async mode - queue job
        redis_client = get_redis()
        queue = get_queue()

        if redis_client and queue:
            try:
                redis_client.setex(
                    f"juris:job:{job_id}",
                    3600,
                    json.dumps(job_data),
                )
                queue.enqueue(
                    "juris_agi.api.worker.process_job",
                    job_id,
                    job_timeout=int(config.max_runtime_seconds) + 30,
                )
            except Exception as e:
                logger.error(f"Failed to enqueue: {e}")
                raise HTTPException(503, "Failed to queue job")
        else:
            # Fallback to sync if Redis not available
            logger.warning("Redis not available, running sync")
            result_data = run_solve_sync(job_id, job_data, config)
            _jobs[job_id] = result_data

        return SolveResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            created_at=created_at,
            estimated_time_seconds=budget.max_time_seconds,
        )


# =============================================================================
# Job Status Endpoint
# =============================================================================

@app.get("/jobs/{job_id}", response_model=JobResult, tags=["Jobs"])
async def get_job(
    job_id: str,
    include_trace: bool = Query(default=False),
):
    """Get job status and results."""
    config = get_local_config()

    job_data = None

    # Check Redis first
    redis_client = get_redis()
    if redis_client:
        try:
            raw = redis_client.get(f"juris:job:{job_id}")
            if raw:
                job_data = json.loads(raw)
        except Exception:
            pass

    # Check in-memory storage
    if job_data is None:
        job_data = _jobs.get(job_id)

    if job_data is None:
        raise HTTPException(404, f"Job {job_id} not found")

    status = JobStatus(job_data.get("status", "pending"))

    result = JobResult(
        job_id=job_id,
        status=status,
        task_id=job_data.get("task_id"),
        created_at=datetime.fromisoformat(job_data["created_at"]),
        started_at=datetime.fromisoformat(job_data["started_at"]) if job_data.get("started_at") else None,
        completed_at=datetime.fromisoformat(job_data["completed_at"]) if job_data.get("completed_at") else None,
        runtime_seconds=job_data.get("runtime_seconds"),
        success=job_data.get("success", False),
        program=job_data.get("program"),
        robustness_score=job_data.get("robustness_score"),
        regime=job_data.get("regime"),
        synthesis_iterations=job_data.get("synthesis_iterations"),
        error_message=job_data.get("error_message"),
        trace_url=job_data.get("trace_url"),
        result_url=job_data.get("result_url"),
    )

    # Add predictions
    if "predictions" in job_data:
        for i, pred in enumerate(job_data["predictions"]):
            result.predictions.append(PredictionResult(
                test_index=i,
                prediction=GridData(data=pred["data"]),
                confidence=pred.get("confidence", 0.0),
            ))

    # Include trace data if requested
    if include_trace and job_data.get("trace_path"):
        try:
            with open(job_data["trace_path"]) as f:
                result.trace_data = json.load(f)
        except Exception:
            pass

    return result


# =============================================================================
# Synchronous Solver
# =============================================================================

def run_solve_sync(job_id: str, job_data: dict, config: LocalPoCConfig) -> dict:
    """Run solver synchronously and return updated job data."""
    from ..core.types import ARCTask, ARCPair, Grid
    from ..controller.router import MetaController, ControllerConfig

    job_data["status"] = JobStatus.RUNNING.value
    job_data["started_at"] = datetime.utcnow().isoformat()

    try:
        # Parse task
        task_payload = job_data["task"]
        train_pairs = []
        for pair in task_payload["train"]:
            input_grid = Grid.from_list(pair["input"]["data"])
            output_grid = Grid.from_list(pair["output"]["data"])
            train_pairs.append(ARCPair(input=input_grid, output=output_grid))

        test_pairs = []
        for pair in task_payload.get("test", []):
            input_grid = Grid.from_list(pair["input"]["data"])
            output_grid = None
            if pair.get("output"):
                output_grid = Grid.from_list(pair["output"]["data"])
            test_pairs.append(ARCPair(input=input_grid, output=output_grid))

        task = ARCTask(
            task_id=job_data["task_id"],
            train=train_pairs,
            test=test_pairs,
        )

        # Configure controller with PoC limits
        budget = job_data["budget"]
        controller_config = ControllerConfig(
            max_synthesis_depth=budget["max_depth"],
            beam_width=budget["beam_width"],
            max_synthesis_iterations=budget["max_iterations"],
        )
        controller = MetaController(controller_config)

        # Solve with timeout
        start_time = time.time()
        result = controller.solve(task)
        runtime = time.time() - start_time

        # Update job data
        job_data["status"] = JobStatus.COMPLETED.value
        job_data["completed_at"] = datetime.utcnow().isoformat()
        job_data["runtime_seconds"] = runtime
        job_data["success"] = result.success

        if result.success:
            job_data["program"] = result.audit_trace.program_source
            job_data["robustness_score"] = result.audit_trace.robustness_score
            job_data["synthesis_iterations"] = result.audit_trace.synthesis_iterations

            # Generate predictions
            predictions = []
            for pred in result.predictions:
                predictions.append({
                    "data": pred.data.tolist(),
                    "confidence": result.audit_trace.robustness_score or 0.0,
                })
            job_data["predictions"] = predictions

            # Save trace with human-readable summary
            if job_data.get("return_trace"):
                trace_path = save_trace_with_summary(
                    job_id,
                    job_data["task_id"],
                    result.audit_trace,
                    config,
                    success=result.success,
                )
                job_data["trace_path"] = str(trace_path)
                job_data["trace_url"] = f"file://{trace_path}"

            # Save result
            result_path = save_result(job_id, job_data["task_id"], job_data, config)
            job_data["result_url"] = f"file://{result_path}"

        else:
            job_data["error_message"] = result.error_message or "Synthesis failed"

        logger.info(f"Job {job_id} completed: success={result.success}, runtime={runtime:.2f}s")

    except Exception as e:
        logger.exception(f"Job {job_id} failed")
        job_data["status"] = JobStatus.FAILED.value
        job_data["completed_at"] = datetime.utcnow().isoformat()
        job_data["error_message"] = str(e)

    return job_data


# =============================================================================
# Trace & Result Storage
# =============================================================================

def save_trace_with_summary(
    job_id: str,
    task_id: str,
    audit_trace,
    config: LocalPoCConfig,
    success: bool = True,
) -> Path:
    """Save trace with human-readable summary."""
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    trace_dir = config.traces_dir / date_str / task_id
    trace_dir.mkdir(parents=True, exist_ok=True)

    trace_path = trace_dir / f"{job_id}.json"

    # Build trace with summary
    trace_data = audit_trace.to_dict()

    # Add human-readable summary
    summary = {
        "job_id": job_id,
        "task_id": task_id,
        "timestamp": datetime.utcnow().isoformat(),
        "success": success,
        "runtime_seconds": getattr(audit_trace, 'runtime_seconds', None),

        # Key synthesis info
        "final_program": audit_trace.program_source,
        "robustness_score": audit_trace.robustness_score,
        "synthesis_iterations": audit_trace.synthesis_iterations,

        # Inferred invariants (if available)
        "inferred_invariants": [],

        # Candidate programs attempted
        "candidate_programs_tried": [],

        # Refinement steps
        "refinement_steps": [],
    }

    # Extract invariants from trace entries if available
    if hasattr(audit_trace, 'entries'):
        for entry in audit_trace.entries:
            entry_dict = entry.to_dict() if hasattr(entry, 'to_dict') else entry
            if isinstance(entry_dict, dict):
                # Look for invariant-related entries
                if 'invariants' in str(entry_dict).lower():
                    summary["inferred_invariants"].append(str(entry_dict))
                # Look for candidate programs
                if entry_dict.get('step_type') == 'candidate' or 'candidate' in str(entry_dict).lower():
                    summary["candidate_programs_tried"].append(entry_dict.get('program', str(entry_dict)))
                # Look for refinement steps
                if 'refine' in str(entry_dict).lower():
                    summary["refinement_steps"].append(str(entry_dict))

    # Combine trace data with summary
    output = {
        "summary": summary,
        "full_trace": trace_data,
    }

    with open(trace_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"Saved trace to {trace_path}")
    return trace_path


def save_result(
    job_id: str,
    task_id: str,
    job_data: dict,
    config: LocalPoCConfig,
) -> Path:
    """Save job result to local filesystem."""
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    result_dir = config.results_dir / date_str / task_id
    result_dir.mkdir(parents=True, exist_ok=True)

    result_path = result_dir / f"{job_id}.json"

    result_data = {
        "job_id": job_id,
        "task_id": task_id,
        "status": job_data.get("status"),
        "success": job_data.get("success"),
        "predictions": job_data.get("predictions", []),
        "program": job_data.get("program"),
        "robustness_score": job_data.get("robustness_score"),
        "synthesis_iterations": job_data.get("synthesis_iterations"),
        "runtime_seconds": job_data.get("runtime_seconds"),
        "created_at": job_data.get("created_at"),
        "completed_at": job_data.get("completed_at"),
    }

    with open(result_path, "w") as f:
        json.dump(result_data, f, indent=2, default=str)

    logger.info(f"Saved result to {result_path}")
    return result_path


# =============================================================================
# Evidence Extraction Models
# =============================================================================

class DocumentTypeEnum(str, Enum):
    """Supported document types for extraction."""
    PITCH_DECK = "pitch_deck"
    FINANCIAL_MODEL = "financial_model"
    TECH_DESCRIPTION = "tech_description"
    IC_MEMO = "ic_memo"


class ExtractionRequest(BaseModel):
    """Request to extract claims from a document."""
    doc_id: str = Field(..., description="Unique identifier for the document")
    doc_type: DocumentTypeEnum = Field(..., description="Type of document")
    content: str = Field(..., description="Text content of the document")
    company_id: Optional[str] = Field(None, description="Associated company ID")


class ClaimStatusEnum(str, Enum):
    """Status of a proposed claim."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"


class ProposedClaimResponse(BaseModel):
    """A proposed claim from extraction."""
    proposal_id: str
    claim_type: str
    field: str
    value: Any
    confidence: float
    polarity: str
    locator: Optional[str] = None
    quote: Optional[str] = None
    rationale: str
    status: ClaimStatusEnum = ClaimStatusEnum.PENDING


class ExtractionResponse(BaseModel):
    """Response from document extraction."""
    doc_id: str
    doc_type: str
    proposed_claims: List[ProposedClaimResponse] = []
    extraction_time_seconds: float = 0.0
    errors: List[str] = []
    success: bool = True


class ClaimReviewRequest(BaseModel):
    """Request to review a proposed claim."""
    proposal_id: str
    action: str = Field(..., description="approve, reject, or modify")
    modified_value: Optional[Any] = None
    modified_confidence: Optional[float] = None
    modified_polarity: Optional[str] = None
    reviewer_notes: Optional[str] = None


class MergeClaimsRequest(BaseModel):
    """Request to merge approved claims into evidence graph."""
    company_id: str
    proposal_ids: List[str]


# In-memory storage for extraction results (for demo/PoC)
_extraction_results: Dict[str, Dict[str, Any]] = {}
_proposed_claims: Dict[str, Dict[str, Any]] = {}


# =============================================================================
# Evidence Extraction Endpoints
# =============================================================================

@app.post("/extract", response_model=ExtractionResponse, tags=["Evidence Extraction"])
async def extract_claims(request: ExtractionRequest):
    """
    Extract evidence claims from a document using LLM.

    Returns proposed claims that require human review before
    being added to the evidence graph.
    """
    from ..evidence.extractors import (
        get_extractor,
        ExtractionConfig,
        create_mock_llm_fn,
    )

    # Create extractor with mock LLM (for PoC without API key)
    # In production, this would use a real LLM function
    config = ExtractionConfig(
        enabled=True,
        min_confidence=0.3,
        max_claims_per_type=10,
    )

    # Use mock LLM for demo (replace with real LLM in production)
    llm_fn = create_mock_llm_fn()

    extractor = get_extractor(request.doc_type.value, config, llm_fn)

    if extractor is None:
        return ExtractionResponse(
            doc_id=request.doc_id,
            doc_type=request.doc_type.value,
            errors=[f"Unknown document type: {request.doc_type}"],
            success=False,
        )

    # Run extraction
    result = extractor.extract(request.content, request.doc_id)

    # Store proposals for later review
    for proposal in result.proposed_claims:
        _proposed_claims[proposal.proposal_id] = proposal.to_dict()
        _proposed_claims[proposal.proposal_id]["company_id"] = request.company_id

    # Store extraction result
    _extraction_results[request.doc_id] = result.to_dict()

    # Convert to response
    proposed_claims = []
    for pc in result.proposed_claims:
        proposed_claims.append(ProposedClaimResponse(
            proposal_id=pc.proposal_id,
            claim_type=pc.claim_type.value,
            field=pc.field,
            value=pc.value,
            confidence=pc.confidence,
            polarity=pc.polarity.value,
            locator=pc.source.locator if pc.source else None,
            quote=pc.source.quote if pc.source else None,
            rationale=pc.rationale,
            status=ClaimStatusEnum(pc.status.value),
        ))

    return ExtractionResponse(
        doc_id=result.doc_id,
        doc_type=result.doc_type,
        proposed_claims=proposed_claims,
        extraction_time_seconds=result.extraction_time_seconds,
        errors=result.errors,
        success=result.success,
    )


@app.get("/extract/{doc_id}", response_model=ExtractionResponse, tags=["Evidence Extraction"])
async def get_extraction_result(doc_id: str):
    """Get extraction result for a document."""
    if doc_id not in _extraction_results:
        raise HTTPException(404, f"Extraction result for {doc_id} not found")

    result_data = _extraction_results[doc_id]

    # Rebuild proposed claims with current status
    proposed_claims = []
    for pc_data in result_data.get("proposed_claims", []):
        proposal_id = pc_data["proposal_id"]
        # Get latest status from proposals store
        if proposal_id in _proposed_claims:
            pc_data = _proposed_claims[proposal_id]

        proposed_claims.append(ProposedClaimResponse(
            proposal_id=pc_data["proposal_id"],
            claim_type=pc_data["claim_type"],
            field=pc_data["field"],
            value=pc_data["value"],
            confidence=pc_data["confidence"],
            polarity=pc_data["polarity"],
            locator=pc_data.get("source", {}).get("locator") if pc_data.get("source") else None,
            quote=pc_data.get("source", {}).get("quote") if pc_data.get("source") else None,
            rationale=pc_data.get("rationale", ""),
            status=ClaimStatusEnum(pc_data.get("status", "pending")),
        ))

    return ExtractionResponse(
        doc_id=result_data["doc_id"],
        doc_type=result_data["doc_type"],
        proposed_claims=proposed_claims,
        extraction_time_seconds=result_data.get("extraction_time_seconds", 0),
        errors=result_data.get("errors", []),
        success=len(result_data.get("errors", [])) == 0,
    )


@app.post("/extract/review", tags=["Evidence Extraction"])
async def review_claim(request: ClaimReviewRequest):
    """
    Review a proposed claim (approve, reject, or modify).

    Human review is required before claims can be merged.
    """
    if request.proposal_id not in _proposed_claims:
        raise HTTPException(404, f"Proposal {request.proposal_id} not found")

    proposal = _proposed_claims[request.proposal_id]

    if request.action == "approve":
        proposal["status"] = "approved"
        proposal["reviewer_notes"] = request.reviewer_notes
    elif request.action == "reject":
        proposal["status"] = "rejected"
        proposal["reviewer_notes"] = request.reviewer_notes
    elif request.action == "modify":
        proposal["status"] = "modified"
        if request.modified_value is not None:
            proposal["modified_value"] = request.modified_value
        if request.modified_confidence is not None:
            proposal["modified_confidence"] = request.modified_confidence
        if request.modified_polarity is not None:
            proposal["modified_polarity"] = request.modified_polarity
        proposal["reviewer_notes"] = request.reviewer_notes
    else:
        raise HTTPException(400, f"Invalid action: {request.action}")

    proposal["reviewed_at"] = datetime.utcnow().isoformat()
    _proposed_claims[request.proposal_id] = proposal

    return {"status": "success", "proposal": proposal}


@app.post("/extract/merge", tags=["Evidence Extraction"])
async def merge_claims(request: MergeClaimsRequest):
    """
    Merge approved claims into evidence graph.

    Only approved or modified claims can be merged.
    Human-approved claims are NEVER overwritten.
    """
    from ..evidence.extractors import ProposedClaim, ExtractionStatus
    from ..evidence.schema import EvidenceGraph, Claim

    merged = []
    errors = []

    for proposal_id in request.proposal_ids:
        if proposal_id not in _proposed_claims:
            errors.append(f"Proposal {proposal_id} not found")
            continue

        proposal_data = _proposed_claims[proposal_id]
        status = proposal_data.get("status", "pending")

        if status not in ("approved", "modified"):
            errors.append(f"Proposal {proposal_id} is not approved (status: {status})")
            continue

        # Convert to claim
        try:
            proposal = ProposedClaim.from_dict(proposal_data)
            claim = proposal.to_claim()
            merged.append(claim.to_dict())

            # Mark as merged
            proposal_data["status"] = "merged"
            _proposed_claims[proposal_id] = proposal_data

        except Exception as e:
            errors.append(f"Failed to convert proposal {proposal_id}: {str(e)}")

    return {
        "company_id": request.company_id,
        "merged_count": len(merged),
        "merged_claims": merged,
        "errors": errors,
    }


@app.get("/extract/supported-types", tags=["Evidence Extraction"])
async def get_supported_document_types():
    """Get list of supported document types for extraction."""
    from ..evidence.extractors import ExtractorRegistry

    return {
        "types": ExtractorRegistry.list_supported_types(),
        "aliases": ExtractorRegistry.get_type_aliases(),
    }


# =============================================================================
# Decision Report Endpoints
# =============================================================================

# In-memory storage for analysis results (for demo/PoC)
_analysis_results: Dict[str, Dict[str, Any]] = {}


@app.post("/vc/analyze", tags=["VC Analysis"])
async def analyze_evidence_graph(evidence_graph: Dict[str, Any]):
    """
    Submit an evidence graph for VC decision analysis.

    Returns a job_id that can be used to retrieve the report.
    """
    from ..vc.decision_analysis import DecisionAnalyzer, DecisionOutcome
    from ..vc.trace import VCDecisionTrace
    from ..evidence.schema import EvidenceGraph, Claim, Polarity as EvidencePolarity
    from ..evidence.ontology import ClaimType
    from ..config import get_config

    config = get_config()
    job_id = f"vc_{uuid.uuid4().hex[:12]}"

    try:
        # Parse evidence graph
        company_id = evidence_graph.get("company_id", "unknown")
        claims_data = evidence_graph.get("claims", [])

        # Convert to internal format
        claims = []
        for c in claims_data:
            try:
                claim_type = ClaimType(c.get("claim_type", "company_identity"))
            except ValueError:
                claim_type = ClaimType.COMPANY_IDENTITY

            try:
                polarity = EvidencePolarity(c.get("polarity", "neutral"))
            except ValueError:
                polarity = EvidencePolarity.NEUTRAL

            claims.append(Claim(
                claim_type=claim_type,
                field=c.get("field", ""),
                value=c.get("value"),
                confidence=c.get("confidence", 0.5),
                polarity=polarity,
            ))

        graph = EvidenceGraph(company_id=company_id, claims=claims)

        # Define simple decision function
        def decision_fn(g: EvidenceGraph):
            supportive = sum(1 for c in g.claims if c.polarity == EvidencePolarity.SUPPORTIVE)
            risk = sum(1 for c in g.claims if c.polarity == EvidencePolarity.RISK)
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

        # Run analysis
        analyzer = DecisionAnalyzer(
            decision_fn=decision_fn,
            seed=config.random_seed,
            num_counterfactuals=config.num_counterfactuals,
        )
        result = analyzer.analyze(graph)

        # Create trace
        trace = VCDecisionTrace.from_analysis_result(company_id, result)

        # Store result
        _analysis_results[job_id] = {
            "job_id": job_id,
            "company_id": company_id,
            "evidence_graph": evidence_graph,
            "trace": trace.to_dict(),
            "decision": result.decision.value,
            "confidence": result.confidence,
            "status": "completed",
            "created_at": datetime.utcnow().isoformat(),
        }

        return {
            "job_id": job_id,
            "status": "completed",
            "decision": result.decision.value,
            "confidence": result.confidence,
        }

    except Exception as e:
        logger.exception(f"Analysis failed for job {job_id}")
        _analysis_results[job_id] = {
            "job_id": job_id,
            "status": "failed",
            "error": str(e),
            "created_at": datetime.utcnow().isoformat(),
        }
        raise HTTPException(500, f"Analysis failed: {str(e)}")


@app.get("/vc/jobs/{job_id}", tags=["VC Analysis"])
async def get_vc_job(job_id: str):
    """Get VC analysis job status and result."""
    if job_id not in _analysis_results:
        raise HTTPException(404, f"Job {job_id} not found")

    return _analysis_results[job_id]


@app.get("/vc/jobs/{job_id}/report", tags=["VC Analysis"])
async def get_decision_report(
    job_id: str,
    format: str = Query(default="html", description="Report format: html, md, or pdf"),
):
    """
    Generate and return a formal decision report.

    Args:
        job_id: The analysis job ID
        format: Output format (html, md, pdf)

    Returns:
        Rendered report in requested format
    """
    from ..report import generate_report, render_html, render_markdown, render_pdf
    from ..config import get_config
    from fastapi.responses import Response

    if job_id not in _analysis_results:
        raise HTTPException(404, f"Job {job_id} not found")

    job_data = _analysis_results[job_id]

    if job_data.get("status") != "completed":
        raise HTTPException(400, f"Job {job_id} is not completed (status: {job_data.get('status')})")

    config = get_config()

    # Generate report
    report = generate_report(
        evidence_graph=job_data["evidence_graph"],
        trace=job_data["trace"],
        final_decision=job_data["decision"],
        seed=config.random_seed,
    )

    # Render in requested format
    format_lower = format.lower()
    if format_lower == "html":
        content = render_html(report)
        return Response(content=content, media_type="text/html")
    elif format_lower in ("md", "markdown"):
        content = render_markdown(report)
        return Response(content=content, media_type="text/markdown")
    elif format_lower == "pdf":
        content = render_pdf(report)
        if isinstance(content, bytes):
            return Response(content=content, media_type="application/pdf")
        else:
            # Fallback to HTML if PDF rendering not available
            return Response(content=content, media_type="text/html")
    else:
        raise HTTPException(400, f"Unsupported format: {format}. Use html, md, or pdf.")


@app.post("/vc/report/generate", tags=["VC Analysis"])
async def generate_report_direct(
    evidence_graph: Dict[str, Any],
    trace: Dict[str, Any],
    decision: str,
    format: str = Query(default="html"),
):
    """
    Generate a report directly from evidence graph and trace.

    This endpoint bypasses the job system for direct report generation.
    """
    from ..report import generate_report, render_html, render_markdown, render_pdf
    from ..config import get_config
    from fastapi.responses import Response

    config = get_config()

    report = generate_report(
        evidence_graph=evidence_graph,
        trace=trace,
        final_decision=decision,
        seed=config.random_seed,
    )

    format_lower = format.lower()
    if format_lower == "html":
        content = render_html(report)
        return Response(content=content, media_type="text/html")
    elif format_lower in ("md", "markdown"):
        content = render_markdown(report)
        return Response(content=content, media_type="text/markdown")
    elif format_lower == "pdf":
        content = render_pdf(report)
        if isinstance(content, bytes):
            return Response(content=content, media_type="application/pdf")
        else:
            return Response(content=content, media_type="text/html")
    else:
        raise HTTPException(400, f"Unsupported format: {format}")


# =============================================================================
# Error Handlers
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": f"HTTP_{exc.status_code}", "message": exc.detail},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.exception("Unexpected error")
    return JSONResponse(
        status_code=500,
        content={"error": "INTERNAL_ERROR", "message": "An unexpected error occurred"},
    )


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Run the local PoC server."""
    import uvicorn

    config = get_local_config()

    print("\n" + "=" * 60)
    print("JURIS-AGI Local Proof-of-Concept Server")
    print("=" * 60)
    print(f"Mode:        {'Sync (no Redis)' if config.sync_mode else 'Async (with Redis)'}")
    print(f"Host:        {config.host}:{config.port}")
    print(f"Runs dir:    {config.runs_dir}")
    print(f"GPU:         {'Enabled' if config.gpu_enabled else 'Disabled (CPU-only)'}")
    print(f"Max grid:    {config.max_grid_size}x{config.max_grid_size}")
    print(f"Max runtime: {config.max_runtime_seconds}s")
    print("=" * 60)
    print("\nEndpoints:")
    print(f"  POST http://{config.host}:{config.port}/solve")
    print(f"  GET  http://{config.host}:{config.port}/jobs/{{job_id}}")
    print(f"  GET  http://{config.host}:{config.port}/health")
    print(f"\nAPI docs: http://{config.host}:{config.port}/docs")
    print("=" * 60 + "\n")

    uvicorn.run(
        "juris_agi.api.local_server:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level="info",
    )


if __name__ == "__main__":
    main()
