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
