"""
FastAPI server for JURIS-AGI cloud deployment.

Endpoints:
- POST /solve : Submit a task for solving
- GET /jobs/{job_id} : Get job status and results
- GET /health : Health check
"""

import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
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
)
from .config import APIConfig

# Optional Redis import
try:
    import redis
    from rq import Queue
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None
    Queue = None

# Check for PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    GPU_AVAILABLE = False

# Logger
logger = logging.getLogger(__name__)

# Global state
_config: Optional[APIConfig] = None
_redis_client: Optional["redis.Redis"] = None
_job_queue: Optional["Queue"] = None


def get_config() -> APIConfig:
    """Get the current configuration."""
    global _config
    if _config is None:
        _config = APIConfig.from_env()
    return _config


def get_redis() -> Optional["redis.Redis"]:
    """Get Redis client."""
    global _redis_client
    if _redis_client is None and REDIS_AVAILABLE:
        config = get_config()
        try:
            _redis_client = redis.from_url(
                config.redis_url,
                max_connections=config.redis_max_connections,
                decode_responses=True,
            )
            _redis_client.ping()
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            _redis_client = None
    return _redis_client


def get_queue() -> Optional["Queue"]:
    """Get RQ job queue."""
    global _job_queue
    if _job_queue is None and REDIS_AVAILABLE:
        redis_client = get_redis()
        if redis_client:
            _job_queue = Queue("juris_default", connection=redis_client)
    return _job_queue


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting JURIS-AGI API server...")
    config = get_config()

    # Create trace storage directory
    os.makedirs(config.trace_storage_path, exist_ok=True)

    # Initialize Redis connection
    if REDIS_AVAILABLE:
        redis_client = get_redis()
        if redis_client:
            logger.info("Connected to Redis")
        else:
            logger.warning("Redis not available - running in standalone mode")
    else:
        logger.warning("Redis package not installed - running in standalone mode")

    logger.info(f"PyTorch available: {TORCH_AVAILABLE}, GPU available: {GPU_AVAILABLE}")

    yield

    # Shutdown
    logger.info("Shutting down JURIS-AGI API server...")
    global _redis_client
    if _redis_client:
        _redis_client.close()
        _redis_client = None


def create_app(config: Optional[APIConfig] = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    global _config
    if config:
        _config = config

    app = FastAPI(
        title="JURIS-AGI API",
        description="Neuro-symbolic system for ARC-style abstract reasoning",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app


# Create the default app instance
app = create_app()


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.

    Returns the status of the API server and its dependencies.
    """
    config = get_config()
    redis_client = get_redis()

    # Check Redis connection
    redis_connected = False
    worker_count = 0
    pending_jobs = 0

    if redis_client:
        try:
            redis_client.ping()
            redis_connected = True

            # Count workers and pending jobs
            worker_count = int(redis_client.get("juris:worker_count") or 0)
            pending_jobs = int(redis_client.llen("rq:queue:juris_default") or 0)
        except Exception:
            redis_connected = False

    return HealthResponse(
        status="healthy" if redis_connected or not REDIS_AVAILABLE else "degraded",
        version="0.1.0",
        gpu_available=GPU_AVAILABLE,
        torch_available=TORCH_AVAILABLE,
        redis_connected=redis_connected,
        worker_count=worker_count,
        pending_jobs=pending_jobs,
    )


@app.post("/solve", response_model=SolveResponse, tags=["Solve"])
async def submit_solve_request(
    request: SolveRequest,
    background_tasks: BackgroundTasks,
    priority: str = Query(default="default", regex="^(high|default|low)$"),
):
    """
    Submit an ARC task for solving.

    Returns a job ID that can be used to check the status and retrieve results.
    """
    config = get_config()
    redis_client = get_redis()

    # Generate job ID
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    task_id = request.task_id or f"task_{uuid.uuid4().hex[:8]}"
    created_at = datetime.utcnow()

    # Create job data
    job_data = {
        "job_id": job_id,
        "task_id": task_id,
        "status": JobStatus.PENDING.value,
        "created_at": created_at.isoformat(),
        "task": request.task.model_dump(),
        "budget": request.budget.model_dump(),
        "use_neural": request.use_neural,
        "return_trace": request.return_trace,
    }

    if redis_client:
        # Store job in Redis
        try:
            redis_client.setex(
                f"juris:job:{job_id}",
                config.job_ttl_seconds,
                json.dumps(job_data),
            )

            # Enqueue job
            queue = get_queue()
            if queue:
                queue_name = f"juris_{priority}"
                priority_queue = Queue(queue_name, connection=redis_client)
                priority_queue.enqueue(
                    "juris_agi.api.worker.process_job",
                    job_id,
                    job_timeout=config.job_timeout_seconds,
                )
        except Exception as e:
            logger.error(f"Failed to enqueue job: {e}")
            raise HTTPException(status_code=503, detail="Failed to enqueue job")
    else:
        # Run synchronously in standalone mode
        background_tasks.add_task(run_job_sync, job_id, job_data)
        # Store in memory (simple dict, lost on restart)
        if not hasattr(app.state, "jobs"):
            app.state.jobs = {}
        app.state.jobs[job_id] = job_data

    # Estimate completion time based on budget
    estimated_time = min(
        request.budget.max_time_seconds,
        request.budget.max_iterations / 100.0  # Rough estimate
    )

    return SolveResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        created_at=created_at,
        estimated_time_seconds=estimated_time,
    )


@app.get("/jobs/{job_id}", response_model=JobResult, tags=["Jobs"])
async def get_job_result(
    job_id: str,
    include_trace: bool = Query(default=False, description="Include inline trace data"),
):
    """
    Get the status and result of a solve job.

    Returns the current status, and if completed, the predictions and program.
    """
    config = get_config()
    redis_client = get_redis()

    job_data = None

    if redis_client:
        try:
            raw_data = redis_client.get(f"juris:job:{job_id}")
            if raw_data:
                job_data = json.loads(raw_data)
        except Exception as e:
            logger.error(f"Failed to fetch job: {e}")

    # Fallback to in-memory storage
    if job_data is None and hasattr(app.state, "jobs"):
        job_data = app.state.jobs.get(job_id)

    if job_data is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Build response
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


@app.delete("/jobs/{job_id}", tags=["Jobs"])
async def cancel_job(job_id: str):
    """
    Cancel a pending or running job.

    Returns success if the job was cancelled.
    """
    redis_client = get_redis()

    if redis_client:
        try:
            raw_data = redis_client.get(f"juris:job:{job_id}")
            if not raw_data:
                raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

            job_data = json.loads(raw_data)
            if job_data.get("status") in [JobStatus.COMPLETED.value, JobStatus.FAILED.value]:
                raise HTTPException(status_code=400, detail="Cannot cancel completed job")

            # Mark as failed/cancelled
            job_data["status"] = JobStatus.FAILED.value
            job_data["error_message"] = "Cancelled by user"
            job_data["completed_at"] = datetime.utcnow().isoformat()

            redis_client.setex(
                f"juris:job:{job_id}",
                get_config().job_ttl_seconds,
                json.dumps(job_data),
            )

            return {"message": f"Job {job_id} cancelled"}
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to cancel job: {e}")
            raise HTTPException(status_code=500, detail="Failed to cancel job")

    raise HTTPException(status_code=503, detail="Redis not available")


# =============================================================================
# Standalone Mode (no Redis)
# =============================================================================

async def run_job_sync(job_id: str, job_data: dict):
    """
    Run a job synchronously (standalone mode without Redis).
    """
    from ..core.types import ARCTask, ARCPair, Grid
    from ..controller.router import MetaController, ControllerConfig

    try:
        # Update status
        job_data["status"] = JobStatus.RUNNING.value
        job_data["started_at"] = datetime.utcnow().isoformat()

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

        # Configure controller
        budget = job_data["budget"]
        config = ControllerConfig(
            max_synthesis_depth=budget["max_depth"],
            beam_width=budget["beam_width"],
            max_synthesis_iterations=budget["max_iterations"],
        )
        controller = MetaController(config)

        # Solve
        import time
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
                    "confidence": result.audit_trace.robustness_score,
                })
            job_data["predictions"] = predictions
        else:
            job_data["error_message"] = result.error_message

    except Exception as e:
        job_data["status"] = JobStatus.FAILED.value
        job_data["completed_at"] = datetime.utcnow().isoformat()
        job_data["error_message"] = str(e)
        logger.exception(f"Job {job_id} failed")


# =============================================================================
# Error Handlers
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=f"HTTP_{exc.status_code}",
            message=exc.detail,
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions."""
    logger.exception("Unexpected error")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="INTERNAL_ERROR",
            message="An unexpected error occurred",
            details={"type": type(exc).__name__} if get_config().debug else None,
        ).model_dump(),
    )


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Run the API server."""
    import uvicorn

    config = get_config()
    uvicorn.run(
        "juris_agi.api.server:app",
        host=config.host,
        port=config.port,
        workers=config.workers,
        reload=config.debug,
    )


if __name__ == "__main__":
    main()
