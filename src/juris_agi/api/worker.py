"""
Worker for processing JURIS-AGI solve jobs.

The worker:
- Pulls jobs from Redis queues (via RQ)
- Loads model weights once at startup
- Runs the solver with budget limits
- Writes results and trace to storage
- Updates Redis job state
"""

import json
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from .config import WorkerConfig
from .models import JobStatus

# Optional imports
try:
    import redis
    from rq import Worker, Queue, Connection
    from rq.job import Job
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None
    Worker = None
    Queue = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import boto3
    from botocore.client import Config as BotoConfig
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False
    boto3 = None

# Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class JurisWorker:
    """
    Worker class for processing JURIS-AGI jobs.

    Handles model loading, job execution, and result storage.
    """

    def __init__(self, config: Optional[WorkerConfig] = None):
        self.config = config or WorkerConfig.from_env()
        self.redis_client: Optional["redis.Redis"] = None
        self.s3_client = None
        self.device = None
        self.sketcher = None
        self.critic = None
        self._running = False

    def setup(self):
        """Initialize worker resources."""
        logger.info("Setting up JURIS-AGI worker...")

        # Determine device
        if self.config.device == "auto":
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = self.config.device

        logger.info(f"Using device: {self.device}")

        # Connect to Redis
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(
                    self.config.redis_url,
                    decode_responses=True,
                )
                self.redis_client.ping()
                logger.info("Connected to Redis")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise

        # Setup S3 client if configured
        if S3_AVAILABLE and self.config.s3_bucket:
            try:
                self.s3_client = boto3.client(
                    "s3",
                    endpoint_url=self.config.s3_endpoint,
                    aws_access_key_id=self.config.s3_access_key,
                    aws_secret_access_key=self.config.s3_secret_key,
                    config=BotoConfig(signature_version="s3v4"),
                )
                logger.info(f"S3 client configured for bucket: {self.config.s3_bucket}")
            except Exception as e:
                logger.warning(f"Failed to setup S3 client: {e}")

        # Create trace storage directory
        os.makedirs(self.config.trace_storage_path, exist_ok=True)

        # Load models
        self._load_models()

        # Register worker with Redis
        if self.redis_client:
            self.redis_client.incr("juris:worker_count")

        logger.info("Worker setup complete")

    def _load_models(self):
        """Load neural models if available."""
        from ..cre.sketcher_model import get_sketcher, SketcherConfig
        from ..cre.critic_neural import get_critic, CriticConfig

        use_neural = TORCH_AVAILABLE and self.device != "cpu"

        logger.info(f"Loading models (neural={use_neural})...")

        # Load sketcher
        sketcher_config = SketcherConfig()
        self.sketcher = get_sketcher(
            use_neural=use_neural,
            config=sketcher_config,
            model_path=self.config.sketcher_model_path,
        )

        # Load critic
        critic_config = CriticConfig()
        self.critic = get_critic(
            use_neural=use_neural,
            config=critic_config,
            model_path=self.config.critic_model_path,
        )

        logger.info("Models loaded")

    def teardown(self):
        """Cleanup worker resources."""
        logger.info("Tearing down worker...")

        if self.redis_client:
            try:
                self.redis_client.decr("juris:worker_count")
                self.redis_client.close()
            except Exception:
                pass

        logger.info("Worker teardown complete")

    def run(self):
        """Run the worker loop."""
        if not REDIS_AVAILABLE:
            logger.error("Redis/RQ not available. Cannot run worker.")
            return

        self.setup()
        self._running = True

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        try:
            with Connection(self.redis_client):
                queues = [Queue(name) for name in self.config.queue_names]
                worker = Worker(queues, name=f"juris-worker-{os.getpid()}")

                logger.info(f"Starting worker on queues: {self.config.queue_names}")
                worker.work(
                    with_scheduler=False,
                    logging_level="INFO",
                )
        except Exception as e:
            logger.exception(f"Worker error: {e}")
        finally:
            self._running = False
            self.teardown()

    def _handle_signal(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self._running = False

    def process_job(self, job_id: str) -> Dict[str, Any]:
        """
        Process a single solve job.

        This is the function called by RQ.
        """
        from ..core.types import ARCTask, ARCPair, Grid
        from ..controller.router import MetaController, ControllerConfig, determine_regime
        from ..core.trace import TraceWriter

        logger.info(f"Processing job: {job_id}")

        # Fetch job data
        raw_data = self.redis_client.get(f"juris:job:{job_id}")
        if not raw_data:
            raise ValueError(f"Job {job_id} not found")

        job_data = json.loads(raw_data)

        # Update status to running
        job_data["status"] = JobStatus.RUNNING.value
        job_data["started_at"] = datetime.utcnow().isoformat()
        self._update_job(job_id, job_data)

        try:
            # Parse task
            task = self._parse_task(job_data)

            # Configure controller
            budget = job_data["budget"]
            config = ControllerConfig(
                max_synthesis_depth=budget["max_depth"],
                beam_width=budget["beam_width"],
                max_synthesis_iterations=budget["max_iterations"],
            )

            # Determine regime
            regime_decision = determine_regime(task)
            job_data["regime"] = regime_decision.regime.name

            # Create controller and solve
            controller = MetaController(config)

            start_time = time.time()
            result = controller.solve(task)
            runtime = time.time() - start_time

            # Update job with results
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

                # Save trace if requested
                if job_data.get("return_trace"):
                    trace_url = self._save_trace(job_id, result.audit_trace)
                    job_data["trace_url"] = trace_url
            else:
                job_data["error_message"] = result.error_message

            logger.info(f"Job {job_id} completed: success={result.success}")

        except Exception as e:
            logger.exception(f"Job {job_id} failed")
            job_data["status"] = JobStatus.FAILED.value
            job_data["completed_at"] = datetime.utcnow().isoformat()
            job_data["error_message"] = str(e)

        # Save final state
        self._update_job(job_id, job_data)
        return job_data

    def _parse_task(self, job_data: Dict[str, Any]) -> "ARCTask":
        """Parse task from job data."""
        from ..core.types import ARCTask, ARCPair, Grid

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

        return ARCTask(
            task_id=job_data["task_id"],
            train=train_pairs,
            test=test_pairs,
        )

    def _update_job(self, job_id: str, job_data: Dict[str, Any]):
        """Update job data in Redis."""
        if self.redis_client:
            self.redis_client.setex(
                f"juris:job:{job_id}",
                self.config.job_timeout_seconds * 2,  # Keep longer for retrieval
                json.dumps(job_data),
            )

    def _save_trace(self, job_id: str, trace) -> Optional[str]:
        """Save execution trace to storage."""
        try:
            trace_data = trace.to_dict()
            trace_filename = f"{job_id}_trace.json"

            # Save locally
            local_path = Path(self.config.trace_storage_path) / trace_filename
            with open(local_path, "w") as f:
                json.dump(trace_data, f, indent=2, default=str)

            # Upload to S3 if configured
            if self.s3_client and self.config.s3_bucket:
                s3_key = f"traces/{trace_filename}"
                self.s3_client.upload_file(
                    str(local_path),
                    self.config.s3_bucket,
                    s3_key,
                )

                # Generate presigned URL
                url = self.s3_client.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": self.config.s3_bucket, "Key": s3_key},
                    ExpiresIn=3600,
                )
                return url

            # Return local path as file:// URL
            return f"file://{local_path}"

        except Exception as e:
            logger.warning(f"Failed to save trace: {e}")
            return None


# =============================================================================
# RQ Job Function
# =============================================================================

def process_job(job_id: str) -> Dict[str, Any]:
    """
    Process a job - called by RQ.

    This is a module-level function that RQ can import and execute.
    """
    # Create worker instance (will use env config)
    worker = JurisWorker()
    worker.setup()

    try:
        return worker.process_job(job_id)
    finally:
        worker.teardown()


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Run the worker."""
    if not REDIS_AVAILABLE:
        print("ERROR: Redis/RQ packages not installed")
        print("Install with: pip install redis rq")
        sys.exit(1)

    worker = JurisWorker()
    worker.run()


if __name__ == "__main__":
    main()
