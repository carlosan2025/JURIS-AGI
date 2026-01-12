# JURIS-AGI Local Proof-of-Concept Guide

This guide explains how to run JURIS-AGI locally on your laptop without any cloud dependencies.

## Overview

The Local PoC provides two execution modes:

| Mode | Redis | Docker | Use Case |
|------|-------|--------|----------|
| **Pure Python** | No | No | Quick testing, development |
| **Docker Compose** | Yes | Yes | Demo, integration testing |

Both modes are **CPU-only** and store all artifacts locally.

---

## Prerequisites

### For Pure Python Mode
- Python 3.10, 3.11, or 3.12
- ~500MB disk space

### For Docker Compose Mode
- Docker 20.10+ with Docker Compose v2
- ~2GB disk space (for images)

---

## Quick Start: Pure Python Mode

### 1. Install Dependencies

```bash
# Clone and enter directory
cd JURIS-AGI

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install with local dependencies only (no Redis, no boto3)
pip install -e ".[local]"
```

### 2. Run the Server

```bash
# Start the local PoC server
python -m juris_agi.api.local_server

# Or use the CLI entry point (after pip install)
juris-local
```

You should see:
```
============================================================
JURIS-AGI Local Proof-of-Concept Server
============================================================
Mode:        Sync (no Redis)
Host:        127.0.0.1:8000
Runs dir:    ./runs
GPU:         Disabled (CPU-only)
Max grid:    30x30
Max runtime: 60.0s
============================================================

Endpoints:
  POST http://127.0.0.1:8000/solve
  GET  http://127.0.0.1:8000/jobs/{job_id}
  GET  http://127.0.0.1:8000/health

API docs: http://127.0.0.1:8000/docs
============================================================
```

### 3. Test with a Request

```bash
# Check health
curl http://127.0.0.1:8000/health | jq

# Submit a simple rotation task
curl -X POST http://127.0.0.1:8000/solve \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "demo_rotation",
    "task": {
      "train": [
        {
          "input": {"data": [[1, 2], [3, 4]]},
          "output": {"data": [[3, 1], [4, 2]]}
        },
        {
          "input": {"data": [[5, 6], [7, 8]]},
          "output": {"data": [[7, 5], [8, 6]]}
        }
      ],
      "test": [
        {"input": {"data": [[9, 0], [1, 2]]}}
      ]
    },
    "budget": {
      "max_time_seconds": 30,
      "max_iterations": 200
    }
  }' | jq
```

### 4. Check Results

In sync mode, the result is available immediately:

```bash
# Get job details (replace job_id with actual ID from response)
curl http://127.0.0.1:8000/jobs/job_abc123def456 | jq

# Include full trace data
curl "http://127.0.0.1:8000/jobs/job_abc123def456?include_trace=true" | jq
```

Results are saved to:
- `./runs/traces/{date}/{task_id}/{job_id}.json`
- `./runs/results/{date}/{task_id}/{job_id}.json`

---

## Quick Start: Docker Compose Mode

### 1. Build and Start

```bash
# Build and start all containers
docker compose -f docker-compose.local.yml up --build

# Or run in background
docker compose -f docker-compose.local.yml up -d --build
```

This starts:
- **redis**: Job queue on port 6379
- **api**: FastAPI server on port 8000
- **worker**: Background job processor

### 2. Test the API

```bash
# Check health
curl http://localhost:8000/health | jq

# Submit a task (returns immediately, job runs async)
curl -X POST http://localhost:8000/solve \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "docker_test",
    "task": {
      "train": [
        {
          "input": {"data": [[1, 0], [0, 1]]},
          "output": {"data": [[0, 1], [1, 0]]}
        }
      ],
      "test": [
        {"input": {"data": [[2, 0], [0, 2]]}}
      ]
    }
  }' | jq

# Poll for result
curl http://localhost:8000/jobs/{job_id} | jq
```

### 3. View Logs

```bash
# All logs
docker compose -f docker-compose.local.yml logs -f

# Just the worker
docker compose -f docker-compose.local.yml logs -f worker
```

### 4. Stop

```bash
docker compose -f docker-compose.local.yml down

# Remove volumes too
docker compose -f docker-compose.local.yml down -v
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LOCAL_POC` | `true` | Enable local PoC mode |
| `SYNC_MODE` | `true` | Sync execution (no Redis) |
| `REDIS_ENABLED` | `false` | Enable Redis queue |
| `GPU_ENABLED` | `false` | Enable GPU (always false in PoC) |
| `RUNS_DIR` | `./runs` | Local storage directory |
| `MAX_GRID_SIZE` | `30` | Maximum grid dimension |
| `MAX_SEARCH_EXPANSIONS` | `500` | Max beam search expansions |
| `MAX_RUNTIME_SECONDS` | `60` | Max time per job |
| `MAX_PROGRAM_DEPTH` | `4` | Max synthesized program depth |
| `JURIS_HOST` | `127.0.0.1` | Server bind address |
| `JURIS_PORT` | `8000` | Server port |

### Example: Custom Configuration

```bash
# Run with custom limits
MAX_RUNTIME_SECONDS=120 MAX_SEARCH_EXPANSIONS=1000 \
  python -m juris_agi.api.local_server
```

---

## API Reference

### POST /solve

Submit an ARC task for solving.

**Request:**
```json
{
  "task_id": "optional_task_name",
  "task": {
    "train": [
      {
        "input": {"data": [[0, 1], [2, 3]]},
        "output": {"data": [[2, 0], [3, 1]]}
      }
    ],
    "test": [
      {"input": {"data": [[4, 5], [6, 7]]}}
    ]
  },
  "budget": {
    "max_time_seconds": 30,
    "max_iterations": 200,
    "beam_width": 20,
    "max_depth": 3
  },
  "return_trace": true
}
```

**Response:**
```json
{
  "job_id": "job_abc123def456",
  "status": "completed",
  "created_at": "2025-01-12T10:30:00Z",
  "estimated_time_seconds": 5.2
}
```

### GET /jobs/{job_id}

Get job status and results.

**Query Parameters:**
- `include_trace`: Include full trace data in response

**Response:**
```json
{
  "job_id": "job_abc123def456",
  "status": "completed",
  "task_id": "demo_rotation",
  "success": true,
  "predictions": [
    {
      "test_index": 0,
      "prediction": {"data": [[1, 9], [2, 0]]},
      "confidence": 0.92
    }
  ],
  "program": "rotate90(1)",
  "robustness_score": 0.92,
  "runtime_seconds": 3.45,
  "trace_url": "file:///path/to/trace.json",
  "result_url": "file:///path/to/result.json"
}
```

### GET /health

Health check with configuration info.

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0-poc",
  "mode": "local_poc",
  "execution": "sync",
  "gpu_available": false,
  "redis_connected": false,
  "config": {
    "mode": "sync",
    "limits": {
      "max_grid_size": 30,
      "max_runtime_seconds": 60
    }
  }
}
```

---

## Trace Output Format

Traces include a human-readable summary:

```json
{
  "summary": {
    "job_id": "job_abc123",
    "task_id": "demo_rotation",
    "success": true,
    "final_program": "rotate90(1)",
    "robustness_score": 0.92,
    "synthesis_iterations": 45,
    "inferred_invariants": [
      "grid_size_preserved",
      "color_set_preserved"
    ],
    "candidate_programs_tried": [
      "identity()",
      "flip_horizontal()",
      "rotate90(1)"
    ],
    "refinement_steps": [
      "rejected: identity() - mismatch on train[0]",
      "rejected: flip_horizontal() - mismatch on train[1]",
      "accepted: rotate90(1) - all train pairs match"
    ]
  },
  "full_trace": { ... }
}
```

---

## Limitations

### CPU-Only
- No GPU acceleration
- Slower synthesis search
- Suitable for small to medium grids

### Search Constraints
- Max 500 search expansions (vs 10,000+ in production)
- Max 60s runtime per job
- Max 30x30 grid size

### Demo Scope
- Best for simple transformations (rotations, flips, color swaps)
- May timeout on complex multi-step patterns
- Not optimized for ARC-AGI competition evaluation

### Not Included
- Neural sketcher hints (requires GPU)
- S3/cloud storage
- Production-grade scaling

---

## Troubleshooting

### "Connection refused" on port 8000
```bash
# Check if server is running
curl http://127.0.0.1:8000/health

# Check for port conflicts
lsof -i :8000
```

### Docker build fails
```bash
# Clean rebuild
docker compose -f docker-compose.local.yml down -v
docker compose -f docker-compose.local.yml build --no-cache
```

### Import errors
```bash
# Ensure you're in the right environment
which python
pip list | grep juris

# Reinstall
pip install -e ".[local]"
```

### Job stuck in "pending"
In Docker mode, check if worker is running:
```bash
docker compose -f docker-compose.local.yml logs worker
```

---

## Example Tasks

### Simple Rotation
```json
{
  "task": {
    "train": [
      {"input": {"data": [[1,2],[3,4]]}, "output": {"data": [[3,1],[4,2]]}}
    ],
    "test": [{"input": {"data": [[5,6],[7,8]]}}]
  }
}
```

### Color Swap
```json
{
  "task": {
    "train": [
      {"input": {"data": [[1,1],[2,2]]}, "output": {"data": [[2,2],[1,1]]}}
    ],
    "test": [{"input": {"data": [[1,2],[1,2]]}}]
  }
}
```

### Fill Pattern
```json
{
  "task": {
    "train": [
      {"input": {"data": [[0,1,0],[0,0,0]]}, "output": {"data": [[1,1,1],[1,1,1]]}}
    ],
    "test": [{"input": {"data": [[0,0,2],[0,0,0]]}}]
  }
}
```

---

## Next Steps

- Explore the API docs at http://127.0.0.1:8000/docs
- Check `./runs/traces/` for detailed execution logs
- Try more complex ARC tasks from the evaluation set
- See `docs/` for architecture documentation
