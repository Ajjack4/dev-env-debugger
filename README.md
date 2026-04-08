---
title: Dev Environment Debugger
emoji: 🐛
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Dev Environment Debugger

An [OpenEnv](https://huggingface.co/openenv) environment where an AI agent debugs a broken multi-service development stack. The agent must read logs, inspect configurations, fix environment variables, and restart services to restore a healthy system — exactly as a senior engineer would during onboarding or an incident.

---

## Motivation

Setting up and debugging a local development environment is one of the most time-consuming tasks in software engineering. Developers routinely lose days diagnosing misconfigured ports, missing env vars, and cascading service failures. No existing agent benchmark targets this problem.

This environment fills that gap: it tests whether an agent can **reason across multiple failure signals, take stateful corrective actions, and know when the system is healthy**.

---

## Environment Description

Simulates a realistic 4-service application stack:

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Reverse    │────▶│   Backend    │────▶│   Database   │
│    Proxy     │     │     API      │     │  (Postgres)  │
└──────────────┘     └──────┬───────┘     └──────────────┘
                            │
                     ┌──────▼───────┐
                     │    Worker    │
                     │   Process    │
                     └──────────────┘
```

All services run **in memory** — no real Docker, no real ports. The environment is fully deterministic and self-contained.

At the start of each episode, one or more faults are injected (wrong env vars, missing config). The agent must fix them all.

---

## Observation Space

```json
{
  "step": 3,
  "services": {
    "proxy":    { "status": "healthy", "port": 80   },
    "api":      { "status": "error",   "port": 8000 },
    "database": { "status": "healthy", "port": 5432 },
    "worker":   { "status": "error",   "port": null }
  },
  "last_action_result": {
    "success": false,
    "message": "API failed to start: cannot connect to database on port 5433"
  },
  "available_actions": ["read_logs", "inspect_env", "edit_env", "restart_service", "run_healthcheck", "submit"],
  "done": false
}
```

---

## Action Space

| Action | Required Params | Description |
|---|---|---|
| `read_logs` | `service` | Read error logs for a service |
| `inspect_env` | `service` | View environment variables for a service |
| `edit_env` | `service`, `key`, `value` | Change an environment variable |
| `restart_service` | `service` | Restart a service after config fix |
| `run_healthcheck` | `service` | Get current health status |
| `submit` | — | Declare environment fixed, end episode |

Valid services: `api`, `worker`, `database`, `proxy`

---

## Reward Function

| Event | Reward |
|---|---|
| Read logs before fixing (correct diagnosis) | +0.05 |
| Inspect env vars | +0.02 |
| Correct env var edit (fixes a known fault) | +0.10 |
| Service transitions unhealthy → healthy | +0.30 |
| Submit with all services healthy | +0.50 |
| Restart service without fixing root cause | -0.10 |
| Redundant/invalid action | -0.05 |
| Submit while services still unhealthy | -0.30 |
| Exceed max steps (25) | -0.20 |

---

## Tasks

### Task 1 — Single Service Fault (Easy)
One service is down because `DB_PORT` is set to `5433` instead of `5432`.

**Agent must:** read API logs → inspect API env → fix `DB_PORT` → restart API → submit.

**Expected score:** ~0.85 (GPT-4o-mini)

---

### Task 2 — Multi-Fault Diagnosis (Medium)
Two independent faults:
- `DB_PORT` wrong on API service
- `WORKER_QUEUE_URL` missing on worker service

**Agent must:** identify and fix both faults independently. Partial score for fixing one.

**Expected score:** ~0.60 (GPT-4o-mini)

---

### Task 3 — Cascading Failure Debug (Hard)
`DB_PASSWORD` is wrong on the database. This causes:
- Database → auth failure (`FATAL: password authentication failed`)
- API → misleading `connection timeout` error
- Worker and Proxy → downstream failures

**Agent must:** trace root cause (database credentials) rather than patching symptoms. Misleading logs make this hard.

**Expected score:** ~0.35 (GPT-4o-mini)

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/reset?task_id=task1` | Start new episode |
| POST | `/step` | Submit action, get observation + reward |
| GET | `/state` | Current environment state |
| GET | `/tasks` | Task list and action schema |
| GET | `/grader?task_id=task1` | Score current episode |
| POST | `/baseline` | Run baseline agent on all tasks |
| GET | `/health` | Health check |

---

## Setup & Running

### Local (Python)

```bash
git clone https://github.com/your-username/dev-env-debugger
cd dev-env-debugger
pip install -r requirements.txt
python api.py
# Server starts at http://localhost:7860
```

### Docker

```bash
docker build -t dev-env-debugger .
docker run -p 7860:7860 dev-env-debugger
```

### Validate OpenEnv spec

```bash
openenv validate
```

---

## Running the Baseline

```bash
export OPENAI_API_KEY=your_key_here

# Run all tasks
python baseline.py

# Run single task
python baseline.py --task task1

# JSON output (used by /baseline endpoint)
python baseline.py --output-json
```

### Baseline Scores (GPT-4o-mini)

| Task | Difficulty | Expected Score |
|---|---|---|
| task1 | Easy | ~0.85 |
| task2 | Medium | ~0.60 |
| task3 | Hard | ~0.35 |
| **Average** | | **~0.60** |

---

## Quick API Example

```bash
# Start episode
curl -X POST "http://localhost:7860/reset?task_id=task1"

# Read logs
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"type": "read_logs", "service": "api"}'

# Fix env var
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"type": "edit_env", "service": "api", "key": "DB_PORT", "value": "5432"}'

# Restart service
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"type": "restart_service", "service": "api"}'

# Submit
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"type": "submit"}'

# Get score
curl "http://localhost:7860/grader?task_id=task1"
```

---

## Project Structure

```
dev-env-debugger/
├── environment.py   # Core simulation — state machine, fault injection, action handlers
├── tasks.py         # Task definitions and programmatic graders
├── models.py        # Pydantic models (Observation, Action, Reward, ServiceStatus)
├── api.py           # FastAPI server exposing all endpoints
├── baseline.py      # LLM baseline inference script
├── openenv.yaml     # OpenEnv metadata
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## License

MIT
