"""
FastAPI server — exposes OpenEnv standard endpoints + required hackathon endpoints.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from models import Action, Observation
from environment import DevEnvEnvironment
from tasks import TASKS, grade_episode

app = FastAPI(
    title="Dev Environment Debugger",
    description="OpenEnv environment where an AI agent debugs a broken multi-service dev stack.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One environment instance per server (single-session for hackathon scope)
env = DevEnvEnvironment()


# ---------------------------------------------------------------------------
# OpenEnv standard endpoints
# ---------------------------------------------------------------------------

@app.post("/reset")
def reset(task_id: str = "task1") -> dict:
    """Start a new episode. Returns initial observation."""
    if task_id not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id. Valid: {list(TASKS.keys())}")
    obs = env.reset(task_id=task_id)
    return obs.dict()


@app.post("/step")
def step(action: Action) -> dict:
    """Submit an action. Returns observation, reward, done, info."""
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.dict(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
def state() -> dict:
    """Returns current environment state."""
    return env.state()


# ---------------------------------------------------------------------------
# Required hackathon endpoints
# ---------------------------------------------------------------------------

@app.get("/tasks")
def list_tasks() -> dict:
    """Returns task list and action schema."""
    action_schema = {
        "type": "string — one of: read_logs | inspect_env | edit_env | restart_service | run_healthcheck | submit",
        "service": "string — target service: api | worker | database | proxy",
        "key": "string — env var key (only for edit_env)",
        "value": "string — env var value (only for edit_env)",
    }
    return {
        "tasks": [
            {
                "id": t["id"],
                "name": t["name"],
                "difficulty": t["difficulty"],
                "description": t["description"],
                "faults_count": len(t["faults"]),
            }
            for t in TASKS.values()
        ],
        "action_schema": action_schema,
    }


@app.get("/grader")
def grader(task_id: str = None) -> dict:
    """Returns grader score for current or specified episode."""
    current_state = env.state()
    tid = task_id or current_state.get("task_id", "task1")
    result = grade_episode(tid, current_state)
    return {
        "task_id": tid,
        "grader_result": result,
        "state_summary": {
            "step": current_state["step"],
            "fixed_faults": current_state["fixed_faults"],
            "remaining_faults": current_state["remaining_faults"],
            "services": {k: v["status"] for k, v in current_state["services"].items()},
        },
    }


@app.post("/baseline")
def baseline() -> dict:
    """
    Runs the baseline inference script against all 3 tasks.
    Returns scores. Requires OPENAI_API_KEY in environment.
    """
    try:
        import subprocess
        import json
        result = subprocess.run(
            ["python", "inference.py"],
            capture_output=True, text=True, timeout=300,
            env={**__import__("os").environ},
        )
        # Parse [END] lines from structured stdout to extract per-task scores
        scores = []
        for line in result.stdout.splitlines():
            if line.startswith("[END] "):
                try:
                    scores.append(json.loads(line[6:]))
                except json.JSONDecodeError:
                    pass
        if result.returncode == 0 or scores:
            return {"status": "success", "scores": scores}
        else:
            return {"status": "error", "message": result.stderr}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ---------------------------------------------------------------------------
# OpenEnv runtime spec endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """OpenEnv spec: must return status=healthy."""
    return {"status": "healthy"}


@app.get("/metadata")
def metadata():
    """OpenEnv spec: environment name and description."""
    return {
        "name": "Dev Environment Debugger",
        "description": (
            "OpenEnv environment where an AI agent debugs a broken multi-service "
            "dev stack by reading logs, inspecting configs, and fixing env vars."
        ),
        "version": "1.0.0",
        "author": "Ajjack4",
    }


@app.get("/schema")
def schema():
    """OpenEnv spec: action, observation, and state schemas."""
    return {
        "action": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["read_logs", "inspect_env", "edit_env",
                             "restart_service", "run_healthcheck", "submit"],
                },
                "service": {"type": "string", "enum": ["api", "worker", "database", "proxy"]},
                "key": {"type": "string"},
                "value": {"type": "string"},
            },
            "required": ["type"],
        },
        "observation": {
            "type": "object",
            "properties": {
                "step": {"type": "integer"},
                "services": {"type": "object"},
                "last_action_result": {"type": "object"},
                "available_actions": {"type": "array", "items": {"type": "string"}},
                "done": {"type": "boolean"},
            },
        },
        "state": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "step": {"type": "integer"},
                "done": {"type": "boolean"},
                "fixed_faults": {"type": "array"},
                "remaining_faults": {"type": "array"},
                "services": {"type": "object"},
            },
        },
    }


@app.post("/mcp")
async def mcp(request: Request):
    """OpenEnv spec: minimal MCP / JSON-RPC 2.0 endpoint."""
    try:
        body = await request.json()
        req_id = body.get("id", 1)
    except Exception:
        req_id = 1
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
            "name": "Dev Environment Debugger",
            "version": "1.0.0",
            "capabilities": {},
        },
    }


@app.get("/")
def root():
    return {
        "name": "Dev Environment Debugger",
        "version": "1.0.0",
        "description": "OpenEnv environment for AI-powered dev environment debugging.",
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/grader",
                      "/baseline", "/health", "/metadata", "/schema", "/mcp"],
        "openenv": True,
    }


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=7860, reload=False)
