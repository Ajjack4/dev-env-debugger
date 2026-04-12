"""
FastAPI server — exposes OpenEnv standard endpoints + required hackathon endpoints.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
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


@app.get("/", response_class=HTMLResponse)
def root():
    return HTMLResponse(content=DASHBOARD_HTML)


@app.get("/api")
def api_info():
    return {
        "name": "Dev Environment Debugger",
        "version": "1.0.0",
        "description": "OpenEnv environment for AI-powered dev environment debugging.",
        "tasks": list(TASKS.keys()),
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/grader",
                      "/baseline", "/health", "/metadata", "/schema", "/mcp"],
        "openenv": True,
    }


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dev Environment Debugger — OpenEnv</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0d1117; color: #e6edf3; min-height: 100vh; }
  header { background: linear-gradient(135deg, #161b22 0%, #1f2937 100%); border-bottom: 1px solid #30363d; padding: 24px 32px; display: flex; align-items: center; gap: 16px; }
  header h1 { font-size: 1.5rem; font-weight: 700; }
  header .badge { background: #238636; color: #fff; padding: 4px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }
  header .subtitle { color: #8b949e; font-size: 0.9rem; margin-top: 4px; }
  main { max-width: 1100px; margin: 0 auto; padding: 32px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 32px; }
  @media(max-width: 700px) { .grid { grid-template-columns: 1fr; } }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 24px; }
  .card h2 { font-size: 1rem; font-weight: 600; color: #58a6ff; margin-bottom: 16px; display: flex; align-items: center; gap: 8px; }
  .service { display: flex; justify-content: space-between; align-items: center; padding: 10px 14px; border-radius: 8px; margin-bottom: 8px; background: #0d1117; border: 1px solid #21262d; }
  .service-name { font-weight: 600; font-size: 0.9rem; }
  .status { padding: 3px 10px; border-radius: 10px; font-size: 0.78rem; font-weight: 700; }
  .healthy { background: #1a4a2a; color: #3fb950; border: 1px solid #238636; }
  .error { background: #4a1a1a; color: #f85149; border: 1px solid #da3633; }
  .task-card { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 20px; margin-bottom: 16px; cursor: pointer; transition: border-color 0.2s; }
  .task-card:hover { border-color: #58a6ff; }
  .task-card.active { border-color: #58a6ff; background: #1c2433; }
  .task-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
  .task-name { font-weight: 700; }
  .diff { padding: 3px 10px; border-radius: 10px; font-size: 0.75rem; font-weight: 700; }
  .easy { background: #1a4a2a; color: #3fb950; }
  .medium { background: #4a3a1a; color: #d29922; }
  .hard { background: #4a1a1a; color: #f85149; }
  .expert { background: #2d1a4a; color: #bc8cff; }
  .task-desc { color: #8b949e; font-size: 0.85rem; line-height: 1.5; }
  .actions { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 16px; }
  button { padding: 9px 18px; border-radius: 8px; border: none; cursor: pointer; font-size: 0.85rem; font-weight: 600; transition: opacity 0.2s; }
  button:hover { opacity: 0.85; }
  .btn-primary { background: #238636; color: #fff; }
  .btn-secondary { background: #21262d; color: #e6edf3; border: 1px solid #30363d; }
  .btn-danger { background: #da3633; color: #fff; }
  .log { background: #0d1117; border: 1px solid #21262d; border-radius: 8px; padding: 14px; font-family: 'Cascadia Code', 'Fira Code', monospace; font-size: 0.8rem; line-height: 1.6; max-height: 260px; overflow-y: auto; color: #8b949e; white-space: pre-wrap; word-break: break-all; }
  .log .ok { color: #3fb950; }
  .log .err { color: #f85149; }
  .log .info { color: #58a6ff; }
  .score-bar { height: 8px; border-radius: 4px; background: #21262d; margin-top: 8px; overflow: hidden; }
  .score-fill { height: 100%; border-radius: 4px; background: linear-gradient(90deg, #238636, #3fb950); transition: width 0.5s; }
  .step-counter { font-size: 2rem; font-weight: 800; color: #58a6ff; }
  .stat-label { color: #8b949e; font-size: 0.8rem; margin-top: 4px; }
  .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 16px; }
  .stat { text-align: center; }
  #action-panel { margin-top: 24px; }
  select, input[type=text] { background: #0d1117; border: 1px solid #30363d; color: #e6edf3; border-radius: 6px; padding: 8px 12px; font-size: 0.85rem; width: 100%; margin-bottom: 8px; }
  .full-width { grid-column: 1 / -1; }
  .arch { display: flex; align-items: center; justify-content: center; gap: 8px; flex-wrap: wrap; padding: 16px 0; }
  .svc-box { padding: 8px 16px; border-radius: 8px; font-size: 0.85rem; font-weight: 600; border: 2px solid; text-align: center; min-width: 80px; }
  .arrow { color: #58a6ff; font-size: 1.2rem; }
</style>
</head>
<body>
<header>
  <div>
    <div style="display:flex;align-items:center;gap:12px">
      <h1>🔧 Dev Environment Debugger</h1>
      <span class="badge">OpenEnv</span>
    </div>
    <div class="subtitle">AI agent benchmark · 5 tasks · realistic multi-service debugging</div>
  </div>
</header>
<main>
  <div class="grid">
    <!-- Services Panel -->
    <div class="card">
      <h2>⚙️ Service Status</h2>
      <div id="arch" class="arch">
        <div class="svc-box" id="box-proxy">proxy</div>
        <div class="arrow">→</div>
        <div class="svc-box" id="box-api">api</div>
        <div class="arrow">→</div>
        <div class="svc-box" id="box-database">database</div>
      </div>
      <div style="text-align:center;margin-bottom:16px">
        <div class="arrow" style="font-size:1rem">↓</div>
        <div class="svc-box" id="box-worker" style="display:inline-block;margin-top:4px">worker</div>
      </div>
      <div id="services-list"></div>
      <div class="stats" style="margin-top:16px">
        <div class="stat"><div class="step-counter" id="step-count">0</div><div class="stat-label">Steps</div></div>
        <div class="stat"><div class="step-counter" id="fixed-count" style="color:#3fb950">0</div><div class="stat-label">Fixed</div></div>
        <div class="stat"><div class="step-counter" id="score-val" style="color:#d29922">—</div><div class="stat-label">Score</div></div>
      </div>
    </div>

    <!-- Action Panel -->
    <div class="card">
      <h2>🎮 Actions</h2>
      <select id="action-type" onchange="updateActionFields()">
        <option value="read_logs">read_logs</option>
        <option value="inspect_env">inspect_env</option>
        <option value="edit_env">edit_env</option>
        <option value="restart_service">restart_service</option>
        <option value="run_healthcheck">run_healthcheck</option>
        <option value="submit">submit</option>
      </select>
      <div id="service-row">
        <select id="action-service">
          <option value="api">api</option>
          <option value="worker">worker</option>
          <option value="database">database</option>
          <option value="proxy">proxy</option>
        </select>
      </div>
      <div id="kv-row" style="display:none">
        <input type="text" id="action-key" placeholder="env key (e.g. DB_PORT)" />
        <input type="text" id="action-value" placeholder="new value (e.g. 5432)" />
      </div>
      <div class="actions">
        <button class="btn-primary" onclick="doStep()">▶ Execute Action</button>
        <button class="btn-secondary" onclick="doGrade()">📊 Grade</button>
      </div>
      <div class="log" id="action-log" style="margin-top:16px">No actions yet. Start an episode below.</div>
    </div>

    <!-- Tasks Panel -->
    <div class="card full-width">
      <h2>📋 Tasks</h2>
      <div id="tasks-list"></div>
    </div>
  </div>
</main>

<script>
let currentTask = 'task1';

async function fetchTasks() {
  const r = await fetch('/tasks');
  const d = await r.json();
  const el = document.getElementById('tasks-list');
  el.innerHTML = d.tasks.map(t => `
    <div class="task-card ${t.id === currentTask ? 'active' : ''}" id="tc-${t.id}" onclick="selectTask('${t.id}')">
      <div class="task-header">
        <span class="task-name">${t.id} — ${t.name}</span>
        <span class="diff ${t.difficulty}">${t.difficulty}</span>
      </div>
      <div class="task-desc">${t.description}</div>
      <div class="actions" style="margin-top:12px">
        <button class="btn-primary" onclick="event.stopPropagation(); resetTask('${t.id}')">▶ Start Episode</button>
        <span style="color:#8b949e;font-size:0.8rem;align-self:center">${t.faults_count} fault${t.faults_count>1?'s':''}</span>
      </div>
    </div>
  `).join('');
}

function selectTask(id) {
  currentTask = id;
  document.querySelectorAll('.task-card').forEach(c => c.classList.remove('active'));
  const el = document.getElementById('tc-' + id);
  if (el) el.classList.add('active');
}

async function resetTask(taskId) {
  selectTask(taskId);
  const r = await fetch('/reset?task_id=' + taskId, {method:'POST'});
  const obs = await r.json();
  updateServices(obs.services);
  log('info', 'Episode started: ' + taskId);
  log('ok', obs.last_action_result.message);
  document.getElementById('step-count').textContent = obs.step;
  document.getElementById('fixed-count').textContent = '0';
  document.getElementById('score-val').textContent = '—';
}

async function doStep() {
  const type = document.getElementById('action-type').value;
  const service = document.getElementById('action-service').value;
  const key = document.getElementById('action-key').value;
  const value = document.getElementById('action-value').value;
  const action = {type};
  if (type !== 'submit') action.service = service;
  if (type === 'edit_env') { action.key = key; action.value = value; }

  const r = await fetch('/step', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(action)});
  const d = await r.json();
  const obs = d.observation;
  updateServices(obs.services);
  document.getElementById('step-count').textContent = obs.step;
  const cls = d.reward > 0 ? 'ok' : d.reward < 0 ? 'err' : 'info';
  log(cls, `[${type}${service && type!=='submit'?' → '+service:''}] reward=${d.reward > 0 ? '+' : ''}${d.reward.toFixed(2)}`);
  log('info', obs.last_action_result.message.replace(/\\n/g, ' '));
  if (d.done) { log('ok', '✓ Episode done!'); doGrade(); }
}

async function doGrade() {
  const r = await fetch('/grader?task_id=' + currentTask);
  const d = await r.json();
  const gr = d.grader_result;
  document.getElementById('score-val').textContent = (gr.score * 100).toFixed(1) + '%';
  document.getElementById('fixed-count').textContent = gr.faults_fixed || 0;
  log('ok', `Score: ${(gr.score*100).toFixed(1)}% — ${gr.reason}`);
}

function updateServices(services) {
  const colors = {healthy: '#3fb950', error: '#f85149'};
  const bg = {healthy: '#1a4a2a', error: '#4a1a1a'};
  const border = {healthy: '#238636', error: '#da3633'};
  const list = document.getElementById('services-list');
  list.innerHTML = Object.entries(services).map(([name, s]) => `
    <div class="service">
      <span class="service-name">${name}</span>
      <div style="display:flex;align-items:center;gap:8px">
        ${s.port ? `<span style="color:#8b949e;font-size:0.8rem">:${s.port}</span>` : ''}
        <span class="status ${s.status}">${s.status}</span>
      </div>
    </div>
  `).join('');
  ['proxy','api','database','worker'].forEach(name => {
    const box = document.getElementById('box-' + name);
    if (box && services[name]) {
      const st = services[name].status;
      box.style.borderColor = border[st] || '#30363d';
      box.style.color = colors[st] || '#e6edf3';
      box.style.background = bg[st] || '#21262d';
    }
  });
}

function log(cls, msg) {
  const el = document.getElementById('action-log');
  const line = document.createElement('div');
  line.className = cls;
  line.textContent = '› ' + msg;
  el.appendChild(line);
  el.scrollTop = el.scrollHeight;
  if (el.children.length > 100) el.removeChild(el.firstChild);
}

function updateActionFields() {
  const type = document.getElementById('action-type').value;
  document.getElementById('service-row').style.display = type === 'submit' ? 'none' : 'block';
  document.getElementById('kv-row').style.display = type === 'edit_env' ? 'block' : 'none';
}

fetchTasks();
resetTask('task1');
</script>
</body>
</html>"""


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=7860, reload=False)
