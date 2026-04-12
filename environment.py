"""
Dev Environment Debugger - Core Environment
Simulates a broken multi-service dev stack in memory.
Agent must diagnose and fix faults to bring all services healthy.
"""

import copy
from typing import Any
from models import Observation, Action, Reward, ServiceStatus, ActionResult

# ---------------------------------------------------------------------------
# Fault catalog — injectable faults with realistic log messages
# ---------------------------------------------------------------------------

FAULT_CATALOG = {
    "wrong_db_port": {
        "service": "api",
        "description": "DB_PORT set to wrong value",
        "env_key": "DB_PORT",
        "bad_value": "5433",
        "fix_key": "DB_PORT",
        "fix_value": "5432",
        "log": "Error: cannot connect to database on port 5433. Connection refused. Is the database running?",
        "requires_restart": ["api"],
    },
    "missing_queue_url": {
        "service": "worker",
        "description": "WORKER_QUEUE_URL not set",
        "env_key": "WORKER_QUEUE_URL",
        "bad_value": None,  # missing
        "fix_key": "WORKER_QUEUE_URL",
        "fix_value": "redis://localhost:6379",
        "log": "Error: WORKER_QUEUE_URL is not set. Worker cannot connect to queue. Set WORKER_QUEUE_URL.",
        "requires_restart": ["worker"],
    },
    "wrong_api_port": {
        "service": "proxy",
        "description": "PROXY_UPSTREAM points to wrong API port",
        "env_key": "PROXY_UPSTREAM",
        "bad_value": "http://api:9999",
        "fix_key": "PROXY_UPSTREAM",
        "fix_value": "http://api:8000",
        "log": "Error: upstream http://api:9999 unreachable. 502 Bad Gateway. Check PROXY_UPSTREAM.",
        "requires_restart": ["proxy"],
    },
    "wrong_db_password": {
        "service": "database",
        "description": "DB_PASSWORD is incorrect causing auth failure",
        "env_key": "DB_PASSWORD",
        "bad_value": "wrongpassword",
        "fix_key": "DB_PASSWORD",
        "fix_value": "secret",
        "log": "FATAL: password authentication failed for user 'app'. Check DB_PASSWORD env var.",
        "requires_restart": ["database", "api", "worker"],
    },
    "wrong_db_host": {
        "service": "api",
        "description": "DB_HOST points to wrong hostname",
        "env_key": "DB_HOST",
        "bad_value": "db-prod-01.internal",
        "fix_key": "DB_HOST",
        "fix_value": "localhost",
        "log": "Error: could not resolve host 'db-prod-01.internal'. Check DB_HOST. Is this a prod config?",
        "requires_restart": ["api"],
    },
    "wrong_api_host": {
        "service": "worker",
        "description": "Worker API_HOST points to wrong address",
        "env_key": "API_HOST",
        "bad_value": "http://api:9000",
        "fix_key": "API_HOST",
        "fix_value": "http://api:8000",
        "log": "Error: worker cannot reach API at http://api:9000. Connection refused. Check API_HOST.",
        "requires_restart": ["worker"],
    },
    "missing_db_password": {
        "service": "api",
        "description": "DB_PASSWORD not set on API service",
        "env_key": "DB_PASSWORD",
        "bad_value": None,
        "fix_key": "DB_PASSWORD",
        "fix_value": "secret",
        "log": "Error: DB_PASSWORD is not set. Database connection rejected (auth requires password).",
        "requires_restart": ["api"],
    },
}

# ---------------------------------------------------------------------------
# Service dependency graph
# ---------------------------------------------------------------------------

DEPENDENCIES = {
    "proxy":    ["api"],
    "api":      ["database"],
    "worker":   ["database"],
    "database": [],
}

# Default healthy env vars per service
DEFAULT_ENV = {
    "api": {
        "DB_HOST": "localhost",
        "DB_PORT": "5432",
        "DB_PASSWORD": "secret",
        "API_PORT": "8000",
    },
    "worker": {
        "WORKER_QUEUE_URL": "redis://localhost:6379",
        "API_HOST": "http://api:8000",
    },
    "database": {
        "DB_PASSWORD": "secret",
        "DB_PORT": "5432",
    },
    "proxy": {
        "PROXY_UPSTREAM": "http://api:8000",
        "PROXY_PORT": "80",
    },
}

SERVICE_PORTS = {
    "api": 8000,
    "worker": None,
    "database": 5432,
    "proxy": 80,
}

MAX_STEPS = 30


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------

class DevEnvEnvironment:
    def __init__(self):
        self._state: dict[str, Any] = {}
        self._active_faults: list[str] = []
        self._fixed_faults: set[str] = set()
        self._step_count: int = 0
        self._done: bool = False
        self._task_id: str = "task1"
        self._log_history: list[str] = []

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self, task_id: str = "task1") -> Observation:
        self._task_id = task_id
        self._step_count = 0
        self._done = False
        self._fixed_faults = set()
        self._log_history = []

        from tasks import TASKS
        task = TASKS[task_id]
        self._active_faults = list(task["faults"])

        self._state = {
            "envs": copy.deepcopy(DEFAULT_ENV),
            "services": {},
        }

        # Inject faults
        for fault_id in self._active_faults:
            fault = FAULT_CATALOG[fault_id]
            svc = fault["service"]
            key = fault["env_key"]
            if fault["bad_value"] is None:
                self._state["envs"][svc].pop(key, None)
            else:
                self._state["envs"][svc][key] = fault["bad_value"]

        self._recompute_statuses()

        return self._build_observation(
            last_action_result=ActionResult(
                success=True,
                message="Environment initialized. Services started. Check service statuses and read logs."
            )
        )

    def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        if self._done:
            obs = self._build_observation(
                ActionResult(success=False, message="Episode already done. Call reset().")
            )
            return obs, 0.0, True, {}

        self._step_count += 1
        reward = 0.0

        if action.type == "read_logs":
            result, r = self._handle_read_logs(action)
            reward += r
        elif action.type == "inspect_env":
            result, r = self._handle_inspect_env(action)
            reward += r
        elif action.type == "edit_env":
            result, r = self._handle_edit_env(action)
            reward += r
        elif action.type == "restart_service":
            result, r = self._handle_restart_service(action)
            reward += r
        elif action.type == "run_healthcheck":
            result, r = self._handle_healthcheck(action)
            reward += r
        elif action.type == "submit":
            result, r, self._done = self._handle_submit()
            reward += r
        else:
            result = ActionResult(success=False, message=f"Unknown action type: '{action.type}'. Valid: read_logs, inspect_env, edit_env, restart_service, run_healthcheck, submit")
            reward -= 0.05

        if self._step_count >= MAX_STEPS and not self._done:
            self._done = True
            reward -= 0.2
            result.message += " | Max steps exceeded."

        self._log_history.append(f"Step {self._step_count}: {action.type} → {result.message}")

        obs = self._build_observation(result)
        return obs, round(reward, 3), self._done, {
            "step": self._step_count,
            "fixed_faults": list(self._fixed_faults),
            "remaining_faults": [f for f in self._active_faults if f not in self._fixed_faults],
        }

    def state(self) -> dict:
        return {
            "task_id": self._task_id,
            "step": self._step_count,
            "done": self._done,
            "active_faults": self._active_faults,
            "fixed_faults": list(self._fixed_faults),
            "remaining_faults": [f for f in self._active_faults if f not in self._fixed_faults],
            "services": {k: v.dict() for k, v in self._state["services"].items()},
            "envs": self._state["envs"],
        }

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _handle_read_logs(self, action: Action) -> tuple[ActionResult, float]:
        svc = action.service
        if svc not in self._state["services"]:
            return ActionResult(success=False, message=f"Unknown service '{svc}'. Valid: api, worker, database, proxy"), -0.05

        status = self._state["services"][svc]
        reward = 0.0

        if status.status == "error":
            logs = []
            for fault_id in self._active_faults:
                if fault_id not in self._fixed_faults:
                    fault = FAULT_CATALOG[fault_id]
                    if fault["service"] == svc:
                        logs.append(f"[ERROR] {fault['log']}")
            # Add downstream symptom logs for dependent failures
            for fault_id in self._active_faults:
                if fault_id not in self._fixed_faults:
                    fault = FAULT_CATALOG[fault_id]
                    if svc in fault.get("requires_restart", []) and fault["service"] != svc:
                        logs.append(f"[ERROR] Service unavailable — upstream dependency '{fault['service']}' is failing.")

            log_text = "\n".join(logs) if logs else "No errors in log."
            reward += 0.05
            return ActionResult(success=True, message=f"[{svc} logs]\n{log_text}"), reward
        else:
            return ActionResult(success=True, message=f"[{svc} logs]\nNo errors. Service running normally."), 0.0

    def _handle_inspect_env(self, action: Action) -> tuple[ActionResult, float]:
        svc = action.service
        if svc not in self._state["envs"]:
            return ActionResult(success=False, message=f"Unknown service '{svc}'. Valid: api, worker, database, proxy"), -0.05

        env = self._state["envs"][svc]
        env_text = "\n".join(f"  {k}={v}" for k, v in sorted(env.items()))
        return ActionResult(success=True, message=f"[{svc} env]\n{env_text}"), 0.02

    def _handle_edit_env(self, action: Action) -> tuple[ActionResult, float]:
        svc = action.service
        if not action.key or action.value is None:
            return ActionResult(success=False, message="edit_env requires 'key' and 'value'."), -0.05
        if svc not in self._state["envs"]:
            return ActionResult(success=False, message=f"Unknown service '{svc}'."), -0.05

        self._state["envs"][svc][action.key] = action.value

        reward = 0.0
        for fault_id in self._active_faults:
            if fault_id in self._fixed_faults:
                continue
            fault = FAULT_CATALOG[fault_id]
            if (fault["service"] == svc and
                    fault["fix_key"] == action.key and
                    fault["fix_value"] == action.value):
                reward += 0.1
                return ActionResult(
                    success=True,
                    message=f"Updated {svc}.{action.key}={action.value}. Now restart {svc} to apply."
                ), reward

        return ActionResult(
            success=True,
            message=f"Updated {svc}.{action.key}={action.value}."
        ), reward

    def _handle_restart_service(self, action: Action) -> tuple[ActionResult, float]:
        svc = action.service
        if svc not in self._state["services"]:
            return ActionResult(success=False, message=f"Unknown service '{svc}'."), -0.05

        reward = 0.0
        prev_statuses = {s: self._state["services"][s].status for s in self._state["services"]}

        for fault_id in self._active_faults:
            if fault_id in self._fixed_faults:
                continue
            fault = FAULT_CATALOG[fault_id]
            if svc in fault["requires_restart"]:
                current_val = self._state["envs"].get(fault["service"], {}).get(fault["fix_key"])
                if current_val == fault["fix_value"]:
                    self._fixed_faults.add(fault_id)

        self._recompute_statuses()

        for s, status_obj in self._state["services"].items():
            if prev_statuses[s] != "healthy" and status_obj.status == "healthy":
                reward += 0.3

        if self._state["services"][svc].status == "error":
            reward -= 0.1
            return ActionResult(
                success=True,
                message=f"{svc} restarted but still in error. Root cause not fixed yet."
            ), reward

        return ActionResult(success=True, message=f"{svc} restarted successfully. Status: healthy."), reward

    def _handle_healthcheck(self, action: Action) -> tuple[ActionResult, float]:
        svc = action.service
        if svc not in self._state["services"]:
            return ActionResult(success=False, message=f"Unknown service '{svc}'."), -0.05

        status = self._state["services"][svc]
        return ActionResult(
            success=True,
            message=f"[{svc} healthcheck] status={status.status} port={status.port}"
        ), 0.01

    def _handle_submit(self) -> tuple[ActionResult, float, bool]:
        all_healthy = all(s.status == "healthy" for s in self._state["services"].values())
        if all_healthy:
            return ActionResult(
                success=True,
                message="All services healthy. Environment fixed successfully!"
            ), 0.5, True
        else:
            unhealthy = [k for k, v in self._state["services"].items() if v.status != "healthy"]
            return ActionResult(
                success=False,
                message=f"Submit failed. Services still unhealthy: {unhealthy}. Fix all services before submitting."
            ), -0.3, True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _recompute_statuses(self):
        statuses = {svc: "healthy" for svc in DEFAULT_ENV}

        directly_broken = set()
        for fault_id in self._active_faults:
            if fault_id not in self._fixed_faults:
                fault = FAULT_CATALOG[fault_id]
                directly_broken.add(fault["service"])

        for svc in ["database", "api", "worker", "proxy"]:
            if svc in directly_broken:
                statuses[svc] = "error"
            else:
                for dep in DEPENDENCIES.get(svc, []):
                    if statuses.get(dep) == "error":
                        statuses[svc] = "error"
                        break

        self._state["services"] = {
            svc: ServiceStatus(name=svc, status=statuses[svc], port=SERVICE_PORTS[svc])
            for svc in DEFAULT_ENV
        }

    def _build_observation(self, last_action_result: ActionResult) -> Observation:
        return Observation(
            step=self._step_count,
            services=self._state["services"],
            last_action_result=last_action_result,
            available_actions=["read_logs", "inspect_env", "edit_env",
                               "restart_service", "run_healthcheck", "submit"],
            done=self._done,
        )
