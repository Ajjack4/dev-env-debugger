"""
Task definitions and programmatic graders.
Each task defines faults to inject and a grader that scores strictly in (0, 1).
"""

TASKS = {
    "task1": {
        "id": "task1",
        "name": "Single Service Fault",
        "difficulty": "easy",
        "description": (
            "One service is down due to a single misconfigured environment variable. "
            "The API cannot connect to the database because DB_PORT is set to the wrong value. "
            "Read the logs, find the bad config, fix it, and restart the service."
        ),
        "faults": ["wrong_db_port"],
    },
    "task2": {
        "id": "task2",
        "name": "Multi-Fault Diagnosis",
        "difficulty": "medium",
        "description": (
            "Two services are failing due to independent faults. "
            "The API has a wrong database port AND the worker is missing its queue URL. "
            "Both must be fixed for full score. Partial credit for fixing one."
        ),
        "faults": ["wrong_db_port", "missing_queue_url"],
    },
    "task3": {
        "id": "task3",
        "name": "Cascading Failure Debug",
        "difficulty": "hard",
        "description": (
            "The database has wrong credentials, causing the API to log a misleading "
            "'connection timeout' error, which cascades to the worker and proxy. "
            "Logs are misleading — the agent must trace the root cause (DB_PASSWORD on the "
            "database service) rather than patching downstream symptoms."
        ),
        "faults": ["wrong_db_password"],
    },
    "task4": {
        "id": "task4",
        "name": "Triple Fault Storm",
        "difficulty": "hard",
        "description": (
            "Three independent faults hit at once: wrong DB_HOST on the API, "
            "wrong API_HOST on the worker, and wrong PROXY_UPSTREAM on the proxy. "
            "Each service is broken for a different reason. Agent must diagnose and fix all three "
            "without being misled by cascading errors."
        ),
        "faults": ["wrong_db_host", "wrong_api_host", "wrong_api_port"],
    },
    "task5": {
        "id": "task5",
        "name": "Silent Misconfiguration",
        "difficulty": "expert",
        "description": (
            "Two subtle faults with no obvious error messages: DB_PASSWORD is missing on the API "
            "(not wrong — missing entirely), and the worker's API_HOST points to the wrong port. "
            "Services appear to start but immediately fail silently. "
            "The agent must inspect env vars carefully rather than relying on logs alone."
        ),
        "faults": ["missing_db_password", "wrong_api_host"],
    },
}


def _clamp(score: float) -> float:
    """Clamp score to strictly (0, 1) exclusive as required by OpenEnv spec."""
    return max(0.001, min(0.999, round(score, 3)))


def grade_episode(task_id: str, env_state: dict) -> dict:
    """
    Programmatic grader. Returns score strictly in (0, 1) exclusive.
    Called after episode ends (done=True).
    """
    task = TASKS.get(task_id)
    if not task:
        return {"score": _clamp(0.001), "reason": f"Unknown task: {task_id}"}

    services = env_state.get("services", {})
    fixed_faults = set(env_state.get("fixed_faults", []))
    active_faults = set(env_state.get("active_faults", []))
    steps = env_state.get("step", 0)

    all_healthy = all(s["status"] == "healthy" for s in services.values())
    total_faults = len(active_faults)
    faults_fixed = len(fixed_faults)

    # ---------------------------------------------------------------
    # Task 1 — single fault, efficiency bonus
    # ---------------------------------------------------------------
    if task_id == "task1":
        if all_healthy and faults_fixed == total_faults:
            efficiency = max(0.0, 1.0 - max(0, steps - 4) * 0.04)
            score = 0.75 + efficiency * 0.20
            return {
                "score": _clamp(score),
                "reason": f"All services healthy. Fixed in {steps} steps.",
                "faults_fixed": faults_fixed,
                "total_faults": total_faults,
            }
        elif faults_fixed > 0:
            return {
                "score": _clamp(0.35),
                "reason": "Fault fixed but episode not completed correctly.",
                "faults_fixed": faults_fixed,
                "total_faults": total_faults,
            }
        return {
            "score": _clamp(0.05),
            "reason": "No faults fixed.",
            "faults_fixed": 0,
            "total_faults": total_faults,
        }

    # ---------------------------------------------------------------
    # Task 2 — partial credit per fault fixed
    # ---------------------------------------------------------------
    if task_id == "task2":
        if all_healthy and faults_fixed == total_faults:
            efficiency = max(0.0, 1.0 - max(0, steps - 6) * 0.03)
            score = 0.70 + efficiency * 0.25
            return {
                "score": _clamp(score),
                "reason": f"Both faults fixed. All services healthy in {steps} steps.",
                "faults_fixed": faults_fixed,
                "total_faults": total_faults,
            }
        elif faults_fixed == 1:
            return {
                "score": _clamp(0.40),
                "reason": "One of two faults fixed. Partial credit.",
                "faults_fixed": faults_fixed,
                "total_faults": total_faults,
            }
        return {
            "score": _clamp(0.05),
            "reason": "No faults fixed.",
            "faults_fixed": 0,
            "total_faults": total_faults,
        }

    # ---------------------------------------------------------------
    # Task 3 — cascading root cause
    # ---------------------------------------------------------------
    if task_id == "task3":
        if all_healthy and faults_fixed == total_faults:
            efficiency = max(0.0, 1.0 - max(0, steps - 5) * 0.04)
            score = 0.70 + efficiency * 0.25
            return {
                "score": _clamp(score),
                "reason": f"Root cause found and fixed. All services recovered in {steps} steps.",
                "faults_fixed": faults_fixed,
                "total_faults": total_faults,
            }
        elif faults_fixed > 0:
            return {
                "score": _clamp(0.30),
                "reason": "Root cause addressed but services not fully recovered.",
                "faults_fixed": faults_fixed,
                "total_faults": total_faults,
            }
        return {
            "score": _clamp(0.05),
            "reason": "Root cause not identified.",
            "faults_fixed": 0,
            "total_faults": total_faults,
        }

    # ---------------------------------------------------------------
    # Task 4 — triple fault storm (partial credit per fault)
    # ---------------------------------------------------------------
    if task_id == "task4":
        if all_healthy and faults_fixed == total_faults:
            efficiency = max(0.0, 1.0 - max(0, steps - 10) * 0.025)
            score = 0.65 + efficiency * 0.30
            return {
                "score": _clamp(score),
                "reason": f"All 3 faults fixed. All services healthy in {steps} steps.",
                "faults_fixed": faults_fixed,
                "total_faults": total_faults,
            }
        elif faults_fixed == 2:
            return {
                "score": _clamp(0.55),
                "reason": "2 of 3 faults fixed.",
                "faults_fixed": faults_fixed,
                "total_faults": total_faults,
            }
        elif faults_fixed == 1:
            return {
                "score": _clamp(0.28),
                "reason": "1 of 3 faults fixed.",
                "faults_fixed": faults_fixed,
                "total_faults": total_faults,
            }
        return {
            "score": _clamp(0.05),
            "reason": "No faults fixed.",
            "faults_fixed": 0,
            "total_faults": total_faults,
        }

    # ---------------------------------------------------------------
    # Task 5 — silent misconfiguration (expert)
    # ---------------------------------------------------------------
    if task_id == "task5":
        if all_healthy and faults_fixed == total_faults:
            efficiency = max(0.0, 1.0 - max(0, steps - 8) * 0.03)
            score = 0.65 + efficiency * 0.30
            return {
                "score": _clamp(score),
                "reason": f"Both silent faults found and fixed in {steps} steps. Expert diagnosis.",
                "faults_fixed": faults_fixed,
                "total_faults": total_faults,
            }
        elif faults_fixed == 1:
            return {
                "score": _clamp(0.38),
                "reason": "One silent fault found. Second still undetected.",
                "faults_fixed": faults_fixed,
                "total_faults": total_faults,
            }
        return {
            "score": _clamp(0.05),
            "reason": "No faults found. Agent must inspect env vars, not just logs.",
            "faults_fixed": 0,
            "total_faults": total_faults,
        }

    return {"score": _clamp(0.05), "reason": "Unhandled task."}
