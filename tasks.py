"""
Task definitions and programmatic graders.
Each task defines faults to inject and a grader that scores 0.0 - 1.0.
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
            "The database has wrong credentials, causing the API to fail with a misleading "
            "'connection timeout' error, which in turn causes the worker and proxy to fail. "
            "Logs are misleading — the agent must trace the root cause (DB_PASSWORD) rather "
            "than patching downstream symptoms."
        ),
        "faults": ["wrong_db_password"],
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
        return {"score": _clamp(0.0), "reason": f"Unknown task: {task_id}"}

    services = env_state.get("services", {})
    fixed_faults = set(env_state.get("fixed_faults", []))
    active_faults = set(env_state.get("active_faults", []))
    remaining = active_faults - fixed_faults
    steps = env_state.get("step", 0)

    all_healthy = all(s["status"] == "healthy" for s in services.values())
    total_faults = len(active_faults)
    faults_fixed = len(fixed_faults)

    # ---------------------------------------------------------------
    # Task 1 grader — single fault, binary but with partial for partial fix
    # ---------------------------------------------------------------
    if task_id == "task1":
        if all_healthy and faults_fixed == total_faults:
            # Efficiency bonus for doing it quickly
            efficiency = max(0.0, 1.0 - (steps - 3) * 0.03)
            score = min(1.0, 0.85 + efficiency * 0.15)
            return {
                "score": _clamp(score),
                "reason": f"All services healthy. Fixed in {steps} steps.",
                "faults_fixed": faults_fixed,
                "total_faults": total_faults,
            }
        elif faults_fixed > 0:
            return {
                "score": _clamp(0.4),
                "reason": "Fault fixed but service not restarted or submit not called.",
                "faults_fixed": faults_fixed,
                "total_faults": total_faults,
            }
        else:
            return {
                "score": _clamp(0.0),
                "reason": "No faults fixed.",
                "faults_fixed": 0,
                "total_faults": total_faults,
            }

    # ---------------------------------------------------------------
    # Task 2 grader — partial credit per fault fixed
    # ---------------------------------------------------------------
    if task_id == "task2":
        if all_healthy and faults_fixed == total_faults:
            efficiency = max(0.0, 1.0 - (steps - 5) * 0.02)
            score = min(1.0, 0.85 + efficiency * 0.15)
            return {
                "score": _clamp(score),
                "reason": f"All services healthy. Both faults fixed in {steps} steps.",
                "faults_fixed": faults_fixed,
                "total_faults": total_faults,
            }
        elif faults_fixed == 1:
            return {
                "score": _clamp(0.45),
                "reason": f"One of two faults fixed. Partial credit.",
                "faults_fixed": faults_fixed,
                "total_faults": total_faults,
            }
        else:
            return {
                "score": _clamp(0.0),
                "reason": "No faults fixed.",
                "faults_fixed": 0,
                "total_faults": total_faults,
            }

    # ---------------------------------------------------------------
    # Task 3 grader — penalizes symptom-patching, rewards root cause fix
    # ---------------------------------------------------------------
    if task_id == "task3":
        if all_healthy and faults_fixed == total_faults:
            # Reward fewer steps heavily (cascading fix is fast if root cause found)
            efficiency = max(0.0, 1.0 - (steps - 4) * 0.04)
            score = min(1.0, 0.80 + efficiency * 0.20)
            return {
                "score": _clamp(score),
                "reason": f"Root cause found and fixed. All services recovered in {steps} steps.",
                "faults_fixed": faults_fixed,
                "total_faults": total_faults,
            }
        elif faults_fixed > 0:
            # Fixed something but not all healthy — partial
            return {
                "score": _clamp(0.3),
                "reason": "Partial fix. Root cause addressed but services not fully recovered.",
                "faults_fixed": faults_fixed,
                "total_faults": total_faults,
            }
        else:
            return {
                "score": _clamp(0.0),
                "reason": "Root cause not identified. All services still broken.",
                "faults_fixed": 0,
                "total_faults": total_faults,
            }

    return {"score": _clamp(0.0), "reason": "Unhandled task."}
