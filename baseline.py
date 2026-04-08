"""
Baseline inference script.
Runs an LLM agent (via OpenAI API) against all 3 tasks and reports scores.

Usage:
    export OPENAI_API_KEY=your_key_here
    python baseline.py

    # For JSON output (used by /baseline endpoint):
    python baseline.py --output-json
"""

import os
import sys
import json
import argparse
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE = os.getenv("ENV_BASE_URL", "http://localhost:7860")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL = os.getenv("BASELINE_MODEL", "gpt-4o-mini")
MAX_STEPS = 20

SYSTEM_PROMPT = """You are an expert DevOps engineer debugging a broken multi-service development environment.

You will receive the current state of services (healthy/error) and the result of your last action.
Your goal is to diagnose faults and fix them so ALL services reach "healthy" status.

Services: api, worker, database, proxy
Available actions:
  - read_logs: read error logs for a service (do this first when a service is broken)
  - inspect_env: view environment variables for a service
  - edit_env: change an environment variable (requires key and value)
  - restart_service: restart a service after fixing its config
  - run_healthcheck: check current status of a service
  - submit: declare done when all services are healthy

STRATEGY:
1. Look at which services are in "error" state
2. Read logs for broken services to understand the error
3. Inspect env vars of the broken service
4. Edit the wrong/missing env var
5. Restart the service
6. Submit when all services show "healthy"

Respond ONLY with a valid JSON action object. Nothing else. Examples:
{"type": "read_logs", "service": "api"}
{"type": "inspect_env", "service": "database"}  
{"type": "edit_env", "service": "api", "key": "DB_PORT", "value": "5432"}
{"type": "restart_service", "service": "api"}
{"type": "submit"}
"""


def call_openai(messages: list) -> str:
    """Call OpenAI chat completions API."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set in environment variables.")

    import urllib.request
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 200,
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"].strip()


def obs_to_text(obs: dict) -> str:
    """Convert observation dict to readable text for the LLM."""
    services = obs.get("services", {})
    svc_lines = "\n".join(
        f"  {name}: {s['status']} (port: {s.get('port', 'N/A')})"
        for name, s in services.items()
    )
    last = obs.get("last_action_result", {})
    return (
        f"Step {obs['step']}\n"
        f"Services:\n{svc_lines}\n"
        f"Last action result: {'✓' if last.get('success') else '✗'} {last.get('message', '')}\n"
        f"Done: {obs.get('done', False)}"
    )


def run_episode(task_id: str, verbose: bool = True) -> dict:
    """Run one full episode for a task. Returns grader result."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Running Task: {task_id}")
        print('='*60)

    # Reset
    resp = requests.post(f"{API_BASE}/reset", params={"task_id": task_id})
    obs = resp.json()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    step = 0

    while not obs.get("done", False) and step < MAX_STEPS:
        step += 1
        obs_text = obs_to_text(obs)

        if verbose:
            print(f"\n[Step {step}] Observation:\n{obs_text}")

        messages.append({"role": "user", "content": obs_text})

        # Get action from LLM
        try:
            raw_action = call_openai(messages)
        except Exception as e:
            if verbose:
                print(f"  LLM error: {e}")
            break

        if verbose:
            print(f"  Agent action: {raw_action}")

        # Parse and validate action
        try:
            action_dict = json.loads(raw_action)
        except json.JSONDecodeError:
            if verbose:
                print(f"  Could not parse action JSON: {raw_action}")
            messages.append({"role": "assistant", "content": raw_action})
            messages.append({"role": "user", "content": "Invalid JSON. Respond ONLY with a valid JSON action."})
            continue

        messages.append({"role": "assistant", "content": raw_action})

        # Step environment
        step_resp = requests.post(f"{API_BASE}/step", json=action_dict)
        step_data = step_resp.json()
        obs = step_data["observation"]
        reward = step_data["reward"]
        done = step_data["done"]

        if verbose:
            print(f"  Reward: {reward} | Done: {done}")

        if done:
            break

    # Get grader score
    grader_resp = requests.get(f"{API_BASE}/grader", params={"task_id": task_id})
    grader_data = grader_resp.json()
    result = grader_data["grader_result"]

    if verbose:
        print(f"\n[RESULT] Task {task_id}: score={result['score']} | {result['reason']}")

    return {
        "task_id": task_id,
        "score": result["score"],
        "reason": result["reason"],
        "steps_taken": step,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", action="store_true", help="Output scores as JSON")
    parser.add_argument("--task", type=str, default=None, help="Run single task only")
    args = parser.parse_args()

    verbose = not args.output_json
    tasks = [args.task] if args.task else ["task1", "task2", "task3"]

    results = []
    for task_id in tasks:
        try:
            result = run_episode(task_id, verbose=verbose)
            results.append(result)
        except Exception as e:
            results.append({
                "task_id": task_id,
                "score": 0.0,
                "reason": f"Error: {str(e)}",
                "steps_taken": 0,
            })

    avg_score = sum(r["score"] for r in results) / len(results)

    if args.output_json:
        output = {"tasks": results, "average_score": round(avg_score, 3)}
        print(json.dumps(output))
    else:
        print(f"\n{'='*60}")
        print("BASELINE SCORES SUMMARY")
        print('='*60)
        for r in results:
            print(f"  {r['task_id']:10} score={r['score']:.3f}  steps={r['steps_taken']}  {r['reason']}")
        print(f"  {'AVERAGE':10} score={avg_score:.3f}")
        print('='*60)


if __name__ == "__main__":
    main()
