"""
Inference script — Dev Environment Debugger (OpenEnv Hackathon)

Required environment variables:
    API_BASE_URL   OpenAI-compatible LLM API endpoint
    MODEL_NAME     Model identifier for inference
    HF_TOKEN       HuggingFace / API key

Optional:
    ENV_BASE_URL   Base URL of the deployed OpenEnv Space
                   (default: http://localhost:7860)
"""

import asyncio
import json
import os
import sys
from typing import List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_missing = [v for v in ("API_BASE_URL", "MODEL_NAME", "HF_TOKEN") if not os.getenv(v)]
if _missing:
    print(f"[ERROR] Missing required environment variables: {', '.join(_missing)}", flush=True)
    sys.exit(1)

API_BASE_URL: str = os.environ["API_BASE_URL"]
MODEL_NAME: str = os.environ["MODEL_NAME"]
HF_TOKEN: str = os.environ["HF_TOKEN"]
ENV_BASE_URL: str = os.getenv("ENV_BASE_URL", "http://localhost:7860")

TASK_IDS = ["task1", "task2", "task3"]
BENCHMARK = "dev-env-debugger"
MAX_STEPS = 20
SUCCESS_SCORE_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Structured logging — [START] / [STEP] / [END]
# ---------------------------------------------------------------------------

def log_start(*, task: str, env: str, model: str) -> None:
    print(
        f"[START] {json.dumps({'task': task, 'env': env, 'model': model})}",
        flush=True,
    )


def log_step(
    *, step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    print(
        f"[STEP] {json.dumps({'step': step, 'action': action, 'reward': reward, 'done': done, 'error': error})}",
        flush=True,
    )


def log_end(*, success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(
        f"[END] {json.dumps({'success': success, 'steps': steps, 'score': score, 'rewards': rewards})}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert DevOps engineer debugging a broken multi-service development environment.

Services: api, worker, database, proxy
Available actions — respond ONLY with valid JSON, nothing else:
  {"type": "read_logs", "service": "<name>"}
  {"type": "inspect_env", "service": "<name>"}
  {"type": "edit_env", "service": "<name>", "key": "<KEY>", "value": "<value>"}
  {"type": "restart_service", "service": "<name>"}
  {"type": "run_healthcheck", "service": "<name>"}
  {"type": "submit"}

STRATEGY:
1. Check which services are in "error" state
2. Read logs for broken services
3. Inspect env vars of the broken service
4. Edit the wrong/missing env var
5. Restart the service
6. Submit when ALL services show "healthy"
"""


def get_model_message(
    client: OpenAI, obs_text: str, history: List[dict]
) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-10:])  # keep last 10 turns to stay within context
    messages.append({"role": "user", "content": obs_text})

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.2,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return '{"type": "submit"}'  # safe fallback — ends episode cleanly


# ---------------------------------------------------------------------------
# HTTP environment adapter
# ---------------------------------------------------------------------------

class EnvAdapter:
    """Thin wrapper around the OpenEnv HTTP API."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def reset(self, task_id: str) -> dict:
        resp = requests.post(
            f"{self.base_url}/reset", params={"task_id": task_id}, timeout=30
        )
        resp.raise_for_status()
        return resp.json()

    def step(self, action: dict) -> dict:
        resp = requests.post(f"{self.base_url}/step", json=action, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def grader(self, task_id: str) -> dict:
        resp = requests.get(
            f"{self.base_url}/grader", params={"task_id": task_id}, timeout=30
        )
        resp.raise_for_status()
        return resp.json()


def obs_to_text(obs: dict) -> str:
    services = obs.get("services", {})
    svc_lines = "\n".join(
        f"  {name}: {s['status']} (port: {s.get('port', 'N/A')})"
        for name, s in services.items()
    )
    last = obs.get("last_action_result", {})
    return (
        f"Step {obs['step']}\n"
        f"Services:\n{svc_lines}\n"
        f"Last action: {'OK' if last.get('success') else 'FAIL'} — {last.get('message', '')}\n"
        f"Done: {obs.get('done', False)}"
    )


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

async def run_task(client: OpenAI, env: EnvAdapter, task_id: str) -> None:
    history: List[dict] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset(task_id)

        for step in range(1, MAX_STEPS + 1):
            if obs.get("done", False):
                break

            obs_text = obs_to_text(obs)
            raw_action = get_model_message(client, obs_text, history)

            error: Optional[str] = None
            try:
                action = json.loads(raw_action)
            except json.JSONDecodeError:
                error = f"Invalid JSON from model: {raw_action}"
                action = {"type": "submit"}

            step_data = env.step(action)
            obs = step_data["observation"]
            reward = float(step_data.get("reward", 0.0))
            done = bool(step_data.get("done", False))

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=raw_action, reward=reward, done=done, error=error)

            history.append({"role": "assistant", "content": raw_action})
            history.append({"role": "user", "content": obs_to_text(obs)})

            if done:
                break

        grader_data = env.grader(task_id)
        grader_result = grader_data.get("grader_result", {})
        score = float(grader_result.get("score", 0.0))
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task {task_id} error: {exc}", flush=True)
        score = 0.0
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = EnvAdapter(ENV_BASE_URL)

    for task_id in TASK_IDS:
        await run_task(client, env, task_id)


if __name__ == "__main__":
    asyncio.run(main())
