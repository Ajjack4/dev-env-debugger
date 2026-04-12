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

# Accept HF_TOKEN or OPENAI_API_KEY (functional req says OPENAI_API_KEY,
# mandatory instructions say HF_TOKEN — support both so either works)
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN: str = (
    os.getenv("HF_TOKEN")
    or os.getenv("OPENAI_API_KEY")
    or ""
)
ENV_BASE_URL: str = os.getenv("ENV_BASE_URL", "https://ajjack404-dev-env-debugger.hf.space")

if not HF_TOKEN:
    print("[DEBUG] Warning: no API key found (HF_TOKEN / OPENAI_API_KEY). LLM calls will fail.", flush=True)

TASK_IDS = ["task1", "task2", "task3"]
BENCHMARK = "dev-env-debugger"
MAX_STEPS = 20
SUCCESS_SCORE_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Structured logging — [START] / [STEP] / [END]
# ---------------------------------------------------------------------------

def log_start(*, task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    *, step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    error_str = error if error else "none"
    print(f"[STEP] step={step} reward={reward} done={done} error={error_str}", flush=True)


def log_end(*, success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(f"[END] success={success} steps={steps} score={score} rewards={rewards}", flush=True)


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
    client: Optional[OpenAI], obs_text: str, history: List[dict]
) -> str:
    if client is None:
        return '{"type": "submit"}'

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
        import time
        for attempt in range(5):
            try:
                resp = requests.post(
                    f"{self.base_url}/reset", params={"task_id": task_id}, timeout=60
                )
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                print(f"[DEBUG] reset() attempt {attempt+1} failed: {exc}", flush=True)
                if attempt < 4:
                    time.sleep(5)
        return {"step": 0, "services": {}, "last_action_result": {"success": False, "message": "env unreachable"}, "available_actions": [], "done": True}

    def step(self, action: dict) -> dict:
        try:
            resp = requests.post(f"{self.base_url}/step", json=action, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            print(f"[DEBUG] step() failed: {exc}", flush=True)
            return {"observation": {"step": 0, "services": {}, "last_action_result": {"success": False, "message": str(exc)}, "available_actions": [], "done": True}, "reward": 0.0, "done": True, "info": {}}

    def grader(self, task_id: str) -> dict:
        try:
            resp = requests.get(
                f"{self.base_url}/grader", params={"task_id": task_id}, timeout=60
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            print(f"[DEBUG] grader() failed: {exc}", flush=True)
            return {"grader_result": {"score": 0.0, "reason": str(exc)}}


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
    # Use placeholder key if none provided — real failure happens at API call
    # level and is caught inside get_model_message(), not here.
    api_key = HF_TOKEN if HF_TOKEN else "none"
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=api_key)
    except Exception as exc:
        print(f"[DEBUG] OpenAI client init failed: {exc}", flush=True)
        client = None

    env = EnvAdapter(ENV_BASE_URL)

    for task_id in TASK_IDS:
        # Wrap individually — one task failing must not prevent others logging.
        try:
            await run_task(client, env, task_id)
        except Exception as exc:
            print(f"[DEBUG] run_task({task_id}) uncaught: {exc}", flush=True)
            print(f"[END] success=False steps=0 score=0.0 rewards=[]", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        print(f"[DEBUG] Fatal error: {exc}", flush=True)
