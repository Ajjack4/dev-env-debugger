"""
Inference script — Dev Environment Debugger (OpenEnv Hackathon)

Environment variables (injected by validator):
    API_BASE_URL     LiteLLM proxy endpoint for LLM calls
    API_KEY          API key for the LiteLLM proxy
    MODEL_NAME       Model identifier (default: gpt-4o-mini)
    LOCAL_IMAGE_NAME Docker image name to start the environment container

Optional (for local testing):
    ENV_BASE_URL     Override environment URL instead of starting Docker
    HF_TOKEN         Alias for API_KEY (local testing)
    OPENAI_API_KEY   Alias for API_KEY (local testing)
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from typing import List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config — follow validator spec exactly
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o-mini")

# Validator injects API_KEY. Accept aliases for local testing.
API_KEY: str = (
    os.getenv("API_KEY")
    or os.getenv("HF_TOKEN")
    or os.getenv("OPENAI_API_KEY")
    or ""
)

LOCAL_IMAGE_NAME: str = os.getenv("LOCAL_IMAGE_NAME", "")
# Validator runs our Docker container at localhost:7860 before running inference.py.
# LOCAL_IMAGE_NAME is used when we need to start it ourselves.
ENV_BASE_URL: str = os.getenv("ENV_BASE_URL", "http://localhost:7860")

if not API_KEY:
    print("[DEBUG] Warning: no API key set (API_KEY / HF_TOKEN / OPENAI_API_KEY).", flush=True)

TASK_IDS = ["task1", "task2", "task3"]
BENCHMARK = "dev-env-debugger"
MAX_STEPS = 20
SUCCESS_SCORE_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Docker container lifecycle
# ---------------------------------------------------------------------------

_container_id: str = ""


def start_env_container(image_name: str) -> str:
    """Start the environment Docker container. Returns its base URL."""
    global _container_id
    print(f"[DEBUG] Starting container from image: {image_name}", flush=True)
    try:
        result = subprocess.run(
            ["docker", "run", "-d", "--rm", "-p", "7860:7860", image_name],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            _container_id = result.stdout.strip()
            print(f"[DEBUG] Container started: {_container_id[:12]}", flush=True)
            # Wait for the server inside the container to be ready
            for _ in range(12):
                time.sleep(5)
                try:
                    r = requests.get("http://localhost:7860/health", timeout=5)
                    if r.status_code == 200:
                        print("[DEBUG] Container health check passed.", flush=True)
                        return "http://localhost:7860"
                except Exception:
                    pass
            print("[DEBUG] Container started but health check timed out.", flush=True)
            return "http://localhost:7860"
        else:
            print(f"[DEBUG] docker run failed: {result.stderr.strip()}", flush=True)
    except FileNotFoundError:
        print("[DEBUG] docker not found in PATH.", flush=True)
    except Exception as exc:
        print(f"[DEBUG] start_env_container error: {exc}", flush=True)
    return ""


def stop_env_container() -> None:
    global _container_id
    if _container_id:
        try:
            subprocess.run(
                ["docker", "stop", _container_id],
                capture_output=True, timeout=15,
            )
            print(f"[DEBUG] Container stopped: {_container_id[:12]}", flush=True)
        except Exception as exc:
            print(f"[DEBUG] stop_env_container error: {exc}", flush=True)
        _container_id = ""


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
    messages.extend(history[-10:])
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
        return '{"type": "submit"}'


# ---------------------------------------------------------------------------
# HTTP environment adapter
# ---------------------------------------------------------------------------

class EnvAdapter:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def reset(self, task_id: str) -> dict:
        for attempt in range(5):
            try:
                resp = requests.post(
                    f"{self.base_url}/reset", params={"task_id": task_id}, timeout=30
                )
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                print(f"[DEBUG] reset() attempt {attempt + 1} failed: {exc}", flush=True)
                if attempt < 4:
                    time.sleep(3)
        return {
            "step": 0, "services": {}, "done": True,
            "last_action_result": {"success": False, "message": "env unreachable"},
            "available_actions": [],
        }

    def step(self, action: dict) -> dict:
        try:
            resp = requests.post(f"{self.base_url}/step", json=action, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            print(f"[DEBUG] step() failed: {exc}", flush=True)
            return {
                "observation": {"step": 0, "services": {}, "done": True,
                                "last_action_result": {"success": False, "message": str(exc)},
                                "available_actions": []},
                "reward": 0.0, "done": True, "info": {},
            }

    def grader(self, task_id: str) -> dict:
        try:
            resp = requests.get(
                f"{self.base_url}/grader", params={"task_id": task_id}, timeout=30
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
        f"Step {obs.get('step', 0)}\n"
        f"Services:\n{svc_lines}\n"
        f"Last action: {'OK' if last.get('success') else 'FAIL'} — {last.get('message', '')}\n"
        f"Done: {obs.get('done', False)}"
    )


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

async def run_task(client: Optional[OpenAI], env: EnvAdapter, task_id: str) -> None:
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
                error = f"Invalid JSON: {raw_action}"
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

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    # Resolve environment URL: Docker container > explicit ENV_BASE_URL > HF Space
    if LOCAL_IMAGE_NAME:
        env_url = start_env_container(LOCAL_IMAGE_NAME)
        if not env_url:
            env_url = ENV_BASE_URL  # already defaults to localhost:7860
    else:
        env_url = ENV_BASE_URL  # localhost:7860 or whatever validator set

    print(f"[DEBUG] Using env URL: {env_url}", flush=True)
    print(f"[DEBUG] Using API_BASE_URL: {API_BASE_URL}", flush=True)
    print(f"[DEBUG] Using MODEL_NAME: {MODEL_NAME}", flush=True)

    # Initialize OpenAI client pointed at validator's LiteLLM proxy
    client: Optional[OpenAI] = None
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "none")
    except Exception as exc:
        print(f"[DEBUG] OpenAI client init failed: {exc}", flush=True)

    env = EnvAdapter(env_url)

    try:
        for task_id in TASK_IDS:
            try:
                await run_task(client, env, task_id)
            except Exception as exc:
                print(f"[DEBUG] run_task({task_id}) uncaught: {exc}", flush=True)
                print(f"[END] success=False steps=0 score=0.0 rewards=[]", flush=True)
    finally:
        stop_env_container()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        print(f"[DEBUG] Fatal error: {exc}", flush=True)
