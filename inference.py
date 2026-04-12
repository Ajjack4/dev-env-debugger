"""
Inference Script — Dev Environment Debugger (OpenEnv Hackathon)

Required environment variables (injected by validator):
    API_BASE_URL     The API endpoint for the LLM.
    MODEL_NAME       The model identifier to use for inference.
    HF_TOKEN         Your Hugging Face / API key.
    IMAGE_NAME       Docker image name for the environment (if using from_docker_image).
"""

import asyncio
import json
import os
import subprocess
import time
from typing import List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config — match sample inference.py exactly
# ---------------------------------------------------------------------------

IMAGE_NAME: str = os.getenv("IMAGE_NAME", "")

# HF_TOKEN checked first (validator injects HF_TOKEN), API_KEY as fallback
API_KEY: Optional[str] = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL: str = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME: str = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

BENCHMARK: str = "dev-env-debugger"
TASK_IDS: List[str] = ["task1", "task2", "task3"]
MAX_STEPS: int = 20
SUCCESS_SCORE_THRESHOLD: float = 0.5

# ---------------------------------------------------------------------------
# Structured logging — exact format from sample
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_val = str(success).lower()
    print(
        f"[END] success={success_val} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Docker container lifecycle
# ---------------------------------------------------------------------------

_container_id: str = ""


def start_env_container(image_name: str) -> str:
    """Start the environment Docker container. Returns its base URL or empty string."""
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
            for _ in range(12):
                time.sleep(5)
                try:
                    r = requests.get("http://localhost:7860/health", timeout=5)
                    if r.status_code == 200:
                        print("[DEBUG] Container health check passed.", flush=True)
                        return "http://localhost:7860"
                except Exception:
                    pass
            print("[DEBUG] Container health check timed out, continuing anyway.", flush=True)
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
            subprocess.run(["docker", "stop", _container_id], capture_output=True, timeout=15)
            print(f"[DEBUG] Container stopped: {_container_id[:12]}", flush=True)
        except Exception as exc:
            print(f"[DEBUG] stop_env_container error: {exc}", flush=True)
        _container_id = ""


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


def get_model_message(client: OpenAI, obs_text: str, history: List[dict]) -> str:
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
        return (response.choices[0].message.content or "").strip()
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
                "observation": {
                    "step": 0, "services": {}, "done": True,
                    "last_action_result": {"success": False, "message": str(exc)},
                    "available_actions": [],
                },
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

    def close(self) -> None:
        pass  # HTTP adapter — no persistent connection to close


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
# Episode runner — mirrors sample inference.py structure
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
                error = f"invalid_json"
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

    finally:
        try:
            env.close()
        except Exception as exc:
            print(f"[DEBUG] env.close() error: {exc}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# Entry point — mirrors sample inference.py structure
# ---------------------------------------------------------------------------

async def main() -> None:
    # Resolve environment URL
    if IMAGE_NAME:
        env_url = start_env_container(IMAGE_NAME)
        if not env_url:
            env_url = os.getenv("ENV_BASE_URL", "http://localhost:7860")
    else:
        env_url = os.getenv("ENV_BASE_URL", "http://localhost:7860")

    print(f"[DEBUG] env_url={env_url} api_base={API_BASE_URL} model={MODEL_NAME}", flush=True)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = EnvAdapter(env_url)

    try:
        for task_id in TASK_IDS:
            await run_task(client, env, task_id)
    finally:
        stop_env_container()


if __name__ == "__main__":
    asyncio.run(main())
