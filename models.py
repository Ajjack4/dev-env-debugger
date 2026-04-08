"""
Typed Pydantic models for OpenEnv spec compliance.
"""

from typing import Optional
from pydantic import BaseModel


class ServiceStatus(BaseModel):
    name: str
    status: str  # "healthy" | "error" | "stopped"
    port: Optional[int] = None


class ActionResult(BaseModel):
    success: bool
    message: str


class Observation(BaseModel):
    step: int
    services: dict[str, ServiceStatus]
    last_action_result: ActionResult
    available_actions: list[str]
    done: bool


class Action(BaseModel):
    type: str                        # read_logs | inspect_env | edit_env | restart_service | run_healthcheck | submit
    service: Optional[str] = None   # target service name
    key: Optional[str] = None       # for edit_env
    value: Optional[str] = None     # for edit_env


class Reward(BaseModel):
    value: float
    reason: str


class TaskInfo(BaseModel):
    id: str
    name: str
    description: str
    difficulty: str
    faults: list[str]
