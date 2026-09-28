"""
Rabbit-2B Core Types & Resource Manager
"""
from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Dict, Any, Optional


class AgentState(str, Enum):
    IDLE = "IDLE"
    UNDERSTAND = "UNDERSTAND"
    DECIDE = "DECIDE"
    PLAN = "PLAN"
    AUTHORIZE = "AUTHORIZE"
    SCHEDULE = "SCHEDULE"
    EXECUTE = "EXECUTE"
    OBSERVE = "OBSERVE"
    VERIFY = "VERIFY"
    RECOVER = "RECOVER"
    REPLAN = "REPLAN"
    COMPLETE = "COMPLETE"
    PAUSED = "PAUSED"
    HALTED = "HALTED"


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    READY = "READY"
    RUNNING = "RUNNING"
    BLOCKED = "BLOCKED"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    ROLLED_BACK = "ROLLED_BACK"
    CANCELLED = "CANCELLED"


class DecisionType(str, Enum):
    ANSWER = "ANSWER"
    TOOL = "TOOL"
    PLAN = "PLAN"
    CLARIFY = "CLARIFY"
    REJECT = "REJECT"
    ESCALATE = "ESCALATE"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class FailureClass(str, Enum):
    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    TOOL_FAILURE = "TOOL_FAILURE"
    COMMAND_FAILURE = "COMMAND_FAILURE"
    TIMEOUT = "TIMEOUT"
    PERMISSION_ERROR = "PERMISSION_ERROR"
    DEPENDENCY_FAILURE = "DEPENDENCY_FAILURE"
    RESOURCE_FAILURE = "RESOURCE_FAILURE"
    ENVIRONMENT_FAILURE = "ENVIRONMENT_FAILURE"
    VERIFICATION_FAILURE = "VERIFICATION_FAILURE"
    UNKNOWN_FAILURE = "UNKNOWN_FAILURE"


@dataclass
class ResourceLimits:
    max_tasks: int = 50
    max_tool_calls: int = 40
    max_retries_per_task: int = 3
    max_runtime_sec: float = 600.0
    max_tokens: int = 32768


@dataclass
class ResourceUsage:
    tasks_executed: int = 0
    tool_calls: int = 0
    total_retries: int = 0
    tokens_generated: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time

    def check_limits(self, limits: ResourceLimits) -> Optional[str]:
        if self.tasks_executed >= limits.max_tasks:
            return f"Exceeded max_tasks limit ({limits.max_tasks})"
        if self.tool_calls >= limits.max_tool_calls:
            return f"Exceeded max_tool_calls limit ({limits.max_tool_calls})"
        if self.elapsed_time >= limits.max_runtime_sec:
            return f"Exceeded max_runtime_sec limit ({limits.max_runtime_sec}s)"
        if self.tokens_generated >= limits.max_tokens:
            return f"Exceeded max_tokens limit ({limits.max_tokens})"
        return None
