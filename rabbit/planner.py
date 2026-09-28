"""
Rabbit-2B Hierarchical Planner, Task DAG & Dependency Scheduler
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Set
import json
import re
from .core import TaskStatus


@dataclass
class TaskNode:
    task_id: str
    description: str
    parent_id: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 1
    dependencies: List[str] = field(default_factory=list)
    required_tools: List[str] = field(default_factory=list)
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    verification_method: str = ""
    retry_limit: int = 3
    retry_count: int = 0
    rollback_strategy: str = ""
    result: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value if isinstance(self.status, TaskStatus) else self.status
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TaskNode":
        status_val = d.get("status", "PENDING")
        if isinstance(status_val, str):
            try:
                status_val = TaskStatus(status_val)
            except ValueError:
                status_val = TaskStatus.PENDING
        return cls(
            task_id=d["task_id"],
            description=d.get("description", ""),
            parent_id=d.get("parent_id"),
            status=status_val,
            priority=d.get("priority", 1),
            dependencies=d.get("dependencies", []),
            required_tools=d.get("required_tools", []),
            preconditions=d.get("preconditions", []),
            postconditions=d.get("postconditions", []),
            verification_method=d.get("verification_method", ""),
            retry_limit=d.get("retry_limit", 3),
            retry_count=d.get("retry_count", 0),
            rollback_strategy=d.get("rollback_strategy", ""),
            result=d.get("result", {}),
        )


class TaskDAG:
    def __init__(self):
        self.nodes: Dict[str, TaskNode] = {}

    def add_task(self, task: TaskNode):
        self.nodes[task.task_id] = task

    def get_task(self, task_id: str) -> Optional[TaskNode]:
        return self.nodes.get(task_id)

    def is_dependency_satisfied(self, task_id: str) -> bool:
        task = self.nodes.get(task_id)
        if not task:
            return False
        for dep_id in task.dependencies:
            dep = self.nodes.get(dep_id)
            if not dep or dep.status != TaskStatus.SUCCESS:
                return False
        return True

    def get_ready_tasks(self) -> List[TaskNode]:
        ready = []
        for task in self.nodes.values():
            if task.status in (TaskStatus.PENDING, TaskStatus.READY):
                if self.is_dependency_satisfied(task.task_id):
                    task.status = TaskStatus.READY
                    ready.append(task)
                else:
                    # check if any dependency failed
                    for dep_id in task.dependencies:
                        dep = self.nodes.get(dep_id)
                        if dep and dep.status == TaskStatus.FAILED:
                            task.status = TaskStatus.BLOCKED
        # Sort ready by priority descending (higher priority first)
        ready.sort(key=lambda t: t.priority, reverse=True)
        return ready

    def mark_success(self, task_id: str, result: Dict[str, Any]):
        if task_id in self.nodes:
            self.nodes[task_id].status = TaskStatus.SUCCESS
            self.nodes[task_id].result = result

    def mark_failed(self, task_id: str, error: str):
        if task_id in self.nodes:
            self.nodes[task_id].status = TaskStatus.FAILED
            self.nodes[task_id].result = {"error": error}
            # Propagate failure to dependent tasks
            for other_task in self.nodes.values():
                if task_id in other_task.dependencies and other_task.status in (TaskStatus.PENDING, TaskStatus.READY):
                    other_task.status = TaskStatus.BLOCKED

    def is_all_completed(self) -> bool:
        if not self.nodes:
            return True
        return all(t.status in (TaskStatus.SUCCESS, TaskStatus.CANCELLED) for t in self.nodes.values())

    def has_unrecoverable_failure(self) -> bool:
        # If any task failed and no more ready tasks can be scheduled
        has_failed = any(t.status == TaskStatus.FAILED for t in self.nodes.values())
        has_ready = len(self.get_ready_tasks()) > 0
        return has_failed and not has_ready

    def to_list(self) -> List[Dict[str, Any]]:
        return [t.to_dict() for t in self.nodes.values()]


class TaskScheduler:
    def __init__(self, dag: TaskDAG):
        self.dag = dag

    def next_task(self) -> Optional[TaskNode]:
        ready = self.dag.get_ready_tasks()
        if not ready:
            return None
        selected = ready[0]
        selected.status = TaskStatus.RUNNING
        return selected


class HierarchicalPlanner:
    """
    Constructs an explicit TaskDAG from an objective.
    Uses model reasoning or structured rule-based decomposition.
    """
    @staticmethod
    def create_plan_from_schema(objective: str, steps: List[Dict[str, Any]]) -> TaskDAG:
        dag = TaskDAG()
        for i, s in enumerate(steps):
            task_id = s.get("id", f"T{i+1}")
            node = TaskNode(
                task_id=task_id,
                description=s.get("description", ""),
                parent_id=s.get("parent_id"),
                priority=s.get("priority", 1),
                dependencies=s.get("dependencies", []),
                required_tools=s.get("required_tools", []),
                preconditions=s.get("preconditions", []),
                postconditions=s.get("postconditions", []),
                verification_method=s.get("verification_method", ""),
                retry_limit=s.get("retry_limit", 3),
                rollback_strategy=s.get("rollback_strategy", "")
            )
            dag.add_task(node)
        return dag
