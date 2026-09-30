"""
Rabbit-2B Master Autonomous Agent Runtime
Coordinates the complete lifecycle:
Understand -> Decide -> Plan -> Authorize -> Schedule -> Execute -> Observe -> Verify -> Recover -> Complete
"""
import os
import re
import sys
import time
import json
from typing import Dict, Any, List, Optional

from .core import AgentState, TaskStatus, DecisionType, ResourceLimits, ResourceUsage
from .planner import TaskNode, TaskDAG, TaskScheduler, HierarchicalPlanner
from .tools import ToolRegistry, PolicyGate, parse_tool_calls
from .memory import PersistentMemory
from .engine import ModelProvider, VerificationEngine, RecoveryEngine
from .decision import LavoirDecisionEngine, DecisionResult


class RabbitExecutiveAgent:
    def __init__(self, model_path: Optional[str] = None):
        self.state = AgentState.IDLE
        self.memory = PersistentMemory()
        self.tools = ToolRegistry()
        self.verifier = VerificationEngine(self.tools)
        self.recovery = RecoveryEngine(self.tools, self.memory)
        self.model = ModelProvider(model_path) if model_path else ModelProvider()
        self.decision_engine = LavoirDecisionEngine()
        self.limits = ResourceLimits()
        self.usage = ResourceUsage()
        self.current_dag: Optional[TaskDAG] = None
        self.current_run_id: str = ""
        self.is_paused: bool = False

    def quick_decide(self, objective: str) -> DecisionResult:
        """LAVOIR Single-pass decision with Gini-impurity capped Value of Information."""
        return self.decision_engine.evaluate(objective)

    def generate_plan(self, objective: str) -> TaskDAG:
        """
        Decomposes high-level objective into executable TaskDAG.
        Uses structured schema or model reasoning.
        """
        self.state = AgentState.PLAN
        print(f"\n[Rabbit Planner] Decomposing objective into Task DAG: '{objective}'")

        # System prompt for planning
        tools_list = ", ".join(self.tools.tools.keys())
        planning_prompt = f"""You are Rabbit's Hierarchical Planner.
Decompose this objective into an ordered JSON list of executable tasks:
Objective: {objective}

Available tools: {tools_list}

Output strictly valid JSON with this structure:
[
  {{
    "id": "T1",
    "description": "Task description",
    "dependencies": [],
    "required_tools": ["write_file"],
    "postconditions": ["file_exists: path/to/file"],
    "verification_method": "python -c 'import X'",
    "rollback_strategy": "file_rollback"
  }}
]
"""
        messages = [{"role": "user", "content": planning_prompt}]
        try:
            resp = self.model.generate(messages, max_tokens=600, temperature=0.3)
            # Extract JSON block
            json_match = re.search(r'\[[\s\S]*\]', resp)
            if json_match:
                steps = json.loads(json_match.group(0))
                return HierarchicalPlanner.create_plan_from_schema(objective, steps)
        except Exception as e:
            print(f"[Planner Warning] Model planning fallback: {e}")

        # Fallback: Construct standard deterministic 3-stage plan
        dag = TaskDAG()
        dag.add_task(TaskNode(
            task_id="T1",
            description=f"Prepare environment & files for: {objective}",
            dependencies=[],
            required_tools=["write_file", "execute_cmd"]
        ))
        dag.add_task(TaskNode(
            task_id="T2",
            description=f"Execute core operations for: {objective}",
            dependencies=["T1"],
            required_tools=["execute_cmd"]
        ))
        dag.add_task(TaskNode(
            task_id="T3",
            description=f"Verify completion and test postconditions",
            dependencies=["T2"],
            required_tools=["verify_condition"]
        ))
        return dag

    def execute_plan(self, dag: TaskDAG, run_id: str) -> Dict[str, Any]:
        self.current_dag = dag
        self.current_run_id = run_id
        scheduler = TaskScheduler(dag)

        print("\n" + "=" * 65)
        print(f"  EXECUTING TASK GRAPH (Run ID: {run_id})")
        print(f"  Total Tasks: {len(dag.nodes)}")
        for t in dag.nodes.values():
            deps = f" (depends on {t.dependencies})" if t.dependencies else ""
            print(f"    [{t.task_id}] {t.description}{deps}")
        print("=" * 65)

        while not dag.is_all_completed():
            # Check resource budget
            limit_err = self.usage.check_limits(self.limits)
            if limit_err:
                print(f"\n[RESOURCE LIMIT EXCEEDED] {limit_err}. Pausing run.")
                self.state = AgentState.HALTED
                return {"status": "BLOCKED", "reason": limit_err, "evidence": {}}

            if self.is_paused:
                print("\n[AGENT PAUSED] Type '/resume' to continue.")
                time.sleep(2)
                continue

            self.state = AgentState.SCHEDULE
            task = scheduler.next_task()

            if not task:
                if dag.has_unrecoverable_failure():
                    print("\n[DAG BLOCKED] Unrecoverable failure in task dependencies.")
                    self.state = AgentState.HALTED
                    return {"status": "FAILED", "reason": "Dependency failure", "evidence": dag.to_list()}
                time.sleep(0.5)
                continue

            print(f"\n>>> [{task.task_id}] EXECUTING: {task.description}")
            self.state = AgentState.EXECUTE
            self.usage.tasks_executed += 1

            # Dispatch task to model to formulate concrete tool action
            messages = [
                {"role": "system", "content": "You are Rabbit, an autonomous agent. Formulate the exact tool call XML to execute the task."},
                {"role": "user", "content": f"Task: {task.description}\nAvailable tools: {list(self.tools.tools.keys())}\nEmit tool call XML: <function name='...'><param name='...'>value</param></function>"}
            ]

            resp = self.model.generate(messages, max_tokens=400, temperature=0.3)
            tool_calls = parse_tool_calls(resp)

            if not tool_calls:
                # If model gave pure text or no tool call needed, treat as direct observation
                obs = {"success": True, "output": resp}
            else:
                call = tool_calls[0]
                t_name, t_params = call["name"], call["params"]
                print(f"  [Tool Dispatch] {t_name}({t_params})")
                self.usage.tool_calls += 1
                obs = self.tools.execute(t_name, t_params)

            if obs.get("authorization_required"):
                print(f"  [AUTHORIZATION REQUIRED] {obs.get('error')}")
                self.state = AgentState.PAUSED
                return {
                    "status": "AUTHORIZATION_REQUIRED",
                    "confirm_token": obs["confirm_token"],
                    "pending_tool": obs.get("tool"),
                    "pending_args": obs.get("args"),
                    "risk": obs.get("risk"),
                    "run_id": run_id,
                }

            self.state = AgentState.OBSERVE
            print(f"  [Observation]: {json.dumps(obs, default=str)[:300]}")

            # Verification step
            self.state = AgentState.VERIFY
            verification = self.verifier.verify_task(task, obs)
            print(f"  [Verification]: {'PASS' if verification['passed'] else 'FAIL'} (Evidence: {verification.get('evidence', '')[:100]})")

            if verification["passed"]:
                dag.mark_success(task.task_id, {"observation": obs, "verification": verification})
                self.memory.save_task(task.to_dict(), run_id)
            else:
                self.state = AgentState.RECOVER
                recovered = self.recovery.handle_failure(task, obs, verification, run_id)
                self.memory.save_task(task.to_dict(), run_id)
                if not recovered:
                    dag.mark_failed(task.task_id, verification.get("failure_reason", "Verification failed"))

        self.state = AgentState.COMPLETE
        success_count = sum(1 for t in dag.nodes.values() if t.status == TaskStatus.SUCCESS)
        total_count = len(dag.nodes)
        status_label = "COMPLETED" if success_count == total_count else "PARTIALLY COMPLETED"

        self.memory.record_episode(
            run_id=run_id,
            objective=run_id,
            status=status_label,
            total=total_count,
            succeeded=success_count,
            duration=self.usage.elapsed_time,
            summary=f"Completed {success_count}/{total_count} tasks"
        )

        return {
            "status": status_label,
            "tasks_total": total_count,
            "tasks_succeeded": success_count,
            "elapsed_time": round(self.usage.elapsed_time, 2),
            "evidence": {t.task_id: t.result for t in dag.nodes.values()}
        }

    def run(self, objective: str = "", confirm_token: str = "") -> Dict[str, Any]:
        # Approval path: execute a previously gated action after user confirm.
        if confirm_token:
            obs = self.tools.confirm(confirm_token)
            return {"status": "CONFIRMED_EXECUTION", "result": obs}

        run_id = f"run_{int(time.time()*1000)}"
        self.usage = ResourceUsage()
        self.state = AgentState.UNDERSTAND

        print("\n" + "=" * 65)
        print(f"  RABBIT AUTONOMOUS EXECUTIVE RUN: {run_id}")
        print(f"  Objective: {objective}")
        print("=" * 65)

        self.state = AgentState.DECIDE
        d_res = self.quick_decide(objective)
        print(f"[LAVOIR Decision Engine] Route: {d_res.decision.value} (Confidence: {d_res.confidence*100:.1f}%, Gini: {d_res.gini_impurity:.3f}, Latency: {d_res.latency_ms:.2f}ms)")

        if d_res.decision == DecisionType.CLARIFY:
            print(f"\n[Rabbit Question (Highest VOI: {d_res.selected_slot})]: {d_res.question_to_ask}")
            self.state = AgentState.COMPLETE
            return {"status": "CLARIFICATION_REQUIRED", "slot": d_res.selected_slot, "question": d_res.question_to_ask}

        if d_res.decision == DecisionType.ANSWER:
            messages = [{"role": "user", "content": objective}]
            resp = self.model.generate(messages, max_tokens=512)
            self.state = AgentState.COMPLETE
            return {"status": "COMPLETED", "decision": "ANSWER", "result": resp}

        if d_res.decision == DecisionType.TOOL:
            # Single tool execution
            messages = [
                {"role": "system", "content": "You are Rabbit. Emit tool call XML: <function name='...'><param name='...'>value</param></function>"},
                {"role": "user", "content": objective}
            ]
            resp = self.model.generate(messages, max_tokens=256)
            calls = parse_tool_calls(resp)
            if calls:
                obs = self.tools.execute(calls[0]["name"], calls[0]["params"])
                if obs.get("authorization_required"):
                    self.state = AgentState.PAUSED
                    return {
                        "status": "AUTHORIZATION_REQUIRED",
                        "confirm_token": obs["confirm_token"],
                        "pending_tool": obs.get("tool"),
                        "pending_args": obs.get("args"),
                        "risk": obs.get("risk"),
                    }
                self.state = AgentState.COMPLETE
                return {"status": "COMPLETED", "decision": "TOOL", "result": obs}
            return {"status": "COMPLETED", "decision": "TOOL", "result": resp}

        # Hierarchical Planner Track
        dag = self.generate_plan(objective)
        result = self.execute_plan(dag, run_id)
        return result

    def handle_command(self, cmd: str):
        c = cmd.strip().lower()
        if c == "/status":
            print(f"\n[Rabbit Status] State: {self.state.value} | Tasks Executed: {self.usage.tasks_executed} | Elapsed: {self.usage.elapsed_time:.1f}s")
        elif c == "/plan":
            if self.current_dag:
                print("\n[Current Plan DAG]:")
                for t in self.current_dag.nodes.values():
                    print(f"  [{t.status.value}] {t.task_id}: {t.description} (deps: {t.dependencies})")
            else:
                print("No active plan.")
        elif c == "/pause":
            self.is_paused = True
            print("[Rabbit] Paused execution.")
        elif c == "/resume":
            self.is_paused = False
            print("[Rabbit] Resumed execution.")
        elif c == "/stop":
            self.state = AgentState.HALTED
            print("[Rabbit] Execution stopped.")
        else:
            print(f"Unknown command: {cmd}")
