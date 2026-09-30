"""
Rabbit-2B Verification Engine, Recovery System, Model Provider & Execution Engine
"""
import os
import json
import time
from typing import Dict, Any, List, Optional

from .core import AgentState, TaskStatus, FailureClass, ResourceUsage, ResourceLimits
from .planner import TaskNode, TaskDAG, TaskScheduler
from .tools import ToolRegistry, parse_tool_calls
from .memory import PersistentMemory


# Model configuration (env-overridable, no hardcoded machine paths):
#   RABBIT_MODEL_URL  - OpenAI-compatible endpoint (e.g. HERMIT's shared vLLM
#                       server: http://localhost:8000/v1). Preferred when set:
#                       one resident model serves both HERMIT and Rabbit.
#   RABBIT_MODEL_NAME - model name to request at that endpoint.
#   RABBIT_MODEL_PATH - local weights dir for the offline transformers fallback.
DEFAULT_MODEL_PATH = os.environ.get(
    "RABBIT_MODEL_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "Rabbit-2B")
)

class ModelProvider:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.model_url = os.environ.get("RABBIT_MODEL_URL", "").strip()
        self.model_name = os.environ.get("RABBIT_MODEL_NAME", "MiniCPM5-2B-FP8")
        self._model = None
        self._tokenizer = None

    def load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print(f"[Rabbit ModelProvider] Loading local weights: {self.model_path}...")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            self._model.eval()

    def _generate_http(self, messages: List[Dict[str, str]], max_tokens: int, temperature: float) -> str:
        """OpenAI-compatible chat/completions against a shared server (e.g. vLLM)."""
        import urllib.request
        payload = json.dumps({
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }).encode("utf-8")
        req = urllib.request.Request(
            self.model_url.rstrip("/") + "/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()

    def generate(self, messages: List[Dict[str, str]], max_tokens: int = 512, temperature: float = 0.4) -> str:
        if self.model_url:
            try:
                return self._generate_http(messages, max_tokens, temperature)
            except Exception as e:
                print(f"[Rabbit ModelProvider] HTTP endpoint failed ({e}); falling back to local weights.")
        self.load()
        import torch
        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9
            )
        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class VerificationEngine:
    def __init__(self, tools: ToolRegistry):
        self.tools = tools

    def verify_task(self, task: TaskNode, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Independent verification step.
        Requires evidence: actual state vs expected state.
        """
        # If explicit verification method is defined, run it
        if task.verification_method:
            res = self.tools.execute("verify_condition", {"test_cmd": task.verification_method})
            passed = res.get("verified", False)
            return {
                "passed": passed,
                "method": task.verification_method,
                "evidence": res.get("stdout") or res.get("stderr") or ("Pass" if passed else "Fail"),
                "failure_reason": None if passed else "Verification command exited non-zero"
            }

        # If postconditions defined (e.g., file paths that must exist)
        for post in task.postconditions:
            if post.startswith("file_exists:"):
                fpath = post.split("file_exists:")[1].strip()
                chk = self.tools.execute("file_exists", {"path": fpath})
                if not chk.get("exists", False):
                    return {
                        "passed": False,
                        "method": "file_exists",
                        "evidence": f"File '{fpath}' does not exist on disk",
                        "failure_reason": f"Expected file {fpath} was not created"
                    }

        # Otherwise verify the tool execution was successful
        tool_success = observation.get("success", False)
        return {
            "passed": tool_success,
            "method": "tool_execution_status",
            "evidence": observation.get("stdout") or observation.get("status") or str(observation),
            "failure_reason": observation.get("error") if not tool_success else None
        }


class RecoveryEngine:
    def __init__(self, tools: ToolRegistry, memory: PersistentMemory):
        self.tools = tools
        self.memory = memory

    def handle_failure(self, task: TaskNode, observation: Dict[str, Any], verification: Dict[str, Any], run_id: str) -> bool:
        """
        Returns True if task can be retried or recovered, False if unrecoverable.
        """
        task.retry_count += 1
        fail_class = observation.get("failure_class") or FailureClass.VERIFICATION_FAILURE.value
        reason = verification.get("failure_reason") or observation.get("error") or "Unknown error"

        print(f"\n[RECOVERY ENGINE] Task '{task.task_id}' FAILED ({fail_class}). Attempt {task.retry_count}/{task.retry_limit}")
        print(f"  Reason: {reason}")

        self.memory.log_event(run_id, task.task_id, "FAILURE_DETECTED", fail_class, 0.0, {
            "reason": reason, "retry_count": task.retry_count
        })

        # Rollback check
        if task.rollback_strategy == "file_rollback":
            for post in task.postconditions:
                if post.startswith("file_exists:"):
                    fpath = post.split("file_exists:")[1].strip()
                    if self.tools.rollback_file(fpath):
                        print(f"  [Rollback] Restored snapshot for {fpath}")

        # Check retry limit
        if task.retry_count < task.retry_limit:
            task.status = TaskStatus.READY
            print(f"  [Recovery Action] Scheduled retry for task {task.task_id}")
            self.memory.log_event(run_id, task.task_id, "RECOVERY_RETRY", "SCHEDULED")
            return True

        task.status = TaskStatus.FAILED
        print(f"  [Recovery Exhausted] Task {task.task_id} failed permanently.")
        self.memory.log_event(run_id, task.task_id, "RECOVERY_EXHAUSTED", "FAILED")
        return False
