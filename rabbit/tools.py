"""
Rabbit-2B Tool Registry, OS Controller, Hardware Telemetry & Policy Gate
"""
import os
import re
import sys
import json
import shutil
import tempfile
import subprocess
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from .core import RiskLevel, FailureClass


@dataclass
class ToolDef:
    name: str
    description: str
    risk: RiskLevel
    params: List[str]
    timeout_sec: int = 30
    requires_sandbox: bool = False


class PolicyGate:
    def __init__(self, auto_approve_medium: bool = True):
        self.auto_approve_medium = auto_approve_medium

    def check(self, tool: ToolDef, args: Dict[str, Any]) -> bool:
        if tool.risk == RiskLevel.LOW:
            return True
        if tool.risk == RiskLevel.MEDIUM and self.auto_approve_medium:
            return True
        # For HIGH or CRITICAL, require explicit prompt
        print(f"\n[POLICY GATE] Tool '{tool.name}' has risk level {tool.risk.value}.")
        print(f"  Arguments: {json.dumps(args)}")
        try:
            confirm = input("  Authorize execution? [y/N]: ").strip().lower()
            return confirm == "y"
        except (KeyboardInterrupt, EOFError):
            return False


class ToolRegistry:
    def __init__(self, policy: Optional[PolicyGate] = None):
        self.tools: Dict[str, ToolDef] = {}
        self.policy = policy or PolicyGate()
        self.snapshots: Dict[str, str] = {}  # file_path -> original_content
        self._register_all()

    def _register_all(self):
        # Shell & Execution
        self.register(ToolDef("execute_cmd", "Execute shell/terminal command", RiskLevel.MEDIUM, ["command"]))
        # Filesystem
        self.register(ToolDef("read_file", "Read text from file", RiskLevel.LOW, ["path"]))
        self.register(ToolDef("write_file", "Write text to file (snapshots on overwrite)", RiskLevel.MEDIUM, ["path", "content"]))
        self.register(ToolDef("delete_file", "Delete a file (with snapshot)", RiskLevel.HIGH, ["path"]))
        self.register(ToolDef("list_dir", "List directory contents", RiskLevel.LOW, ["path"]))
        self.register(ToolDef("file_exists", "Check if file or directory exists", RiskLevel.LOW, ["path"]))
        # OS & Telemetry
        self.register(ToolDef("get_system_info", "Get CPU, RAM and OS version", RiskLevel.LOW, []))
        self.register(ToolDef("get_gpu_status", "Get live GPU VRAM, compute and temperature", RiskLevel.LOW, []))
        self.register(ToolDef("list_processes", "List running processes matching name pattern", RiskLevel.LOW, ["pattern"]))
        # Git Controller
        self.register(ToolDef("git_status", "Run git status in directory", RiskLevel.LOW, ["path"]))
        self.register(ToolDef("git_diff", "Run git diff in directory", RiskLevel.LOW, ["path"]))
        # Verification
        self.register(ToolDef("verify_condition", "Assert command exits with code 0", RiskLevel.LOW, ["test_cmd"]))

    def register(self, t: ToolDef):
        self.tools[t.name] = t

    def rollback_file(self, path: str) -> bool:
        if path in self.snapshots:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.snapshots[path])
            return True
        return False

    def execute(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name not in self.tools:
            return {"success": False, "error": f"Tool '{name}' not found", "failure_class": FailureClass.TOOL_FAILURE.value}

        tool = self.tools[name]
        if not self.policy.check(tool, args):
            return {"success": False, "error": "Execution denied by policy", "failure_class": FailureClass.PERMISSION_ERROR.value}

        try:
            if name == "execute_cmd":
                cmd = args.get("command", "")
                cwd = args.get("cwd", os.getcwd())
                res = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=tool.timeout_sec)
                return {
                    "stdout": res.stdout,
                    "stderr": res.stderr,
                    "exit_code": res.returncode,
                    "success": res.returncode == 0,
                    "failure_class": FailureClass.COMMAND_FAILURE.value if res.returncode != 0 else None
                }

            elif name == "read_file":
                p = args.get("path", "")
                if not os.path.exists(p):
                    return {"success": False, "error": f"File '{p}' does not exist", "failure_class": FailureClass.ENVIRONMENT_FAILURE.value}
                with open(p, "r", encoding="utf-8", errors="replace") as f:
                    return {"success": True, "content": f.read()}

            elif name == "write_file":
                p = args.get("path", "")
                content = args.get("content", "")
                if os.path.exists(p) and p not in self.snapshots:
                    with open(p, "r", encoding="utf-8", errors="replace") as f:
                        self.snapshots[p] = f.read()
                os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
                with open(p, "w", encoding="utf-8") as f:
                    f.write(content)
                return {"success": True, "status": f"Wrote {len(content)} chars to {p}", "path": p}

            elif name == "delete_file":
                p = args.get("path", "")
                if os.path.exists(p):
                    with open(p, "r", encoding="utf-8", errors="replace") as f:
                        self.snapshots[p] = f.read()
                    os.remove(p)
                    return {"success": True, "status": f"Deleted {p}"}
                return {"success": False, "error": "File does not exist"}

            elif name == "list_dir":
                p = args.get("path", ".")
                entries = os.listdir(p) if os.path.isdir(p) else []
                return {"success": True, "entries": entries, "count": len(entries)}

            elif name == "file_exists":
                p = args.get("path", "")
                exists = os.path.exists(p)
                return {"success": True, "exists": exists, "is_file": os.path.isfile(p), "is_dir": os.path.isdir(p)}

            elif name == "get_system_info":
                import platform
                return {
                    "success": True,
                    "os": platform.platform(),
                    "python": sys.version.split()[0],
                    "cpu_count": os.cpu_count()
                }

            elif name == "get_gpu_status":
                res = subprocess.run(
                    "nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu,utilization.gpu --format=csv,noheader",
                    shell=True, capture_output=True, text=True, timeout=10
                )
                if res.returncode == 0:
                    return {"success": True, "telemetry": res.stdout.strip()}
                return {"success": False, "error": "nvidia-smi failed or no GPU found"}

            elif name == "list_processes":
                pat = args.get("pattern", "").lower()
                cmd = f'tasklist /FI "IMAGENAME eq *{pat}*" /FO CSV' if os.name == "nt" else f"ps aux | grep {pat}"
                res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
                return {"success": True, "output": res.stdout[:2000]}

            elif name == "git_status":
                p = args.get("path", os.getcwd())
                res = subprocess.run("git status --short", shell=True, cwd=p, capture_output=True, text=True, timeout=10)
                return {"success": True, "status": res.stdout.strip()}

            elif name == "git_diff":
                p = args.get("path", os.getcwd())
                res = subprocess.run("git diff", shell=True, cwd=p, capture_output=True, text=True, timeout=10)
                return {"success": True, "diff": res.stdout[:4000]}

            elif name == "verify_condition":
                test_cmd = args.get("test_cmd", "")
                res = subprocess.run(test_cmd, shell=True, capture_output=True, text=True, timeout=tool.timeout_sec)
                return {
                    "success": res.returncode == 0,
                    "verified": res.returncode == 0,
                    "stdout": res.stdout,
                    "stderr": res.stderr,
                    "failure_class": FailureClass.VERIFICATION_FAILURE.value if res.returncode != 0 else None
                }

        except subprocess.TimeoutExpired:
            return {"success": False, "error": f"Tool '{name}' timed out", "failure_class": FailureClass.TIMEOUT.value}
        except Exception as e:
            return {"success": False, "error": str(e), "failure_class": FailureClass.UNKNOWN_FAILURE.value}


def parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    pattern = r'<function name="([^"]+)">([\s\S]*?)</function>'
    matches = re.findall(pattern, text)
    calls = []
    for name, body in matches:
        params = {}
        for p_match in re.finditer(r'<param name="([^"]+)">([\s\S]*?)</param>', body):
            p_name, p_val = p_match.group(1), p_match.group(2).strip()
            params[p_name] = p_val
        calls.append({"name": name, "params": params})
    return calls
