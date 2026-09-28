"""
Rabbit-2B Autonomous Agent Test Suite
Verifies: Planner, Task DAG, Scheduler, Policy, Verification, Recovery, Persistence & E2E
"""
import os
import sys
import unittest
import tempfile
import shutil

# Ensure workspace root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rabbit.core import AgentState, TaskStatus, RiskLevel, ResourceLimits, ResourceUsage
from rabbit.planner import TaskNode, TaskDAG, TaskScheduler, HierarchicalPlanner
from rabbit.tools import ToolRegistry, ToolDef, PolicyGate, parse_tool_calls
from rabbit.memory import PersistentMemory
from rabbit.engine import VerificationEngine, RecoveryEngine


class TestRabbitAutonomousSuite(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="rabbit_test_")
        self.db_path = os.path.join(self.temp_dir, "test_rabbit.db")
        self.memory = PersistentMemory(self.db_path)
        self.tools = ToolRegistry()
        self.verifier = VerificationEngine(self.tools)
        self.recovery = RecoveryEngine(self.tools, self.memory)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # 1. DAG & Scheduler Tests
    def test_dag_dependency_resolution(self):
        dag = TaskDAG()
        t1 = TaskNode("T1", "Setup project", priority=1)
        t2 = TaskNode("T2", "Build code", dependencies=["T1"], priority=2)
        t3 = TaskNode("T3", "Run tests", dependencies=["T2"], priority=3)
        dag.add_task(t1)
        dag.add_task(t2)
        dag.add_task(t3)

        scheduler = TaskScheduler(dag)
        # Initially only T1 should be ready
        ready = dag.get_ready_tasks()
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0].task_id, "T1")

        # After T1 success, T2 becomes ready
        dag.mark_success("T1", {"output": "ok"})
        ready = dag.get_ready_tasks()
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0].task_id, "T2")

        # If T2 fails, T3 must be blocked
        dag.mark_failed("T2", "Compilation error")
        ready = dag.get_ready_tasks()
        self.assertEqual(len(ready), 0)
        self.assertEqual(dag.get_task("T3").status, TaskStatus.BLOCKED)

    # 2. Tool Parsing & Policy Gate
    def test_xml_tool_parsing(self):
        xml_text = '<function name="execute_cmd"><param name="command">pytest -v</param></function>'
        calls = parse_tool_calls(xml_text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "execute_cmd")
        self.assertEqual(calls[0]["params"]["command"], "pytest -v")

    # 3. Verification Engine Tests
    def test_verification_evidence(self):
        test_file = os.path.join(self.temp_dir, "verified_output.txt")
        task = TaskNode("T_VERIFY", "Create file", postconditions=[f"file_exists: {test_file}"])

        # Fail before creation
        res_fail = self.verifier.verify_task(task, {"success": True})
        self.assertFalse(res_fail["passed"])

        # Pass after creation
        self.tools.execute("write_file", {"path": test_file, "content": "verified_content"})
        res_pass = self.verifier.verify_task(task, {"success": True})
        self.assertTrue(res_pass["passed"])

    # 4. Recovery & Rollback Tests
    def test_recovery_retry_and_rollback(self):
        test_file = os.path.join(self.temp_dir, "config.json")
        self.tools.execute("write_file", {"path": test_file, "content": "initial_safe_content"})

        task = TaskNode(
            "T_RECOVER", "Mutate config",
            retry_limit=2,
            postconditions=[f"file_exists: {test_file}"],
            rollback_strategy="file_rollback",
            verification_method='python -c "exit(1)"'  # forces failure
        )

        # First failure triggers retry
        v_res = self.verifier.verify_task(task, {"success": True})
        can_retry = self.recovery.handle_failure(task, {"success": True}, v_res, "run_test")
        self.assertTrue(can_retry)
        self.assertEqual(task.retry_count, 1)

        # Second failure exhausts retries
        v_res2 = self.verifier.verify_task(task, {"success": True})
        can_retry2 = self.recovery.handle_failure(task, {"success": True}, v_res2, "run_test")
        self.assertFalse(can_retry2)
        self.assertEqual(task.status, TaskStatus.FAILED)

    # 5. Persistent State & SQLite Memory
    def test_task_state_persistence(self):
        task_data = {
            "task_id": "T_SAVED",
            "description": "Persistent task check",
            "status": "SUCCESS",
            "priority": 2,
            "dependencies": ["T0"],
            "required_tools": ["execute_cmd"]
        }
        self.memory.save_task(task_data, "run_persisted_123")
        loaded = self.memory.load_run_tasks("run_persisted_123")
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["task_id"], "T_SAVED")
        self.assertEqual(loaded[0]["status"], "SUCCESS")

    # 6. Critical End-to-End Lifecycle Test
    def test_end_to_end_repair_and_verify(self):
        """
        Simulates:
        1. Write buggy function
        2. Run test (fails)
        3. Diagnose & repair bug
        4. Re-run test (passes)
        5. Verify final evidence
        """
        code_file = os.path.join(self.temp_dir, "calculator.py")
        test_file = os.path.join(self.temp_dir, "test_calc.py")

        # Step 1: Write buggy code (add returns a - b)
        self.tools.execute("write_file", {"path": code_file, "content": "def add(a, b):\n    return a - b\n"})
        self.tools.execute("write_file", {"path": test_file, "content": f"from calculator import add\nassert add(2, 3) == 5\nprint('TESTS_PASSED')\n"})

        # Step 2: Verification fails
        verify_cmd = f'"{sys.executable}" -B -c "import sys; sys.path.insert(0, r\'{self.temp_dir}\'); import test_calc"'
        obs = self.tools.execute("execute_cmd", {"command": verify_cmd})
        self.assertFalse(obs["success"])

        # Step 3: Repair code
        self.tools.execute("write_file", {"path": code_file, "content": "def add(a, b):\n    return a + b\n"})

        # Step 4: Verification succeeds
        obs_repaired = self.tools.execute("execute_cmd", {"command": verify_cmd})
        if not obs_repaired["success"]:
            print("OBS_REPAIRED STDERR:", obs_repaired.get("stderr"), "STDOUT:", obs_repaired.get("stdout"))
        self.assertTrue(obs_repaired["success"])
        self.assertIn("TESTS_PASSED", obs_repaired["stdout"])

    # 7. LAVOIR Single-Pass Decision & VOI Tests
    def test_lavoir_single_pass_decisions(self):
        from rabbit.decision import LavoirDecisionEngine, DecisionType
        engine = LavoirDecisionEngine()

        # Direct Answer (high confidence, low Gini)
        r_ans = engine.evaluate("What is the speed of light?")
        self.assertEqual(r_ans.decision, DecisionType.ANSWER)
        self.assertLess(r_ans.latency_ms, 50.0)

        # Destructive Confirmation (high VOI, routes to clarify)
        r_dest = engine.evaluate("Delete the build directory")
        self.assertEqual(r_dest.decision, DecisionType.CLARIFY)
        self.assertEqual(r_dest.selected_slot, "destructive_confirm")

        # Direct Tool Execution (show GPU)
        r_tool = engine.evaluate("Show GPU memory and status")
        self.assertEqual(r_tool.decision, DecisionType.TOOL)


if __name__ == "__main__":
    unittest.main()
