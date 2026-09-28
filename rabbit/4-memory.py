"""
Rabbit-2B Layered Memory & Persistent Storage (SQLite)
Working Memory -> Task Memory -> Episodic Memory -> Long-Term Memory
"""
import os
import json
import time
import sqlite3
from typing import Dict, Any, List, Optional
from .core import TaskStatus


class PersistentMemory:
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = os.environ.get(
                "RABBIT_DB_PATH",
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "rabbit_state.db")
            )
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_tables()

    def _init_tables(self):
        with self.conn:
            # 1. Audit Trail
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    run_id TEXT,
                    task_id TEXT,
                    event_type TEXT,
                    status TEXT,
                    duration REAL,
                    payload TEXT
                )
            """)
            # 2. Task Graph State
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS persistent_tasks (
                    task_id TEXT PRIMARY KEY,
                    run_id TEXT,
                    parent_id TEXT,
                    description TEXT,
                    status TEXT,
                    priority INTEGER,
                    dependencies TEXT,
                    required_tools TEXT,
                    preconditions TEXT,
                    postconditions TEXT,
                    verification_method TEXT,
                    retry_limit INTEGER,
                    retry_count INTEGER,
                    rollback_strategy TEXT,
                    result TEXT,
                    updated_at REAL
                )
            """)
            # 3. Episodic Runs
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS episodic_runs (
                    run_id TEXT PRIMARY KEY,
                    objective TEXT,
                    status TEXT,
                    tasks_total INTEGER,
                    tasks_succeeded INTEGER,
                    duration REAL,
                    summary TEXT,
                    created_at REAL
                )
            """)
            # 4. Long-Term Key-Value Store
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS long_term_store (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at REAL
                )
            """)

    def log_event(self, run_id: str, task_id: str, event_type: str, status: str = "", duration: float = 0.0, payload: Any = None):
        with self.conn:
            self.conn.execute(
                "INSERT INTO audit_events (timestamp, run_id, task_id, event_type, status, duration, payload) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (time.time(), run_id, task_id, event_type, status, duration, json.dumps(payload or {}, default=str))
            )

    def save_task(self, task: Dict[str, Any], run_id: str):
        with self.conn:
            self.conn.execute("""
                INSERT OR REPLACE INTO persistent_tasks (
                    task_id, run_id, parent_id, description, status, priority,
                    dependencies, required_tools, preconditions, postconditions,
                    verification_method, retry_limit, retry_count, rollback_strategy,
                    result, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task["task_id"], run_id, task.get("parent_id", ""),
                task["description"], task["status"], task.get("priority", 1),
                json.dumps(task.get("dependencies", [])),
                json.dumps(task.get("required_tools", [])),
                json.dumps(task.get("preconditions", [])),
                json.dumps(task.get("postconditions", [])),
                task.get("verification_method", ""),
                task.get("retry_limit", 3),
                task.get("retry_count", 0),
                task.get("rollback_strategy", ""),
                json.dumps(task.get("result", {})),
                time.time()
            ))

    def load_run_tasks(self, run_id: str) -> List[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM persistent_tasks WHERE run_id = ?", (run_id,))
        rows = cur.fetchall()
        tasks = []
        for r in rows:
            tasks.append({
                "task_id": r[0], "run_id": r[1], "parent_id": r[2], "description": r[3],
                "status": r[4], "priority": r[5], "dependencies": json.loads(r[6] or "[]"),
                "required_tools": json.loads(r[7] or "[]"), "preconditions": json.loads(r[8] or "[]"),
                "postconditions": json.loads(r[9] or "[]"), "verification_method": r[10],
                "retry_limit": r[11], "retry_count": r[12], "rollback_strategy": r[13],
                "result": json.loads(r[14] or "{}"), "updated_at": r[15]
            })
        return tasks

    def record_episode(self, run_id: str, objective: str, status: str, total: int, succeeded: int, duration: float, summary: str):
        with self.conn:
            self.conn.execute("""
                INSERT OR REPLACE INTO episodic_runs (run_id, objective, status, tasks_total, tasks_succeeded, duration, summary, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (run_id, objective, status, total, succeeded, duration, summary, time.time()))

    def get_ltm(self, key: str, default: Any = None) -> Any:
        cur = self.conn.cursor()
        cur.execute("SELECT value FROM long_term_store WHERE key = ?", (key,))
        row = cur.fetchone()
        return json.loads(row[0]) if row else default

    def set_ltm(self, key: str, value: Any):
        with self.conn:
            self.conn.execute("INSERT OR REPLACE INTO long_term_store (key, value, updated_at) VALUES (?, ?, ?)",
                              (key, json.dumps(value, default=str), time.time()))
