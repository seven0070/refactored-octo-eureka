"""
LAVOIR Single-Pass Decision Engine for Rabbit-2B
Based on: "LAVOIR: Teaching a Single-Pass Decision Encoder When and What to Ask with Amortized Value of Information" (arXiv:2609.30706)

Key mechanisms:
1. Calibrated Decision Distribution P(Action | x) over [ANSWER, TOOL, PLAN, CLARIFY, REJECT]
2. Gini-Impurity Uncertainty Cap: G(P) = 1 - sum(p_i^2)
3. Amortized Value of Information (VOI) for candidate missing slots
4. Policy: Act immediately when confident; ask only when VOI > threshold and capped by Gini impurity.
"""

import time
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from .core import DecisionType


@dataclass
class DecisionResult:
    decision: DecisionType
    confidence: float
    gini_impurity: float
    slot_voi: Dict[str, float]
    selected_slot: Optional[str] = None
    question_to_ask: Optional[str] = None
    latency_ms: float = 0.0


class LavoirDecisionEngine:
    def __init__(self, ask_threshold: float = 0.15, confidence_threshold: float = 0.70):
        self.ask_threshold = ask_threshold
        self.confidence_threshold = confidence_threshold

        # Candidate missing information slots and their diagnostic queries
        self.slot_templates = {
            "target_path": "What target file or directory path should be operated on?",
            "command_spec": "What specific command or arguments should be executed?",
            "destructive_confirm": "This operation modifies or deletes resources. Please confirm intent.",
            "test_target": "Which test file or function should be verified?",
        }

    def _extract_slot_indicators(self, text: str) -> Dict[str, float]:
        """Detect missing information slots and evaluate ambiguity signals."""
        t = text.lower().strip()
        indicators = {
            "target_path": 0.0,
            "command_spec": 0.0,
            "destructive_confirm": 0.0,
            "test_target": 0.0,
        }

        # Check for ambiguous target path
        if any(w in t for w in ("file", "directory", "project", "folder")) and not any(ext in t for ext in (".py", ".json", ".txt", ".md", "/", "\\")):
            indicators["target_path"] = 0.65

        # Check for vague execution commands
        if any(w in t for w in ("run", "execute", "launch", "start")) and not any(w in t for w in ("pytest", "python", "git", "nvidia", "dir", "ls")):
            indicators["command_spec"] = 0.60

        # Check for destructive operations
        if any(w in t for w in ("delete", "remove", "drop", "wipe", "clean -f", "reset --hard")):
            indicators["destructive_confirm"] = 0.90

        # Check for ambiguous testing targets
        if "test" in t and not any(w in t for w in ("all", "test_", ".py", "calc")):
            indicators["test_target"] = 0.50

        return indicators

    def evaluate(self, objective: str) -> DecisionResult:
        """
        Single-pass evaluation returning decision distribution, Gini impurity,
        and Amortized Value of Information (VOI) for missing slots.
        """
        t0 = time.perf_counter()
        t = objective.lower().strip()

        # 1. Base logit heuristics / single-pass scoring
        scores = {
            DecisionType.ANSWER: 0.1,
            DecisionType.TOOL: 0.1,
            DecisionType.PLAN: 0.1,
            DecisionType.CLARIFY: 0.05,
            DecisionType.REJECT: 0.05,
        }

        # Semantic routing features
        if any(t.startswith(w) for w in ("who", "what is", "why", "explain", "describe", "define", "how does", "tell me")):
            scores[DecisionType.ANSWER] += 2.5
        elif any(w in t for w in ("audit", "refactor", "build", "pipeline", "diagnose and repair", "full-lifecycle", "end-to-end")):
            scores[DecisionType.PLAN] += 2.8
        elif any(w in t for w in ("gpu", "nvidia", "read", "write", "list", "show", "cat", "git status", "git diff", "tasklist")):
            scores[DecisionType.TOOL] += 2.2
        else:
            # Default task orientation
            scores[DecisionType.TOOL] += 1.0
            scores[DecisionType.PLAN] += 0.8

        # 2. Softmax normalization -> calibrated probability distribution P(A | x)
        max_score = max(scores.values())
        exp_scores = {k: math.exp(v - max_score) for k, v in scores.items()}
        total_exp = sum(exp_scores.values())
        probs = {k: v / total_exp for k, v in exp_scores.items()}

        # 3. Gini-Impurity calculation: G(P) = 1 - sum(p_i^2)
        # Bounds maximum value that asking ANY question can possibly yield.
        gini_impurity = 1.0 - sum(p ** 2 for p in probs.values())

        # 4. Amortized Value of Information (VOI) for missing slots
        indicators = self._extract_slot_indicators(objective)
        slot_voi = {}
        for slot, prob_missing in indicators.items():
            # Raw expected gain is bounded by the Gini impurity of the current distribution
            # VOI = P(slot_needed) * Gini_cap
            slot_voi[slot] = prob_missing * gini_impurity

        best_slot = max(slot_voi, key=slot_voi.get) if slot_voi else None
        max_voi = slot_voi.get(best_slot, 0.0) if best_slot else 0.0

        # 5. Decision Policy (LAVOIR Theorem):
        # If Gini impurity is low (high confidence) OR max_voi is below threshold -> act immediately!
        # If max_voi exceeds threshold capped by Gini impurity -> CLARIFY with highest VOI slot.
        top_decision = max(probs, key=probs.get)
        top_confidence = probs[top_decision]

        latency = (time.perf_counter() - t0) * 1000.0

        # High destructive risk -> Reject or prompt authorization
        if indicators.get("destructive_confirm", 0.0) > 0.85:
            return DecisionResult(
                decision=DecisionType.CLARIFY,
                confidence=top_confidence,
                gini_impurity=gini_impurity,
                slot_voi=slot_voi,
                selected_slot="destructive_confirm",
                question_to_ask=self.slot_templates["destructive_confirm"],
                latency_ms=latency
            )

        if max_voi >= self.ask_threshold and top_confidence < self.confidence_threshold:
            return DecisionResult(
                decision=DecisionType.CLARIFY,
                confidence=top_confidence,
                gini_impurity=gini_impurity,
                slot_voi=slot_voi,
                selected_slot=best_slot,
                question_to_ask=self.slot_templates.get(best_slot),
                latency_ms=latency
            )

        return DecisionResult(
            decision=top_decision,
            confidence=top_confidence,
            gini_impurity=gini_impurity,
            slot_voi=slot_voi,
            latency_ms=latency
        )
