"""Trace logging for Stage 2's agentic retrieval decisions.

Every step is Reason (why this fired) / Act (what was done) / Observe
(what came back) — so the Trace view can show not just the final answer
but what the system actually did to get there.
"""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TraceStep:
    label: str
    act: str
    reason: str = ""
    observe: str = ""


@dataclass
class Trace:
    question: str
    steps: list[TraceStep] = field(default_factory=list)
    answer: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def log(self, label: str, act: str, reason: str = "", observe: str = "") -> None:
        self.steps.append(TraceStep(label=label, act=act, reason=reason, observe=observe))
