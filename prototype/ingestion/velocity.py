"""
ingestion/velocity.py

Computes per-phase historical velocity (actual vs. estimated hours) from
normalized Task objects. This is the "calibrated against the tenant's own
historical delivery velocity" mechanism the assignment specifically asks
for - a deterministic calculation, not left to LLM guessing.
"""

from collections import defaultdict
from typing import Dict, List

from prototype.models import PhaseStat, Task


class VelocityCalculator:
    @staticmethod
    def compute_phase_stats(tasks: List[Task]) -> Dict[str, PhaseStat]:
        grouped = defaultdict(list)
        for task in tasks:
            if task.actual_hours is not None:
                grouped[task.phase].append(task)

        stats = {}
        for phase, phase_tasks in grouped.items():
            total_est = sum(t.estimated_hours for t in phase_tasks)
            total_act = sum(t.actual_hours for t in phase_tasks)
            stats[phase] = PhaseStat(
                phase=phase,
                total_estimated_hours=total_est,
                total_actual_hours=total_act,
                task_count=len(phase_tasks),
            )
        return stats
