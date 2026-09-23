"""
connectors/pm_connector.py

Pluggable PM connector interface (mirrors Section 1.4 of the design
document: "not Asana-only"). All plan-generation logic depends on this
interface and on the normalized Task model - never on Asana's specific
field names - so adding Jira/Monday/Linear later means writing one new
adapter class, nothing else changes.

AsanaMockConnector reads a local JSON file shaped like a simplified Asana
export, standing in for a real Asana API call (mocked per the assignment's
allowance to mock integrations for the prototype).
"""

import json
from abc import ABC, abstractmethod
from typing import List

from prototype.models import Task


class PMConnector(ABC):
    name: str = "base"

    @abstractmethod
    def fetch_historical_tasks(self) -> List[Task]:
        """Return normalized Task objects from completed past engagement(s)."""
        raise NotImplementedError

    @abstractmethod
    def push_plan(self, plan_summary: dict) -> bool:
        """Write an approved plan back into the PM tool. Returns success flag."""
        raise NotImplementedError


class AsanaMockConnector(PMConnector):
    """
    Mock Asana connector. In production this would call the real Asana API;
    here it reads a local JSON export so the prototype can run without any
    live PM tool credentials, per the assignment's mocking allowance.
    """

    name = "asana"

    def __init__(self, export_file_path: str):
        self._export_file_path = export_file_path

    def fetch_historical_tasks(self) -> List[Task]:
        with open(self._export_file_path, "r") as f:
            data = json.load(f)

        tasks = []
        for raw in data.get("tasks", []):
            tasks.append(Task(
                id=raw["gid"],
                title=raw["name"],
                phase=raw["phase"],
                assignee_role=raw["assignee_role"],
                estimated_hours=float(raw["estimated_hours"]),
                actual_hours=float(raw["actual_hours"]) if raw.get("actual_hours") is not None else None,
                status=raw["status"],
                start_on=raw.get("start_on"),
                due_on=raw.get("due_on"),
                dependencies=raw.get("dependencies", []),
            ))
        return tasks

    def push_plan(self, plan_summary: dict) -> bool:
        # Mocked write-back. Per the design document (Section 4.2), fully
        # autonomous write-back is explicitly out of scope for v1 - this
        # only runs after a simulated human-approval flag is set, and it
        # just prints/logs rather than calling a real API.
        print(f"[MOCK PUSH] Would write plan for engagement "
              f"'{plan_summary.get('engagement_id')}' back to Asana. "
              f"(No real API call made - connector is mocked.)")
        return True
