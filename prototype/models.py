"""
models.py

Plain data structures shared across the pipeline. Kept separate from any
provider/connector logic (Single Responsibility) so any component can import
these without pulling in unrelated dependencies.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Task:
    """A normalized PM ticket, regardless of which source tool it came from."""
    id: str
    title: str
    phase: str
    assignee_role: str
    estimated_hours: float
    actual_hours: Optional[float]
    status: str
    start_on: Optional[str] = None
    due_on: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)

    @property
    def variance_ratio(self) -> Optional[float]:
        """actual / estimated. >1 means it ran over. None if not completed yet."""
        if self.actual_hours is None or self.estimated_hours == 0:
            return None
        return round(self.actual_hours / self.estimated_hours, 3)


@dataclass
class RetrievedChunk:
    """A single piece of context pulled back from the vector store."""
    text: str
    source: str
    content_type: str
    score: float


@dataclass
class PhaseStat:
    """Historical velocity summary for one phase, computed from past Tasks."""
    phase: str
    total_estimated_hours: float
    total_actual_hours: float
    task_count: int

    @property
    def velocity_ratio(self) -> float:
        """>1 means this phase historically runs over estimate."""
        if self.total_estimated_hours == 0:
            return 1.0
        return round(self.total_actual_hours / self.total_estimated_hours, 3)


@dataclass
class PlanPhase:
    """One phase of the generated project plan."""
    name: str
    deliverables: List[str]
    resource_roles: List[str]
    estimated_hours: float
    duration_days: int
    dependencies: List[str]
    risks: List[str]
    confidence: str  # "high" | "medium" | "low" - based on strength of historical evidence


@dataclass
class ProjectPlan:
    """The final structured output of the workflow."""
    tenant_id: str
    engagement_id: str
    engagement_name: str
    generated_at: str
    phases: List[PlanPhase]
    overall_risks: List[str]
    assumptions: List[str]
    total_estimated_hours: float
    status: str  # e.g. "pending_human_review"
    model_used: str
    embedding_provider_used: str
    vector_store_used: str
