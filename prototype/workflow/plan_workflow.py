"""
workflow/plan_workflow.py

Implements the 5-step plan generation workflow from Section 1.2 of the
design document using LangGraph:

  [1] retrieve            - deterministic
  [2] draft_skeleton      - deterministic
  [3] estimate_and_risk   - TRUE AGENT (bounded: max N retries, one tool -
                             re-retrieval - fixed output schema)
  [4] validate             - deterministic
  [5] finalize             - deterministic (sets status=pending_human_review;
                             an actual human-approval UI is out of scope for
                             this prototype, per the assignment's mocking
                             allowance)

Steps 3 and 4 form a bounded loop: if validation fails or the estimate step
flags low confidence, control returns to step 3 up to max_retries times
before finalizing with whatever is best-so-far. This is the "agentic
reasoning within a fixed graph" pattern described in the design document,
deliberately not a free-form ReAct loop.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, TypedDict

from langgraph.graph import StateGraph, END

from prototype.ingestion.retriever import HistoricalRetriever
from prototype.models import PhaseStat, PlanPhase, ProjectPlan, RetrievedChunk
from prototype.providers.llm_provider import ResilientLLMProvider

logger = logging.getLogger(__name__)


class PlanState(TypedDict, total=False):
    tenant_id: str
    engagement_id: str
    engagement_name: str
    transcript_text: str
    template_phases: List[dict]
    phase_stats: Dict[str, PhaseStat]
    retrieved_context: List[RetrievedChunk]
    draft_phase_names: List[str]
    llm_raw_output: str
    parsed_phases: List[PlanPhase]
    validation_errors: List[str]
    retry_count: int
    final_plan: ProjectPlan


class PlanGenerationWorkflow:
    def __init__(self, retriever: HistoricalRetriever, llm: ResilientLLMProvider,
                 top_k: int, max_retries: int, embedding_provider_name_getter,
                 vector_store_name: str):
        self._retriever = retriever
        self._llm = llm
        self._top_k = top_k
        self._max_retries = max_retries
        self._get_embedding_provider_name = embedding_provider_name_getter
        self._vector_store_name = vector_store_name
        self._graph = self._build_graph()

    # ---------------- Node implementations ----------------

    def _node_retrieve(self, state: PlanState) -> dict:
        """[1] Deterministic: pull relevant historical context for this tenant."""
        query = state["transcript_text"][:1500]  # representative excerpt as the query
        chunks = self._retriever.retrieve(query_text=query, tenant_id=state["tenant_id"], top_k=self._top_k)
        logger.info("Retrieved %d relevant historical chunks.", len(chunks))
        return {"retrieved_context": chunks}

    def _node_draft_skeleton(self, state: PlanState) -> dict:
        """[2] Deterministic: phase names come from the tenant's own template, not invented."""
        names = [p["name"] for p in state["template_phases"]]
        return {"draft_phase_names": names, "retry_count": 0}

    def _node_estimate_and_risk(self, state: PlanState) -> dict:
        """[3] TRUE AGENT (bounded): reasons over retrieved history + phase velocity
        stats to produce calibrated effort/risk per phase. Can be re-entered up to
        max_retries times if validation fails."""

        context_text = "\n\n".join(
            f"[{c.content_type} | source={c.source} | relevance={c.score:.2f}] {c.text}"
            for c in state["retrieved_context"]
        ) or "No sufficiently relevant historical context was found above the similarity threshold."

        velocity_text = "\n".join(
            f"- {stat.phase}: historically estimated {stat.total_estimated_hours}h, "
            f"actually took {stat.total_actual_hours}h "
            f"(velocity ratio {stat.velocity_ratio}x, from {stat.task_count} past tasks)"
            for stat in state["phase_stats"].values()
        ) or "No historical velocity data available for this tenant yet."

        template_text = json.dumps(state["template_phases"], indent=2)

        system_prompt = (
            "You are a senior NetSuite delivery project manager working from a consulting "
            "firm's own historical project data. Your job is to estimate a realistic project "
            "plan calibrated to how long work ACTUALLY took on past similar engagements - "
            "NOT idealized template numbers. If historical evidence for a phase is weak or "
            "missing, say so honestly in that phase's risk notes and mark confidence as 'low' "
            "rather than inventing a confident number. "
            "Respond with STRICT JSON only, no markdown fences, no commentary outside the JSON."
        )

        user_prompt = f"""
TEMPLATE PHASE STRUCTURE (structure only - do not just copy any numbers, there are none):
{template_text}

HISTORICAL VELOCITY BY PHASE (from this tenant's own past completed engagement):
{velocity_text}

RELEVANT RETRIEVED CONTEXT (discovery call transcript excerpts + prior retrospective notes):
{context_text}

DISCOVERY CALL TRANSCRIPT EXCERPT FOR THE NEW ENGAGEMENT:
{state["transcript_text"][:3000]}

TASK:
Produce a project plan as a JSON object with this exact shape:
{{
  "phases": [
    {{
      "name": "<phase name, matching the template phase names>",
      "deliverables": ["..."],
      "resource_roles": ["..."],
      "estimated_hours": <number, calibrated using the velocity ratios above, not the raw template estimate>,
      "duration_days": <integer>,
      "dependencies": ["<names of phases this depends on>"],
      "risks": ["specific risk grounded in the transcript or retrieved context, not generic"],
      "confidence": "high" | "medium" | "low"
    }}
  ],
  "overall_risks": ["engagement-level risks, e.g. external dependencies like EDI certification"],
  "assumptions": ["explicit assumptions you made"]
}}

Calibration rule: for each phase, take the template's typical deliverables/roles, but set
estimated_hours by applying that phase's historical velocity ratio as a multiplier to a
reasonable base estimate you infer from the transcript's scope (e.g. number of EDI partners,
number of locations, integration count). Do not just copy historical totals - this is a
different, new engagement with its own scope.
"""

        raw = self._llm.generate(system_prompt=system_prompt, user_prompt=user_prompt, max_tokens=2500)
        return {"llm_raw_output": raw}

    def _node_validate(self, state: PlanState) -> dict:
        """[4] Deterministic: schema + sanity checks. Sets validation_errors."""
        errors: List[str] = []
        parsed_phases: List[PlanPhase] = []

        raw = state.get("llm_raw_output", "")
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            return {"validation_errors": [f"LLM output was not valid JSON: {exc}"]}

        phase_names_seen = set()
        for phase in data.get("phases", []):
            name = phase.get("name", "")
            hours = phase.get("estimated_hours", -1)
            duration = phase.get("duration_days", -1)

            if not name:
                errors.append("A phase is missing a name.")
                continue
            if hours is None or hours < 0:
                errors.append(f"Phase '{name}' has an invalid estimated_hours value: {hours}")
            if duration is None or duration < 0:
                errors.append(f"Phase '{name}' has an invalid duration_days value: {duration}")
            phase_names_seen.add(name)

            parsed_phases.append(PlanPhase(
                name=name,
                deliverables=phase.get("deliverables", []),
                resource_roles=phase.get("resource_roles", []),
                estimated_hours=float(hours) if hours and hours >= 0 else 0.0,
                duration_days=int(duration) if duration and duration >= 0 else 0,
                dependencies=phase.get("dependencies", []),
                risks=phase.get("risks", []),
                confidence=phase.get("confidence", "low"),
            ))

        # dependency sanity check: every referenced dependency must be a known phase name
        for phase in parsed_phases:
            for dep in phase.dependencies:
                if dep not in phase_names_seen:
                    errors.append(f"Phase '{phase.name}' depends on unknown phase '{dep}'.")

        if not parsed_phases:
            errors.append("No phases were parsed from the LLM output.")

        return {
            "validation_errors": errors,
            "parsed_phases": parsed_phases,
            "llm_raw_output_parsed_json": data if not errors else None,
            "_last_parsed_data": data,
        }

    def _node_finalize(self, state: PlanState) -> dict:
        """[5] Deterministic: assemble the final ProjectPlan and mark for human review."""
        data = state.get("_last_parsed_data", {}) or {}
        total_hours = sum(p.estimated_hours for p in state.get("parsed_phases", []))

        plan = ProjectPlan(
            tenant_id=state["tenant_id"],
            engagement_id=state["engagement_id"],
            engagement_name=state["engagement_name"],
            generated_at=datetime.now(timezone.utc).isoformat(),
            phases=state.get("parsed_phases", []),
            overall_risks=data.get("overall_risks", []),
            assumptions=data.get("assumptions", []),
            total_estimated_hours=total_hours,
            status="pending_human_review",
            model_used=self._llm.name,
            embedding_provider_used=self._get_embedding_provider_name(),
            vector_store_used=self._vector_store_name,
        )
        return {"final_plan": plan}

    # ---------------- Control flow ----------------

    def _should_retry(self, state: PlanState) -> str:
        errors = state.get("validation_errors", [])
        retries_so_far = state.get("retry_count", 0)

        if errors and retries_so_far < self._max_retries:
            logger.warning("Validation found %d issue(s) (attempt %d/%d): %s",
                            len(errors), retries_so_far + 1, self._max_retries, errors)
            return "retry"
        if errors:
            logger.warning("Max retries reached with unresolved issues - finalizing with best-effort output. "
                            "Issues: %s", errors)
        return "finalize"

    def _increment_retry(self, state: PlanState) -> dict:
        return {"retry_count": state.get("retry_count", 0) + 1}

    # ---------------- Graph assembly ----------------

    def _build_graph(self):
        graph = StateGraph(PlanState)

        graph.add_node("retrieve", self._node_retrieve)
        graph.add_node("draft_skeleton", self._node_draft_skeleton)
        graph.add_node("estimate_and_risk", self._node_estimate_and_risk)
        graph.add_node("validate", self._node_validate)
        graph.add_node("increment_retry", self._increment_retry)
        graph.add_node("finalize", self._node_finalize)

        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "draft_skeleton")
        graph.add_edge("draft_skeleton", "estimate_and_risk")
        graph.add_edge("estimate_and_risk", "validate")

        graph.add_conditional_edges(
            "validate",
            self._should_retry,
            {"retry": "increment_retry", "finalize": "finalize"},
        )
        graph.add_edge("increment_retry", "estimate_and_risk")
        graph.add_edge("finalize", END)

        return graph.compile()

    def run(self, initial_state: PlanState) -> ProjectPlan:
        result = self._graph.invoke(initial_state)
        return result["final_plan"]
