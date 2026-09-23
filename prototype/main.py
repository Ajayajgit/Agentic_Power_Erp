"""
main.py

Entry point. Wires together every component built so far and runs the
full pipeline end to end:

  1. Load config from .env
  2. Build resilient embedding provider (OpenAI primary / sentence-transformers fallback)
  3. Build resilient LLM provider (OpenAI primary / Groq fallback)
  4. Build vector store (Pinecone if configured / local in-memory fallback)
  5. Ingest: chunk + embed + store the new engagement's transcript,
     the tenant's prior engagement retrospective, and the template
  6. Pull historical PM tasks (mocked Asana export) and compute phase velocity
  7. Run the LangGraph plan-generation workflow
  8. Print + save the resulting draft plan as JSON
  9. Simulate the (mocked, gated) push-back to the PM tool

Run with:  python main.py
"""

import json
import logging
import os
from dataclasses import asdict
from datetime import datetime

from prototype.config import settings
from prototype.connectors.pm_connector import AsanaMockConnector
from prototype.ingestion.chunker import TextChunker
from prototype.ingestion.loaders import PlainTextLoader, TemplateLoader, TranscriptLoader
from prototype.ingestion.pipeline import IngestionPipeline
from prototype.ingestion.retriever import HistoricalRetriever
from prototype.ingestion.velocity import VelocityCalculator
from prototype.providers.embedding_provider import EmbeddingProviderFactory
from prototype.providers.llm_provider import LLMProviderFactory
from prototype.vectorstore.factory import VectorStoreFactory
from prototype.workflow.plan_workflow import PlanGenerationWorkflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def main():
    print("=" * 70)
    print("NetSuite Project Plan Generation - Prototype Run")
    print("=" * 70)

    # ---- 1. Config already loaded via config.py's load_dotenv() ----
    logger.info("Tenant: %s | Engagement: %s", settings.tenant_id, settings.engagement_id)

    # ---- 2 & 3. Providers ----
    embedder = EmbeddingProviderFactory.create(settings)
    llm = LLMProviderFactory.create(settings)

    # figure out embedding dimension by doing one throwaway embed call
    probe_vector = embedder.embed(["dimension probe"])[0]
    embedding_dim = len(probe_vector)
    logger.info("Embedding provider ready (dimension=%d).", embedding_dim)

    # ---- 4. Vector store ----
    vector_store = VectorStoreFactory.create(settings, embedding_dimension=embedding_dim)
    logger.info("Vector store backend: %s", vector_store.name)

    # ---- 5. Ingestion ----
    chunker = TextChunker(chunk_size_words=180, overlap_words=30)
    ingestion = IngestionPipeline(chunker=chunker, embedder=embedder, vector_store=vector_store)

    transcript_text = TranscriptLoader.load(os.path.join(DATA_DIR, "sample_transcript.txt"))
    retro_text = PlainTextLoader.load(os.path.join(DATA_DIR, "prior_engagement_summary.txt"))
    template = TemplateLoader.load(os.path.join(DATA_DIR, "plan_template.json"))

    n1 = ingestion.ingest_document(
        text=transcript_text, tenant_id=settings.tenant_id, engagement_id=settings.engagement_id,
        content_type="transcript", source="meridian_discovery_call_2",
    )
    n2 = ingestion.ingest_document(
        text=retro_text, tenant_id=settings.tenant_id, engagement_id="northwind_2025",
        content_type="retrospective", source="northwind_retrospective",
    )
    logger.info("Ingested %d transcript chunks and %d retrospective chunks.", n1, n2)

    # ---- 6. Historical PM data + velocity ----
    pm_connector = AsanaMockConnector(export_file_path=os.path.join(DATA_DIR, "mock_pm_export.json"))
    historical_tasks = pm_connector.fetch_historical_tasks()
    phase_stats = VelocityCalculator.compute_phase_stats(historical_tasks)
    logger.info("Computed velocity stats for %d phases from %d historical tasks.",
                len(phase_stats), len(historical_tasks))

    # ---- 7. Run the workflow ----
    retriever = HistoricalRetriever(embedder=embedder, vector_store=vector_store)
    workflow = PlanGenerationWorkflow(
        retriever=retriever,
        llm=llm,
        top_k=settings.top_k_retrieval,
        max_retries=settings.max_estimate_retries,
        embedding_provider_name_getter=lambda: embedder.name,
        vector_store_name=vector_store.name,
    )

    initial_state = {
        "tenant_id": settings.tenant_id,
        "engagement_id": settings.engagement_id,
        "engagement_name": "Meridian Outdoor Supply Co. - NetSuite Implementation",
        "transcript_text": transcript_text,
        "template_phases": template["phases"],
    }
    # phase_stats gets attached directly since it's precomputed, not part of the state dict above
    initial_state["phase_stats"] = phase_stats

    logger.info("Running plan generation workflow (LLM provider primary=%s)...", llm.name if llm.name != "unresolved" else "not yet called")
    plan = workflow.run(initial_state)

    # ---- 8. Output ----
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"draft_plan_{settings.engagement_id}.json")
    with open(output_path, "w") as f:
        json.dump(asdict(plan), f, indent=2, default=str)

    print("\n" + "=" * 70)
    print(f"DRAFT PLAN GENERATED - status: {plan.status}")
    print(f"  LLM provider used:        {plan.model_used}")
    print(f"  Embedding provider used:  {plan.embedding_provider_used}")
    print(f"  Vector store used:        {plan.vector_store_used}")
    print(f"  Total estimated hours:    {plan.total_estimated_hours}")
    print(f"  Phases generated:         {len(plan.phases)}")
    print(f"  Saved to:                 {output_path}")
    print("=" * 70)

    for phase in plan.phases:
        print(f"\n[{phase.confidence.upper()} CONFIDENCE] {phase.name} - "
              f"{phase.estimated_hours}h over {phase.duration_days} days")
        if phase.risks:
            print(f"   Risks: {'; '.join(phase.risks)}")

    if plan.overall_risks:
        print("\nOverall engagement risks:")
        for r in plan.overall_risks:
            print(f"  - {r}")

    # ---- 9. Simulated (mocked, gated) write-back ----
    print("\n--- Simulated human review gate ---")
    print("[MOCKED] A consultant would review/edit this plan here before anything is pushed downstream.")
    pm_connector.push_plan({"engagement_id": settings.engagement_id, "status": "approved_for_demo"})


if __name__ == "__main__":
    main()


def generate_plan_from_files(transcript_path: str, retro_path: str, template_path: str,
                             pm_export_path: str, tenant_id: str, engagement_id: str,
                             output_dir: str | None = None) -> dict:
    """Run the full pipeline using explicit file paths and return the generated plan as a dict.

    Keeps the same behavior as the CLI but accepts uploaded file paths from an API.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    # ---- Providers ----
    embedder = EmbeddingProviderFactory.create(settings)
    llm = LLMProviderFactory.create(settings)

    # figure out embedding dimension by doing one throwaway embed call
    probe_vector = embedder.embed(["dimension probe"])[0]
    embedding_dim = len(probe_vector)

    # ---- Vector store ----
    vector_store = VectorStoreFactory.create(settings, embedding_dimension=embedding_dim)

    # ---- Ingestion ----
    chunker = TextChunker(chunk_size_words=180, overlap_words=30)
    ingestion = IngestionPipeline(chunker=chunker, embedder=embedder, vector_store=vector_store)

    transcript_text = TranscriptLoader.load(transcript_path)
    retro_text = PlainTextLoader.load(retro_path)
    template = TemplateLoader.load(template_path)

    ingestion.ingest_document(
        text=transcript_text, tenant_id=tenant_id, engagement_id=engagement_id,
        content_type="transcript", source="uploaded_transcript",
    )
    ingestion.ingest_document(
        text=retro_text, tenant_id=tenant_id, engagement_id="historical_retrospective",
        content_type="retrospective", source="uploaded_retro",
    )

    # ---- Historical PM data + velocity ----
    pm_connector = AsanaMockConnector(export_file_path=pm_export_path)
    historical_tasks = pm_connector.fetch_historical_tasks()
    phase_stats = VelocityCalculator.compute_phase_stats(historical_tasks)

    # ---- Run the workflow ----
    retriever = HistoricalRetriever(embedder=embedder, vector_store=vector_store)
    workflow = PlanGenerationWorkflow(
        retriever=retriever,
        llm=llm,
        top_k=settings.top_k_retrieval,
        max_retries=settings.max_estimate_retries,
        embedding_provider_name_getter=lambda: embedder.name,
        vector_store_name=vector_store.name,
    )

    initial_state = {
        "tenant_id": tenant_id,
        "engagement_id": engagement_id,
        "engagement_name": "Uploaded Engagement",
        "transcript_text": transcript_text,
        "template_phases": template["phases"],
    }
    initial_state["phase_stats"] = phase_stats

    plan = workflow.run(initial_state)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"draft_plan_{engagement_id}.json")
    with open(output_path, "w") as f:
        json.dump(asdict(plan), f, indent=2, default=str)

    return asdict(plan)
