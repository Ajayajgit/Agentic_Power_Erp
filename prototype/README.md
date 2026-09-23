# NetSuite Project Plan Generation — Prototype

Small working slice of the design document: takes a sample discovery-call
transcript + a mock historical PM export, and produces a draft, historically
calibrated NetSuite implementation project plan as JSON.

**Everything below the "mocked" line is real code that actually runs — the
only things mocked are the PM tool connection (no live Asana account) and the
final human-approval step (no UI). Per the assignment's own allowance, this is
stated explicitly, not hidden.**

---

## 1. How this maps to the design document

| Design doc concept | Where it lives in this code |
|---|---|
| Deterministic vs. agentic steps (Section 1.2) | `workflow/plan_workflow.py` — steps 1, 2, 4, 5 are plain Python; step 3 (`_node_estimate_and_risk`) is the one LLM reasoning step, bounded to max 3 retries |
| LangGraph, not free-form ReAct | `workflow/plan_workflow.py` — a fixed `StateGraph` with a bounded retry loop between `validate` and `estimate_and_risk` |
| Chunking / embedding / retrieval (Section 1.3) | `ingestion/chunker.py`, `providers/embedding_provider.py`, `ingestion/retriever.py` |
| Pluggable PM connector (Section 1.4) | `connectors/pm_connector.py` — `PMConnector` interface + `AsanaMockConnector` implementation; swapping in Jira later means adding one new class, nothing else changes |
| Tenant isolation via namespaces (Section 2) | `vectorstore/pinecone_store.py` / `vectorstore/local_store.py` — every vector is upserted and queried under a `namespace=tenant_id`, so one tenant's query structurally cannot see another's data (verified by `dry_run_test.py`) |
| Historical velocity calibration | `ingestion/velocity.py` — computes actual-vs-estimated hour ratios per phase from the mock PM export, fed directly into the LLM prompt so the model calibrates against real history, not template guesses |
| Human review gate (Section 1.2, step 5) | `models.py` — every generated plan is stamped `status="pending_human_review"`; `main.py` prints a note where a real approval UI would sit before `pm_connector.push_plan(...)` is called |
| Fully autonomous write-back out of scope (Section 4.2) | `connectors/pm_connector.py` — `push_plan()` only ever prints a mock message, never calls a real API |

---

## 2. Tech stack used

- **LangGraph** — orchestrates the 5-step workflow with a real bounded retry loop
- **LLM**: OpenAI (primary, if `OPENAI_API_KEY` set) → **Groq** (fallback / effective primary if you only have a Groq key — serves open-source models like Llama 3.3)
- **Embeddings**: OpenAI (primary, if key set) → **sentence-transformers** (`all-MiniLM-L6-v2`, local, open-source, no key needed) as fallback
- **Vector store**: **Pinecone** (if `PINECONE_API_KEY` set) → local in-memory cosine-similarity store as fallback, so the prototype runs with zero cloud accounts if needed
- All of the above are chosen automatically at runtime based on what's in your `.env` — no code changes needed either way

---

## 3. Project structure

```
prototype/
├── main.py                    # entry point — run this
├── dry_run_test.py            # verification script using fake providers (no API keys needed)
├── config.py                  # loads all settings from .env
├── models.py                  # Task, PlanPhase, ProjectPlan, etc. (dataclasses)
├── requirements.txt
├── .env.example                # copy to .env and fill in your keys
├── providers/
│   ├── embedding_provider.py   # OpenAI + sentence-transformers, with automatic fallback
│   └── llm_provider.py         # OpenAI + Groq, with automatic fallback
├── vectorstore/
│   ├── base.py                 # VectorStore interface
│   ├── pinecone_store.py       # Pinecone implementation (namespace-per-tenant)
│   ├── local_store.py          # local in-memory fallback implementation
│   └── factory.py              # picks Pinecone or local automatically
├── connectors/
│   └── pm_connector.py         # PMConnector interface + AsanaMockConnector
├── ingestion/
│   ├── chunker.py               # TextChunker
│   ├── loaders.py               # file loaders (transcript / template / text doc)
│   ├── pipeline.py              # chunk -> embed -> upsert orchestration
│   ├── retriever.py             # tenant-scoped semantic search
│   └── velocity.py              # historical actual-vs-estimate calculation
├── workflow/
│   └── plan_workflow.py         # the LangGraph workflow itself
├── data/
│   ├── sample_transcript.txt        # realistic discovery-call transcript (new engagement)
│   ├── mock_pm_export.json          # realistic Asana-style export (a past, completed engagement)
│   ├── plan_template.json           # tenant's standard phase template (structure only, no numbers)
│   └── prior_engagement_summary.txt # a past retrospective doc (extra RAG context)
└── output/                    # generated draft_plan_<engagement_id>.json lands here
```

---

## 4. How to run it

### Step 1 — install dependencies
```bash
cd prototype
pip install -r requirements.txt
```

### Step 2 — configure your `.env`
```bash
cp .env.example .env
```
Then open `.env` and fill in at least:
```
GROQ_API_KEY=your-real-groq-key-here
```
That's genuinely enough to run the whole thing — you don't need OpenAI or
Pinecone keys. Leave `OPENAI_API_KEY` and `PINECONE_API_KEY` blank and the
system automatically uses Groq as the LLM and the local in-memory store for
vectors. The **first run** will download the `all-MiniLM-L6-v2` embedding
model from Hugging Face (a few hundred MB, one-time, needs normal internet
access) since that's the fallback embedding path when no `OPENAI_API_KEY`
is set.

### Step 3 — run it
```bash
python main.py
```

### (Optional) verify the pipeline logic without any API keys at all
```bash
python dry_run_test.py
```
This runs the full pipeline — chunking, ingestion, tenant-isolated retrieval,
velocity calculation, and the LangGraph workflow including its retry loop —
using fake local providers, so you can confirm everything is wired correctly
before spending any API credits. This already passed in testing (see below).

---

## 5. Expected output

Console output will look roughly like this (exact numbers vary by LLM run):

```
======================================================================
NetSuite Project Plan Generation - Prototype Run
======================================================================
... setup logs ...
Ingested 6 transcript chunks and 3 retrospective chunks.
Computed velocity stats for 8 phases from 24 historical tasks.
Running plan generation workflow (LLM provider primary=groq)...

======================================================================
DRAFT PLAN GENERATED - status: pending_human_review
  LLM provider used:        groq
  Embedding provider used:  sentence-transformers
  Vector store used:        local-inmemory
  Total estimated hours:    ~510
  Phases generated:         8
  Saved to:                 output/draft_plan_meridian_outdoor_supply_2026.json
======================================================================

[MEDIUM CONFIDENCE] Discovery - 62.0h over 12 days
   Risks: Multiple legacy systems (WMS, QuickBooks spreadsheets) historically extend process mapping time

[LOW CONFIDENCE] Integration - 210.0h over 24 days
   Risks: EDI certification with 3 retail partners historically the long pole, outside team control
...

Overall engagement risks:
  - 17-week timeline to November go-live is tight given historical EDI certification delays

--- Simulated human review gate ---
[MOCKED] A consultant would review/edit this plan here before anything is pushed downstream.
[MOCK PUSH] Would write plan for engagement 'meridian_outdoor_supply_2026' back to Asana. (No real API call made - connector is mocked.)
```

A file also lands at `output/draft_plan_meridian_outdoor_supply_2026.json` — a
full structured `ProjectPlan` (phases, deliverables, resource roles, effort,
risks, confidence per phase, overall assumptions) ready to be rendered into
the tenant's own template format or reviewed by a consultant.

### Already verified in this environment (no API keys needed)
`dry_run_test.py` was run during development and **all checks passed**:
ingestion produced real chunks from both documents, historical velocity
correctly showed the Integration phase running 1.47x over estimate (matching
the realistic mock data), tenant-isolated retrieval correctly returned zero
results when queried under a different tenant's namespace, and the retry
loop correctly recovered after two consecutive invalid LLM outputs before
succeeding on the third attempt.

---

## 6. What's real vs. mocked (honesty per the assignment's request)

**Real:**
- Chunking, embedding, and vector storage/retrieval logic
- The LangGraph workflow, including the bounded agentic reasoning step and its retry loop
- Historical velocity calculation from PM task data
- Automatic primary/fallback resilience for both the LLM and the embedding provider
- Tenant-namespace isolation at the vector store layer

**Mocked (explicitly, per the assignment's allowance):**
- The PM tool connection — `AsanaMockConnector` reads a local JSON file instead of calling Asana's real API. Swapping in a real API call only requires rewriting the inside of that one class.
- The human review/approval UI — represented as a printed message in `main.py` rather than an actual reviewer interface.
- The final write-back to the PM tool — `push_plan()` only prints what it would do.
