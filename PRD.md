# PRD — Local-First Agentic RAG Pipeline with Evaluation

**Source:** `to_read.pdf` (NTU Singapore workshop deck, "Three stages. Three working systems.")
**Status:** Draft v1
**Owner:** Kentz Wong

## 1. Summary

Build a RAG (Retrieval-Augmented Generation) system in three incremental stages, each extending the previous one within the same project:

1. **Stage 1 — Classic RAG.** A fixed ingestion + query pipeline (documents → chunks → embeddings → vector store → LLM answer). Local-first, no API keys required.
2. **Stage 2 — Agentic enhancements.** Add conditional retrieval behavior — query rewriting when retrieval is weak, reranking to cut noise, and a trace/observability view so decisions are visible.
3. **Stage 3 — Measurement.** Add a golden-set evaluation harness using Ragas (LLM-as-judge) to score the Stage 2 system on faithfulness, answer relevance, context precision, and context recall — so future changes are measurable, not guesswork.

Each stage should ship as a working system with a UI, not just a script.

## 2. Goals

- End-to-end local RAG pipeline that ingests arbitrary documents and answers questions grounded in them.
- Adaptive retrieval that self-corrects on weak matches (query rewriting) and improves precision (reranking).
- Full observability into agentic decisions (what was tried, what fired, what didn't).
- A repeatable, automated way to score system quality (Ragas over a golden question set) so regressions and improvements are quantifiable.
- Minimal external dependencies / cost: local LLM + embeddings by default; paid APIs only where explicitly justified (reranking, judge model) and only via generous free tiers.

## 3. Non-goals

- Multi-user auth, deployment/hosting, horizontal scaling, or production security hardening.
- Support for arbitrary vector DBs beyond Chroma (interface should stay swappable, but only Chroma is implemented).
- Conversation memory / multi-turn chat (Stage 1 and 2 are single-turn: "each question stands alone").
- Fine-tuning or training any model.
- Building a custom UI framework — pick one (Streamlit recommended) and don't over-invest in polish.

## 4. Tech Stack

| Layer | Choice | Notes |
|---|---|---|
| LLM | Ollama running `llama3.2`, local | Zero per-token cost. `temperature=0` for grounded answers. |
| Embeddings | Ollama `nomic-embed-text` | 768-dim, local, deterministic (same text → same vector). First pull ~270MB. |
| Vector store | Chroma, embedded mode | SQLite file on disk in project folder, no server. |
| Orchestration | LangChain | Document Loaders, Text Splitters, Embeddings, VectorStores, Retrievers, Prompts, ChatModel, Output Parsers. |
| Reranker (Stage 2+) | Cohere Rerank (`cohere-rerank-v3`), free tier | ~1000 calls/month free. Cross-encoder, more accurate than bi-encoder similarity. |
| Judge model (Stage 3) | Gemini Flash Lite, free tier | Used only by Ragas as LLM-as-judge. |
| Eval framework (Stage 3) | Ragas (open source) | Implements faithfulness, answer relevance, context precision, context recall out of the box. |
| UI | Streamlit (recommended; Gradio/React/plain HTML acceptable) | Framework choice not graded — pick fastest to iterate. |

**Operational prerequisite:** `ollama serve` must be running in a separate terminal before the app starts. Don't restart mid-pull of the embedding model.

## 5. Architecture Overview

```
Stage 1 (chain):
  Indexing:  Upload doc → Chunk → Embed (Ollama) → Store in Chroma
  Query:     Question → Embed → Top-3 chunks (Chroma) → Prompt + chunks → Ollama → Answer

Stage 2 (loop, extends Stage 1's query path):
  1. Question → Embed → Search Chroma (top-20)
       └ top score < 0.5 ?  yes → step 2   no → step 3
  2. Rewrite query (LLM) → Embed → Search → recheck score  (max 2 rewrites, bounded)
  3. Cohere Rerank (20 → 3)
  4. Prompt + top-3 chunks → Ollama → Answer
  Throughout: log every decision → render as a trace timeline in the UI.

Stage 3 (evaluation harness, wraps Stage 2 as a black box):
  For each question in a 10-question golden set:
    Run through the Stage 2 agent → capture (question, retrieved chunks, answer)
    Score with Ragas (4 metrics)
  Aggregate → scorecard (per-metric averages + per-question breakdown)
```

**Key design principle — the Retriever is the seam.** The `Retriever` abstraction (`vectorstore.as_retriever()`) is what changes between stages; the downstream chain (`prompt | llm`) stays identical. Stage 2 wraps the Stage 1 retriever in `ContextualCompressionRetriever` (rerank) or `MultiQueryRetriever` (query expansion) without touching anything downstream. Keep this interface clean.

## 6. Configuration & Model Transparency

All tunable pipeline parameters must be exposed as editable controls in the UI (a **Settings view**), not buried in a config file only. A config file/env still supplies the initial defaults, but the UI is the source of truth at runtime — changes apply on the next ingest/query without restarting the app. The active LLM and embedding model must always be visible read-only in the UI, so it's never ambiguous what produced a given answer.

### 6.1 Configurable Parameters

| Parameter | Introduced | Default | UI control | Effect |
|---|---|---|---|---|
| `chunk_size` | Stage 1 | 1000 chars | number input / slider | Size of each chunk fed to the splitter. Takes effect on next ingestion. |
| `chunk_overlap` | Stage 1 | 200 chars | number input / slider | Shared text between adjacent chunks. Takes effect on next ingestion. |
| `search_k` | Stage 1 | 3 | number input / slider | Number of chunks retrieved for the final prompt. Takes effect on next query. |
| `score_threshold` | Stage 2 | 0.5 | slider (0–1) | Similarity score below which a query rewrite fires. Takes effect on next query. |
| `max_rewrites` | Stage 2 | 2 | number input | Upper bound on the rewrite retry loop. Takes effect on next query. |
| `rerank_top_n` | Stage 2 | 3 | number input | Number of chunks kept after Cohere reranking. Takes effect on next query. |

Existing embeddings are not retroactively re-chunked when `chunk_size`/`chunk_overlap` change — those only affect documents ingested after the change.

### 6.2 Model Transparency

- **R0.1** The UI must display, at all times (e.g. a header or sidebar visible on every view), the active LLM model (e.g. `llama3.2` via Ollama) and the active embedding model (e.g. `nomic-embed-text` via Ollama).
- **R0.2** Model identity is read-only in the UI (switching models is a config/env change, not a UI action) — the goal is transparency about what's answering, not a model picker.
- **R0.3** When Stage 2/3 introduce additional models (Cohere reranker, Ragas judge model), display those alongside the LLM/embedding model wherever their feature is active — e.g. show the reranker model on the Ask/Trace view once Stage 2 ships, and the judge model on the Evaluate view once Stage 3 ships.

### 6.3 Acceptance Criteria

- Changing `chunk_size` or `chunk_overlap` in Settings and re-ingesting a document produces a different chunk count than before the change.
- Changing `search_k` changes the number of source chunks shown alongside an answer.
- Changing `score_threshold` changes whether a given borderline query triggers a rewrite (verifiable in the Trace view).
- Changing `max_rewrites` caps the number of rewrite cycles visible in the trace.
- Changing `rerank_top_n` changes the number of chunks that reach the final prompt.
- The active LLM and embedding model names are visible on every view, not just on Ingest.
- No parameter requires a code change or app restart to take effect.

## 7. Stage 1 — Classic RAG (Target: ~75 min build)

### 7.1 Requirements

- **R1.1 Document ingestion:** Accept plain text, markdown, and PDF files. Use a `DocumentLoader` per format (`TextLoader`, `PyPDFLoader`, etc.), producing `Document` objects with text + metadata (source path, page number).
- **R1.2 Chunking:** Use `RecursiveCharacterTextSplitter` as the default splitter. `chunk_size` and `chunk_overlap` are user-configurable via the Settings view (see §6.1), not hardcoded.
- **R1.3 Embedding + storage:** Embed each chunk via Ollama (`nomic-embed-text`) and store in a Chroma collection persisted to disk in the project folder. Re-running ingestion on an unchanged doc should not require re-embedding everything (nice-to-have, not blocking).
- **R1.4 Query pipeline:** Given a question, embed it, retrieve top-`search_k` chunks via `similarity_search` (`search_k` configurable via Settings, default 3), stuff them into a prompt template (system message sets "answer using only the provided context, say so if it doesn't cover the question"; human message carries context + question), call `ChatOllama(model="llama3.2", temperature=0)`, parse output with `StrOutputParser`.
- **R1.5 No agentic behavior:** This is a fixed chain — no tools, no loop, no memory. Same five steps for every question, always.
- **R1.6 UI — three views:**
  - **Ingest view:** upload a document, trigger chunk→embed→store, show resulting chunk count.
  - **Ask view:** enter a question, show the answer and the retrieved source chunks (for basic groundedness sanity-checking).
  - **Settings view:** editable controls for `chunk_size`, `chunk_overlap`, `search_k` (see §6.1 for defaults); read-only display of the active LLM model and embedding model (see §6.2). This view is visible from Stage 1 onward and gains more controls in Stage 2.

### 7.2 Tasks

1. Scaffold project (LangChain + Chroma dependencies, Ollama client, folder structure separating ingestion and query code).
2. Assemble or generate a sample document set (e.g., HR handbook, a few policy docs, an FAQ) to develop and test against.
3. Implement the ingestion pipeline (Loader → Splitter → Embeddings → Chroma) reading `chunk_size`/`chunk_overlap` from the runtime config.
4. Implement the query pipeline (Embed → Retriever(k=`search_k`) → Prompt → ChatOllama → StrOutputParser).
5. Build the UI with Ingest, Ask, and Settings views; wire the Settings controls to the runtime config and surface the active model names.
6. Smoke-test: ingest sample docs, ask a handful of questions, confirm answers are grounded in retrieved chunks and confirm the "no answer in context" path degrades gracefully. Confirm changing `chunk_size`/`chunk_overlap`/`search_k` in Settings visibly changes behavior.

### 7.3 Acceptance Criteria

- A document can be uploaded and is queryable within the same session.
- Asking a question grounded in the docs returns a correct, cited-by-context answer.
- Asking a question with no relevant content in the docs causes the model to say so, not hallucinate.
- No API keys are required to run Stage 1 end-to-end.
- `chunk_size`, `chunk_overlap`, and `search_k` are changeable from the Settings view without editing code.
- The active LLM model and embedding model names are visible in the UI.

## 8. Stage 2 — Agentic Enhancements (Target: ~55 min build)

### 8.1 Motivation (problems being fixed)

- **Vocabulary mismatch:** user phrasing doesn't match document phrasing; embedding similarity misses the synonym.
- **Noisy top-K:** irrelevant-but-high-similarity chunks confuse the LLM.
- **Hidden failures:** a plausible-looking answer with no visibility into what the system actually did to produce it.

### 8.2 Requirements

- **R2.1 Score-gated query rewriting:** After the initial top-20 retrieval, check the top similarity score. If `< score_threshold`, have the LLM rewrite the query (more specific/expanded phrasing) and retrieve again. Bound retries to `max_rewrites` to prevent infinite loops. Both `score_threshold` (default 0.5) and `max_rewrites` (default 2) are configurable via the Settings view (see §6.1).
- **R2.2 Reranking:** Retrieve a wide candidate set (top-20) from Chroma, then narrow to the final top-`rerank_top_n` using Cohere Rerank (`cohere-rerank-v3`, cross-encoder) via `ContextualCompressionRetriever`. `rerank_top_n` (default 3) is configurable via the Settings view. This should be the default retrieval path (not conditional).
- **R2.3 Conditional behavior = agentic:** The system must behave differently depending on retrieval quality per-turn — that conditionality is what makes it "agentic," not merely "has tools." Implement via an explicit branch (score check), not a general-purpose agent framework, unless a framework is preferred for clarity.
- **R2.4 Trace logging:** Log every decision point per query: original query, retrieval scores, whether a rewrite fired (and the rewritten query), whether reranking fired, final chunks used. Structure as Reason / Act / Observe steps.
- **R2.5 Trace UI:** Add a fourth UI view rendering the trace as a timeline, so a user can see exactly what the system did for a given question — not just the final answer.
- **R2.6 Settings extension:** Extend the Settings view (introduced in Stage 1) with controls for `score_threshold`, `max_rewrites`, and `rerank_top_n`, and add the active reranker model (`cohere-rerank-v3`) to the read-only model display alongside the LLM/embedding models.
- **R2.7 Stretch (optional):** Swap the query-expansion step for `MultiQueryRetriever` (LLM generates 3 phrasings, retrieve×3, union+dedupe) or HyDE (embed a hypothetical answer instead of the raw question) as alternate retrieval-harness techniques. Not required for the core build.

### 8.3 Tasks

1. Add a similarity-score check after initial retrieval, reading `score_threshold` from the runtime config (default 0.5).
2. Implement bounded query-rewrite retry (LLM rewrite → re-embed → re-search → recheck, up to `max_rewrites` attempts).
3. Obtain a free-tier Cohere API key; wire `CohereRerank` between the wide retrieval (top-20) and the LLM call, reading `rerank_top_n` from the runtime config (default 3).
4. Add a trace log data structure and persist/append a row per decision made during a query.
5. Add a Trace view to the UI rendering the timeline for the most recent (or a selected past) query.
6. Extend the Settings view with `score_threshold`, `max_rewrites`, `rerank_top_n` controls and the reranker model name.
7. Re-run the questions that failed or looked shaky in Stage 1 through the Stage 2 system; confirm rewrite/rerank visibly engage in the trace and improve the answer. Confirm changing the new settings visibly changes trace behavior.

### 8.4 Acceptance Criteria

- A query that scored well in Stage 1 does not trigger a rewrite in Stage 2 (rewrite is conditional, not unconditional).
- A query with a vocabulary mismatch (weak initial retrieval) triggers exactly one rewrite cycle (or up to the bound) and the retrieved chunks measurably improve (higher post-rewrite similarity score).
- Every query's final chunks come from the reranker, not raw vector similarity order, and their count matches `rerank_top_n`.
- The Trace view shows, for any given answer, the original query, scores, whether/what was rewritten, and confirmation reranking ran.
- Downstream chain code (`prompt | llm`) is unchanged from Stage 1 — only the retriever construction differs.
- `score_threshold`, `max_rewrites`, and `rerank_top_n` are changeable from the Settings view without editing code, and the reranker model name is visible in the UI.

### 8.5 Risks / Notes

- Cohere free tier caps at ~1000 calls/month — don't loop reranking calls unnecessarily.
- Don't over-engineer the trace UI; a simple ordered list/timeline is sufficient.

## 9. Stage 3 — Measurement / Evaluation (Target: ~35 min build)

### 9.1 Motivation

"If you can't measure it, you can't improve it, and you can't tell whether your last change made it better or worse." Stage 3 turns quality into a number so future changes to Stage 1/2 are evaluable rather than vibes-based.

### 9.2 Metrics (Ragas, four total)

| Side | Metric | Question it answers | Failure mode it catches |
|---|---|---|---|
| Generation | Faithfulness | Is each claim in the answer supported by retrieved chunks? | Model invents facts beyond what it read. |
| Generation | Answer relevance | Does the answer actually address the question? | Technically correct but unhelpful answer. |
| Retrieval | Context precision | Of retrieved chunks, how many were actually relevant? | Noise crowds out signal. |
| Retrieval | Context recall | Of chunks that should've been retrieved, how many were? | Right info exists but never made top-K. |

Ragas uses LLM-as-judge: for each metric it prompts an LLM (Gemini Flash Lite, free tier) with the question, retrieved chunks, and answer, and returns a score plus a per-claim breakdown.

### 9.3 Requirements

- **R3.1 Golden set:** A fixed set of 10 questions against the ingested documents, each with a known-correct answer (or a known-correct refusal, for out-of-scope questions). Store as a plain JSON file so it's version-controlled and reusable.
- **R3.2 Ragas integration:** Install and configure Ragas with the judge model. For each golden-set question, run it through the full Stage 2 agent, capture `(question, retrieved_chunks, answer, ground_truth)`, and score all 4 metrics.
- **R3.3 Evaluate UI view:** A fifth UI view that triggers a full golden-set run and displays a scorecard: the 4 metric averages across the whole set, plus a per-question breakdown table. Also display the judge model (`Gemini Flash Lite`) read-only, per §6.2.
- **R3.4 Actionable output:** The scorecard should make it obvious which metric is weakest, pointing to the next investment (e.g., low context recall → improve retrieval; low faithfulness → tighten the prompt or lower temperature further).

### 9.4 Tasks

1. Author a 10-question golden set (question + ground-truth answer) covering a spread of easy/hard/out-of-scope cases, saved as JSON.
2. Install Ragas; configure the judge model (Gemini Flash Lite, free tier) and API key.
3. Write the evaluation runner: iterate the golden set through the Stage 2 agent (using whatever `search_k`/`score_threshold`/`max_rewrites`/`rerank_top_n` are currently set in Settings), collect the four inputs Ragas needs per question, invoke Ragas scoring.
4. Add the Evaluate view to the UI: a "Run evaluation" action plus a rendered scorecard (aggregate + per-question), with the judge model name displayed.
5. Run the evaluation, identify the weakest metric, and write down what it points to as the next improvement.

### 9.5 Acceptance Criteria

- Running the golden set produces a scorecard with all 4 Ragas metrics, both aggregate and per-question.
- The evaluation run completes in a bounded, predictable time (10 questions × 4 metrics ≈ 40+ LLM calls; budget roughly 90 seconds) and doesn't hang.
- The scorecard is re-runnable after any Stage 1/2 change or any Settings change, so before/after comparisons are possible.
- A short write-up (even informal) exists identifying: strongest metric, weakest metric, remaining failure cases, and what they suggest for future work.
- The judge model name is visible on the Evaluate view.

### 9.6 Risks / Notes

- Ragas is slow (one LLM call per metric per question) — don't let the UI block without a loading indicator, and don't loop full-set evaluation in a tight cycle.
- Judge model bias: scores are directionally useful and good for tracking trend over time, but are not ground truth — don't over-index on small point differences between runs.

## 10. Cross-Cutting / Open Questions

- **UI framework:** confirm Streamlit vs. alternative before Stage 1 build starts (not graded, but changing mid-project is wasted effort).
- **Sample corpus:** decide whether to bring existing documents or generate a synthetic HR-handbook-style set for development and golden-set authoring.
- **Config management:** `chunk_size`, `chunk_overlap`, `search_k`, `score_threshold`, `max_rewrites`, `rerank_top_n`, and API keys/model names live in one config file/env as defaults; the Settings view (§6) is the runtime override surface — no magic numbers hardcoded in pipeline code.
- **Secrets handling:** Cohere and Gemini API keys (Stage 2/3) must be read from environment variables / a local `.env`, never hardcoded, committed, or exposed in the Settings UI.

## 11. Milestones

| Milestone | Depends on | Deliverable |
|---|---|---|
| M1 — Classic RAG working | — | Stage 1 acceptance criteria met, sample docs ingested and queryable, Settings view + model display live |
| M2 — Agentic retrieval working | M1 | Stage 2 acceptance criteria met, trace view functional, Settings extended with score/rewrite/rerank controls |
| M3 — Measured baseline | M2 | Stage 3 acceptance criteria met, first scorecard produced, weakest metric identified |
