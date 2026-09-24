# Equity Research Copilot

A simple student project for reading and comparing company financial reports. Upload PDFs, review key numbers, and ask questions with source pages. Built with Streamlit, LangChain, OpenAI embeddings and Chroma.

## Run locally

Use Python 3.11–3.13.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add your OPENAI_API_KEY to .env.
streamlit run app.py
```

Open http://localhost:8501. The interface loads without a key; report processing and research require a configured key with access to the selected model. Requests incur provider charges.

## Using the app

1. **Upload one PDF.** Use the upload control at the top to choose a searchable annual, quarterly, or earnings report (up to 50 MB). The app validates and analyzes it automatically. Failed uploads have a retry button.
2. **Review the summary.** Expand **Report summary** to read key insights, revenue, net income, diluted EPS, and risks. Financial values appear in full-width rows with explicit units. Unverified figures are labeled as unavailable.
3. **Check sources.** **Report sources and downloads** contains evidence and downloads. Page references appear beside findings. Open **Sources and evidence** to choose a page, review its figures in a table and read supporting quotes. Enable **Show original extracted text** to inspect the raw excerpt, or download the original PDF.
4. **Ask a question.** Enter your question in the central conversation box and press **Ask**. Chat uses only the selected PDF and remembers follow-up questions. Comparisons are possible when that PDF contains comparable prior-year figures.
5. **Switch PDFs.** Uploading another PDF selects it for analysis. The current PDF stays active while you chat; older reports are not mixed into its answers.
6. **Export notes.** Download the financial table as CSV or the conversation as Markdown.

## How it works

```text
One PDF → financial-report validation + issuer/period detection
    → duplicate lookup → page-preserving chunks
    → OpenAI embeddings → company-specific Chroma index
    → structured analysis of the selected PDF → quote and numeric checks
    → cached research dashboard + comparable financial series

Question + conversation → standalone question → query expansion
    → retrieval and reranking within the selected PDF → grounded answer + checked citations
```

The dashboard extracts core consolidated financial metrics, diluted EPS and disclosed product/cloud figures from current and comparative periods. Metrics preserve their printed sign and scale. Numeric quotes, period headers and unit excerpts must exist on the cited PDF page; the value's fiscal year must appear in its period quote. Narrative findings also need an exact supporting quote. These checks verify excerpts and numeric values, not full semantic entailment of every generated statement.

Charts normalize monetary units, keep currencies, accounting bases, entity scopes, fiscal-period types and reporting durations separate, and require at least two comparable points. Repeated comparative figures are deduplicated. Conflicting values are excluded from calculations and flagged rather than silently treated as restatements. Changes are computed in Python from accepted figures; changes in growth rates are expressed in percentage points.

Each report's dashboard analysis is cached under its company index in `research_cache/`, keyed by report metadata, model and analysis version. Adding a report analyzes only the new report. Chat reruns do not regenerate the dashboard. Failed report analyses are reported separately and can be retried without reprocessing successful reports.

Embeddings use `text-embedding-3-small`; the default research model is `gpt-4.1-mini`. `OPENAI_MODEL` overrides the chat model. Existing indexes remain readable in persistent mode. Older indexed reports without a saved original PDF remain searchable but have no PDF download.

Duplicate detection hashes normalized extracted text, so renaming a PDF does not create another report. A filesystem lock serializes ingestion, deterministic chunk IDs support retries, and manifests publish atomically after each report finishes indexing. Previously indexed reports remain available if a new upload fails.

## Workspace storage

| Setting | Default | Behavior |
| --- | --- | --- |
| `OPENAI_API_KEY` | Required for AI actions | Environment variable, `.env`, or Streamlit secrets |
| `OPENAI_MODEL` | `gpt-4.1-mini` | Chat, metadata, reranking and extraction model |
| `EQUITY_DATA_DIR` | `chroma_db` | Root directory for vectors, manifests and PDFs |
| `EQUITY_WORKSPACE_MODE` | `session` | `session` for isolated libraries; `persistent` for a trusted shared library |

**Session mode** isolates each Streamlit browser session under a random server-side directory. A browser reload/reconnect that starts a new session does not recover the previous library. Session files remain on disk until the deployment operator removes them; there is no automatic retention cleanup. Chat and results are held in session memory, so export notes before closing the app.

**Persistent mode** lists all company libraries in `EQUITY_DATA_DIR`, including the original project's existing indexes. Set `EQUITY_WORKSPACE_MODE=persistent` in `.env` to keep your local company library across sessions. This mode is shared across all visitors and has no built-in authentication; use it only locally or behind access control for trusted users. Local Chroma supports this single-instance app; horizontally scaled deployments require a shared database service and a redesigned locking/storage layer.

## Deploy to Streamlit Community Cloud

1. Push the project to a Git repository, excluding `.env`, `.streamlit/secrets.toml`, report PDFs and `chroma_db`.
2. Create an app in Community Cloud, choose the repository and branch, and set the entry point to `app.py`.
3. Add the following to the app's **Secrets** settings:

   ```toml
   OPENAI_API_KEY = "your-api-key"
   OPENAI_MODEL = "gpt-4.1-mini"
   EQUITY_WORKSPACE_MODE = "session"
   ```

4. Deploy, upload a report, let the dashboard generate, and verify a finding against its source page.

See Streamlit's [deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) and [secrets guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management). Do not rely on a hosting container's local filesystem for durable storage across replacement or redeployment. Use a host with a persistent volume when long-term libraries are required. A public deployment using your server API key needs platform access controls or additional authentication/rate limiting to control access and spending.

## Deploy with Docker and durable storage

```bash
docker build -t equity-research .
docker run --rm -p 8501:8501 --env-file .env \
  -e EQUITY_WORKSPACE_MODE=persistent \
  -v equity-data:/app/chroma_db equity-research
```

The image runs as a non-root user and includes a Streamlit health check. Use a named volume as above; bind-mounted directories must be writable by the container user. Put a TLS/authentication proxy in front of an externally accessible persistent deployment. Neither `.env` nor local report data is copied into the image.

## Verification

```bash
python -m unittest discover -s tests -v
python -m pip check
```

Tests use real PDFs, a temporary real Chroma database with deterministic test embeddings, and Streamlit's AppTest. AI calls are mocked so tests do not incur charges. Coverage includes automatic and incremental uploads, mixed-company rejection, duplicate handling, company isolation, rollback, citations, conversational history, dashboard caching and partial failures, chart unit conversion, incompatible-period separation, conflicting figures and percentage-point calculations. Model responses are mocked; real-model classification and extraction accuracy require manual validation with representative reports.

For manual live evaluation after indexing reports:

```bash
python rag.py chroma_db/your-company
python retrieve.py chroma_db/your-company "How did revenue change?"
python evaluation.py chroma_db/your-company
```

These commands use the provider API. For session-mode indexes, pass the actual company directory under `chroma_db/sessions/<session-id>/`.

## Boundaries

- Searchable PDFs only. OCR and specialized table-layout reconstruction are not implemented.
- Financial-report classification and metadata are AI-detected, with verbatim issuer and financial-results excerpts checked against the document. Classification can still make mistakes. Issuer detection happens automatically; punctuation and common legal suffixes are normalized conservatively. Unrelated names are never fuzzy-merged. Annual/quarterly reports and earnings releases are accepted; unrelated documents are rejected. Legacy duplicate uploads are revalidated. Existing libraries are not automatically deleted or rewritten.
- Citation validation checks that returned source labels exist in retrieved evidence; it does **not** prove every sentence is entailed. Analysts should inspect the displayed passages.
- Financial extraction checks that the quote is present and the signed value occurs in it. Period, scale, currency and accounting classification still require analyst review. Dashboard extraction runs automatically after indexing, but is not an exhaustive audit of every financial statement.
- Chat comparisons are generated by the model from retrieved evidence. Dashboard trend changes are calculated in Python from validated extracted values. Missing context, incompatible periods and insufficient disclosure can prevent a complete comparison. A fully reconciled financial model remains outside the scope of this app.
- PDFs and extracted text are stored on the server; relevant text is sent to OpenAI for processing. The app has no portfolio execution or live market-data integration.

## Project files

- `app.py`: interface and session-scoped workflows
- `ingest.py`: PDF validation, duplicate handling, company libraries and Chroma writes
- `report_metadata.py`: financial-report validation and structured issuer/period detection
- `upload_queue.py`: upload fingerprints and same-issuer batch validation
- `research.py`: conversational retrieval, reranking, answer generation and citation checks
- `dashboard.py`: automatic report briefs, persistent caching, validated metrics and trend calculations
- `metrics.py`: reusable financial extraction and numeric-quote validation helpers
- `settings.py`: environment configuration and lazy API clients
- `tests/`: offline backend and Streamlit interaction checks
- `.streamlit/config.toml`, `Dockerfile`: theme and deployment configuration

## Code style and learning guide

See [WALKTHROUGH.md](WALKTHROUGH.md) for a step-by-step explanation of the upload and question pipelines, design decisions, and limitations you can discuss in a project demonstration.

Install development tools and run the same checks before committing:

```bash
pip install -r requirements-dev.txt
ruff check .
ruff format --check .
python -m unittest discover -s tests -v
```

Use `ruff format .` to apply formatting. The style configuration lives in `pyproject.toml`. Keep functions focused, use named constants for processing limits, and document why a validation rule exists. When changing dashboard extraction rules, increase `VERSION` in `dashboard.py` so old cached analyses are regenerated.
