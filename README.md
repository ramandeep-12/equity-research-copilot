# Equity Research Copilot

An AI equity-research platform for conversational analysis and comparison of company financial reports. Built with Streamlit, LangChain, OpenAI embeddings and Chroma.

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

## Analyst workflow

1. **Report library:** upload searchable PDFs (50 MB / 2,000 pages maximum per file), then select **Detect report details**. Review and correct issuer, fiscal year, report type and fiscal period. Use one consistent company name for all reports from an issuer. Select **Confirm & index reports**. Different issuers are placed in separate knowledge bases.
2. **Research chat:** choose a company and up to 12 reports. Ask questions about growth, margins, cash flow, strategy or risks. Follow-ups use the last six messages; changing the company or report scope switches to a separate conversation. Export notes as Markdown.
3. **Compare periods:** choose distinct baseline and comparison reports, enter a topic and select **Analyze changes**. Evidence is retrieved and reranked independently for each report. Responses distinguish periods, numeric differences, disclosed drivers and a takeaway when the evidence permits.
4. **Financial metrics:** extract consolidated revenue, operating income, net income or operating cash flow for each report's current covered period. Rows retain period, reported scale, currency, accounting basis, source page and quote. Missing or unverifiable figures are explicitly reported. Export CSV for further analysis.
5. Expand numbered citations to inspect evidence and download the original report. Citations use physical, one-based **PDF pages, including the cover**, which can differ from printed page labels.

## How it works

```text
PDF uploads → content fingerprint → duplicate lookup → metadata detection + analyst review
    → page-preserving text extraction → overlapping chunks → OpenAI embeddings
    → company-specific Chroma index + atomic report manifest

Question + conversation → standalone question → query expansion
    → retrieval per selected report → reranking per report
    → structured answer → citation-reference validation → answer + evidence
```

Embeddings use `text-embedding-3-small`; the default research and metadata model is `gpt-4.1-mini`. `OPENAI_MODEL` can override the chat model. Model clients initialize only when needed. Existing company indexes from the original project remain readable in persistent mode. Previously indexed PDFs without a saved original remain searchable, but have no PDF download.

Duplicate detection hashes normalized extracted text, so renaming a file or changing PDF container metadata does not create another index entry. Image-only changes with identical extracted text are considered duplicates. A filesystem lock serializes ingestion, deterministic chunk IDs support retries, and manifests publish atomically only after indexing completes. Failed attempts remove their report's vectors. No other reports are removed.

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

4. Deploy, upload a report, confirm metadata, and verify an answer against its source page.

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

Tests use real PDFs, a temporary real Chroma database with deterministic test embeddings, and Streamlit's AppTest. AI calls are mocked so tests do not incur charges. Coverage includes duplicates, isolation, invalid PDFs, rollback, report-balanced retrieval, citations, conversational history, extraction quote/value checks, missing-key startup, chat and comparison interactions.

For manual live evaluation after indexing reports:

```bash
python rag.py chroma_db/your-company
python retrieve.py chroma_db/your-company "How did revenue change?"
python evaluation.py chroma_db/your-company
```

These commands use the provider API. For session-mode indexes, pass the actual company directory under `chroma_db/sessions/<session-id>/`.

## Boundaries

- Searchable PDFs only. OCR and specialized table-layout reconstruction are not implemented.
- Metadata is AI-detected and explicitly reviewed before indexing. Company aliases are resolved by using the same reviewed name.
- Citation validation checks that returned source labels exist in retrieved evidence; it does **not** prove every sentence is entailed. Analysts should inspect the displayed passages.
- Financial extraction checks that the quote is present and the signed value occurs in it. Period, scale, currency and accounting classification still require analyst review. Extraction is on-demand, not an exhaustive audit of every financial statement.
- Comparisons and change calculations are generated by the model from retrieved evidence. Missing context, incompatible periods and insufficient disclosure can prevent a complete comparison. Automatic trend charts and an independently reconciled financial model are future work.
- PDFs and extracted text are stored on the server; relevant text is sent to OpenAI for processing. The app has no portfolio execution or live market-data integration.

## Project files

- `app.py`: interface and session-scoped workflows
- `ingest.py`: PDF validation, duplicate handling, company libraries and Chroma writes
- `report_metadata.py`: structured issuer and period detection
- `research.py`: conversational retrieval, reranking, answer generation and citation checks
- `metrics.py`: evidence-checked structured financial extraction
- `settings.py`: environment configuration and lazy API clients
- `tests/`: offline backend and Streamlit interaction checks
- `.streamlit/config.toml`, `Dockerfile`: theme and deployment configuration
