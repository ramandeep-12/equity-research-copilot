# Explaining Equity Research Copilot

## A short introduction

“Equity Research Copilot helps a user read a company's financial reports. The user uploads a searchable PDF, gets a summary with financial metrics, and asks follow-up questions. The app retrieves relevant passages and shows source pages so the user can review the evidence.”

Use this as a starting point in your own words. Describe the parts you understand and contributed to, and acknowledge libraries and coding assistance when relevant.

## Follow one upload through the code

1. **Interface — `app.py`.** Streamlit displays the upload control, summary, sources, and conversation. Session state keeps the selected report and chat history between interface reruns.
2. **Read and identify — `ingest.py`, `report_metadata.py`.** PyMuPDF extracts text with page numbers. The app checks size and readability, then asks the model for structured issuer and fiscal-period metadata. Supporting excerpts are checked against the extracted text.
3. **Prevent duplicates — `ingest.py`.** A hash of whitespace-normalized document text identifies a report even if its filename changes. Company names are normalized conservatively to keep separate issuers apart.
4. **Create searchable chunks — `ingest.py`.** Text is split into 1,800-character chunks with 300 characters of overlap. Overlap helps preserve context near chunk boundaries. Each chunk retains its report identity and PDF page number.
5. **Store embeddings — `ingest.py`, `settings.py`.** OpenAI converts chunks into numerical vectors. Chroma stores them in a company-specific directory for similarity search. A filesystem lock serializes ingestion, and a temporary manifest is renamed into place after indexing succeeds.
6. **Build the summary — `dashboard.py`.** Topic searches find evidence for financial results, segments, risks, and growth drivers. Pydantic defines the output fields the model must return. Python validates supporting excerpts before findings are displayed.
7. **Reuse results — `dashboard.py`.** Each report's analysis is cached using its metadata, model name, and analysis version. A normal chat rerun can reuse the analysis instead of generating it again.

## Follow one question through the code

The main entry point is `ask_equity_question()` in `research.py`:

1. Check that the question is nonempty and within the length limit.
2. Rewrite a follow-up such as “Why did it increase?” into a standalone question using recent chat history.
3. Generate alternative search phrases while retaining the original question.
4. Search the selected reports and rerank candidate chunks for relevance.
5. Group chunks by report and page and assign source labels such as `S1`.
6. Ask the model for an answer using those passages and citation labels.
7. Check that cited labels match the declared sources and refer to retrieved evidence.

This is retrieval-augmented generation (RAG): retrieved document text provides context for the model's answer. It does not train the model on the uploaded PDF. The current interface selects one PDF; the backend also contains support for comparisons across selected reports.

## Explain the financial checks

- **Quotes:** an excerpt must occur in the retrieved content after whitespace normalization.
- **Values:** the printed number must match the extracted value. Parentheses are interpreted as a negative sign, as in `(250)`.
- **Periods and units:** dashboard metrics need supporting period and unit excerpts. The fiscal year must appear in the period quote, and the duration must match the period type.
- **Comparability:** chart series separate scope, currency, accounting basis, period, and duration. Monetary scales are normalized before comparison.
- **Conflicts:** when reports give different normalized values for the same series and year, that point is excluded and flagged.
- **Changes:** Python calculates dashboard differences. Moving from 10% to 12% is an increase of 2 percentage points.

These checks catch certain unsupported outputs. They do not prove that the model interpreted the correct table column or that every sentence follows from its citation. The source view remains part of the review workflow.

## Design decisions you should understand

| Decision | Reason | Tradeoff |
| --- | --- | --- |
| Streamlit | Build an interactive Python interface quickly | Reruns require careful session-state handling |
| Chroma | Persist and search report embeddings locally | The current storage design targets one app instance |
| Structured model output | Require predictable fields for validation and display | Correct structure does not guarantee correct facts |
| Reranking | Refine the initial similarity search | Adds model calls, cost, and latency |
| Per-report cache | Avoid repeating dashboard analysis | Extraction-rule changes require an analysis-version bump |
| Page metadata | Let users trace findings to the original report | PDF text extraction can lose table layout |

## Demonstrate the project

Upload a searchable report, open a summary finding's source page, and check the number and period. Ask a specific question, then a follow-up. Show how the citation connects the answer to the report. Explain what happens when comparable evidence is missing rather than promising that the model always answers correctly.

For code review, start with `ask_equity_question()`, then `validate_brief()`, then `ingest_pdf()`. Those functions show the question pipeline, evidence checks, and storage lifecycle.

## Explain the verification honestly

The automated suite uses mocked model responses, temporary storage, real PDF fixtures, and Streamlit interaction tests. It checks application behavior without paid API calls. `evaluation.py` prints live answers and sources for manual inspection; it does not calculate an accuracy score. Model quality needs separate evaluation against representative reports and reviewed expected answers.

Possible future improvements include OCR for scanned PDFs, better table extraction, a reviewed evaluation dataset, and authenticated persistent workspaces. These are future work, not implemented capabilities.
