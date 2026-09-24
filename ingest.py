"""Page-preserving ingestion with content deduplication and atomic manifests."""

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pymupdf
from filelock import FileLock
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from report_metadata import ReportMetadata, company_key, extract_report_metadata
from settings import data_root, get_embeddings

MAX_PDF_BYTES = 50 * 1024 * 1024
MAX_PDF_PAGES = 2000
CHUNK_SIZE = 1800
CHUNK_OVERLAP = 300
EMBEDDING_BATCH_SIZE = 96


def create_company_slug(company_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", company_name.lower()).strip("-")
    return slug or hashlib.sha256(company_name.encode()).hexdigest()[:16]


def read_pdf(file_bytes: bytes) -> tuple[list[tuple[int, str]], int, str]:
    """Return numbered text pages, total page count, and a content-based report ID."""
    if len(file_bytes) > MAX_PDF_BYTES:
        raise ValueError("Each report must be 50 MB or smaller.")
    try:
        with pymupdf.open(stream=file_bytes, filetype="pdf") as pdf:
            if pdf.needs_pass:
                raise ValueError("This PDF is password protected. Upload an unlocked copy.")
            if len(pdf) > MAX_PDF_PAGES:
                raise ValueError("Reports must have no more than 2,000 pages.")
            total = len(pdf)
            pages = [(i + 1, page.get_text("text")) for i, page in enumerate(pdf)]
    except (pymupdf.FileDataError, RuntimeError) as exc:
        raise ValueError("This file could not be read as a PDF.") from exc
    pages = [(number, text) for number, text in pages if text.strip()]
    if not pages:
        raise ValueError("No readable text found. Scanned PDFs require OCR before uploading.")
    normalized = re.sub(r"\s+", " ", "\n".join(t for _, t in pages)).strip()
    # Compatible with the project's existing report IDs.
    return pages, total, hashlib.sha256(normalized.encode()).hexdigest()[:12]


def list_companies(root=None):
    root = Path(root) if root is not None else data_root()
    companies = []
    for path in sorted(root.glob("*/metadata.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("reports"):
            companies.append({**manifest, "index_dir": str(path.parent)})
    return companies


def find_duplicate(report_id, root=None):
    for company in list_companies(root):
        for report in company["reports"]:
            if report["report_id"] == report_id:
                return {
                    **report,
                    "company_name": company["company_name"],
                    "index_dir": company["index_dir"],
                    "reused": True,
                }
    return None


def prepare_report(file_bytes, filename, root=None):
    """Detect report identity without writing PDFs, vectors, or manifests."""
    _, pages, report_id = read_pdf(file_bytes)
    duplicate = find_duplicate(report_id, root)
    if duplicate:
        # Old indexes predate document classification and cannot bypass validation.
        if not duplicate.get("validation_version"):
            extract_report_metadata(file_bytes)
        return duplicate
    metadata = extract_report_metadata(file_bytes)
    return {
        **metadata.model_dump(),
        "filename": Path(filename).name,
        "pages": pages,
        "report_id": report_id,
        "reused": False,
    }


def ingest_pdf(file_bytes: bytes, filename: str, metadata=None, root=None):
    """Validate and index one PDF, publishing its manifest only after successful writes."""
    pages, total_pages, report_id = read_pdf(file_bytes)
    root = Path(root) if root is not None else data_root()
    root.mkdir(parents=True, exist_ok=True)
    # Serialize mutations across sessions/processes, including duplicate checks.
    with FileLock(str(root / ".ingest.lock"), timeout=180):
        duplicate = find_duplicate(report_id, root)
        if duplicate:
            if not duplicate.get("validation_version"):
                detected = extract_report_metadata(file_bytes)
                if company_key(detected.company_name) != company_key(duplicate["company_name"]):
                    raise ValueError("The previously indexed issuer does not match this report.")
            return duplicate
        # Revalidate at the write boundary: manually supplied metadata must never
        # turn an unrelated PDF into an accepted financial report.
        detected = extract_report_metadata(file_bytes)
        if metadata is None:
            metadata = detected
        elif isinstance(metadata, dict):
            metadata = ReportMetadata.model_validate(metadata)
        if company_key(metadata.company_name) != company_key(detected.company_name):
            raise ValueError("The reviewed company does not match the document's issuer.")
        if metadata.report_type not in {"Annual Report", "Quarterly Report", "Earnings Report"}:
            raise ValueError("Select a supported financial report type.")
        metadata.company_name = metadata.company_name.strip()
        if not metadata.company_name or metadata.company_name.lower() == "unknown":
            raise ValueError("Confirm the company name before indexing.")
        if not re.fullmatch(r"\d{4}", metadata.fiscal_year):
            raise ValueError("Confirm a four-digit fiscal year before indexing.")
        matches = [
            company
            for company in list_companies(root)
            if company_key(company["company_name"]) == company_key(metadata.company_name)
        ]
        if len(matches) > 1:
            raise ValueError(
                "Multiple legacy libraries match this issuer. Consolidate them before adding reports."
            )
        if matches:
            metadata.company_name = matches[0]["company_name"]
        index_dir = (
            Path(matches[0]["index_dir"])
            if matches
            else root / create_company_slug(metadata.company_name)
        )
        index_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = index_dir / "metadata.json"
        manifest = (
            json.loads(manifest_path.read_text())
            if manifest_path.exists()
            else {"company_name": metadata.company_name, "reports": []}
        )
        documents = [
            Document(
                page_content=text,
                metadata={
                    "source": Path(filename).name,
                    "page": page,
                    "report_id": report_id,
                    **metadata.model_dump(),
                },
            )
            for page, text in pages
        ]
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
        ).split_documents(documents)
        store = Chroma(
            collection_name="equity_research",
            embedding_function=get_embeddings(),
            persist_directory=str(index_dir),
        )
        ids = [f"{report_id}-{i}" for i in range(len(chunks))]
        try:
            # Remove remnants of an interrupted attempt before deterministic upsert.
            store.delete(where={"report_id": report_id})
            for offset in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
                store.add_documents(
                    chunks[offset : offset + EMBEDDING_BATCH_SIZE],
                    ids=ids[offset : offset + EMBEDDING_BATCH_SIZE],
                )
            (index_dir / f"{report_id}.pdf").write_bytes(file_bytes)
            report = {
                "report_id": report_id,
                "filename": Path(filename).name,
                **metadata.model_dump(),
                "pages": total_pages,
                "chunks": len(chunks),
                "indexed_at": datetime.now(timezone.utc).isoformat(),
                "validation_version": 1,
            }
            manifest["reports"].append(report)
            temporary = manifest_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            temporary.replace(manifest_path)
        except Exception:
            store.delete(where={"report_id": report_id})
            raise
        return {**report, "index_dir": str(index_dir), "reused": False}
