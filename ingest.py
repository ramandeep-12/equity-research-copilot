"""Page-preserving ingestion with content deduplication and atomic manifests."""
import hashlib
import json
import re
from pathlib import Path
from datetime import datetime, timezone

import pymupdf
from filelock import FileLock
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from report_metadata import ReportMetadata, extract_report_metadata
from settings import data_root, get_embeddings


def create_company_slug(company_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", company_name.lower()).strip("-")
    return slug or hashlib.sha256(company_name.encode()).hexdigest()[:16]


def read_pdf(file_bytes):
    if len(file_bytes) > 50 * 1024 * 1024:
        raise ValueError("Each report must be 50 MB or smaller.")
    try:
        with pymupdf.open(stream=file_bytes, filetype="pdf") as pdf:
            if pdf.needs_pass:
                raise ValueError("This PDF is password protected. Upload an unlocked copy.")
            if len(pdf) > 2000:
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
                return {**report, "company_name": company["company_name"],
                        "index_dir": company["index_dir"], "reused": True}
    return None


def prepare_report(file_bytes, filename, root=None):
    _, pages, report_id = read_pdf(file_bytes)
    duplicate = find_duplicate(report_id, root)
    if duplicate:
        return duplicate
    metadata = extract_report_metadata(file_bytes)
    return {**metadata.model_dump(), "filename": Path(filename).name,
            "pages": pages, "report_id": report_id, "reused": False}


def ingest_pdf(file_bytes: bytes, filename: str, metadata=None, root=None):
    pages, total_pages, report_id = read_pdf(file_bytes)
    root = Path(root) if root is not None else data_root()
    root.mkdir(parents=True, exist_ok=True)
    # Serialize mutations across sessions/processes, including duplicate checks.
    with FileLock(str(root / ".ingest.lock"), timeout=180):
        duplicate = find_duplicate(report_id, root)
        if duplicate:
            return duplicate
        if metadata is None:
            metadata = extract_report_metadata(file_bytes)
        elif isinstance(metadata, dict):
            metadata = ReportMetadata.model_validate(metadata)
        metadata.company_name = metadata.company_name.strip()
        if not metadata.company_name or metadata.company_name.lower() == "unknown":
            raise ValueError("Confirm the company name before indexing.")
        if not re.fullmatch(r"\d{4}", metadata.fiscal_year):
            raise ValueError("Confirm a four-digit fiscal year before indexing.")
        index_dir = root / create_company_slug(metadata.company_name)
        index_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = index_dir / "metadata.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
            "company_name": metadata.company_name, "reports": []}
        documents = [Document(page_content=text, metadata={
            "source": Path(filename).name, "page": page,
            "report_id": report_id, **metadata.model_dump()}) for page, text in pages]
        chunks = RecursiveCharacterTextSplitter(chunk_size=1800, chunk_overlap=300,
                                                add_start_index=True).split_documents(documents)
        store = Chroma(collection_name="equity_research", embedding_function=get_embeddings(),
                       persist_directory=str(index_dir))
        ids = [f"{report_id}-{i}" for i in range(len(chunks))]
        try:
            # Remove remnants of an interrupted attempt before deterministic upsert.
            store.delete(where={"report_id": report_id})
            for offset in range(0, len(chunks), 96):
                store.add_documents(chunks[offset:offset + 96], ids=ids[offset:offset + 96])
            (index_dir / f"{report_id}.pdf").write_bytes(file_bytes)
            report = {"report_id": report_id, "filename": Path(filename).name,
                      **metadata.model_dump(), "pages": total_pages, "chunks": len(chunks),
                      "indexed_at": datetime.now(timezone.utc).isoformat()}
            manifest["reports"].append(report)
            temporary = manifest_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            temporary.replace(manifest_path)
        except Exception:
            store.delete(where={"report_id": report_id})
            raise
        return {**report, "index_dir": str(index_dir), "reused": False}
