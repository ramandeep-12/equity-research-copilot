import hashlib
import json
import re
from pathlib import Path

import pymupdf

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from report_metadata import extract_report_metadata


# --------------------------------------------------
# Create safe folder name for company
# --------------------------------------------------

def create_company_slug(company_name: str) -> str:

    slug = company_name.lower()

    slug = re.sub(
        r"[^a-z0-9]+",
        "-",
        slug
    )

    return slug.strip("-")


# --------------------------------------------------
# Ingest financial report
# --------------------------------------------------

def ingest_pdf(
    file_bytes: bytes,
    filename: str
):

    # --------------------------------------------------
    # 1. Detect report metadata
    # --------------------------------------------------

    metadata = extract_report_metadata(
        file_bytes
    )


    company_slug = create_company_slug(
        metadata.company_name
    )


    # --------------------------------------------------
    # 2. Company-specific database folder
    # --------------------------------------------------

    index_dir = (
        Path("chroma_db")
        / company_slug
    )

    index_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    metadata_file = (
        index_dir
        / "metadata.json"
    )


    # --------------------------------------------------
    # 3. Extract PDF text once
    # --------------------------------------------------

    pdf = pymupdf.open(
        stream=file_bytes,
        filetype="pdf"
    )

    total_pages = len(pdf)

    extracted_pages = []
    all_text = []


    for page_number, page in enumerate(
        pdf,
        start=1
    ):

        text = page.get_text("text")

        if text.strip():

            all_text.append(text)

            extracted_pages.append(
                {
                    "page": page_number,
                    "text": text
                }
            )


    pdf.close()


    # --------------------------------------------------
    # 4. Create report ID from actual document content
    # --------------------------------------------------

    normalized_text = re.sub(
        r"\s+",
        " ",
        "\n".join(all_text)
    ).strip()


    report_id = hashlib.sha256(
        normalized_text.encode("utf-8")
    ).hexdigest()[:12]


    # --------------------------------------------------
    # 5. Load company metadata
    # --------------------------------------------------

    if metadata_file.exists():

        with open(
            metadata_file,
            "r",
            encoding="utf-8"
        ) as f:

            company_metadata = json.load(f)

    else:

        company_metadata = {
            "company_name": metadata.company_name,
            "reports": []
        }


    # --------------------------------------------------
    # 6. Check if exact report already exists
    # --------------------------------------------------

    existing_report = next(
        (
            report
            for report in company_metadata["reports"]
            if report["report_id"] == report_id
        ),
        None
    )


    if existing_report:

        return {
            "report_id": report_id,
            "index_dir": str(index_dir),

            "filename": existing_report["filename"],

            "company_name": company_metadata["company_name"],

            "fiscal_year": existing_report["fiscal_year"],
            "report_type": existing_report["report_type"],

            "pages": existing_report.get("pages"),
            "chunks": existing_report.get("chunks"),

            "reused": True
        }


    # --------------------------------------------------
    # 7. Create LangChain Documents
    # --------------------------------------------------

    documents = []


    for page_data in extracted_pages:

        documents.append(
            Document(
                page_content=page_data["text"],

                metadata={
                    "source": filename,
                    "page": page_data["page"],

                    "report_id": report_id,

                    "company_name": metadata.company_name,

                    "fiscal_year": metadata.fiscal_year,

                    "report_type": metadata.report_type
                }
            )
        )


    # --------------------------------------------------
    # 8. Split into chunks
    # --------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )


    chunks = splitter.split_documents(
        documents
    )


    # --------------------------------------------------
    # 9. Embeddings
    # --------------------------------------------------

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )


    # --------------------------------------------------
    # 10. Open/create company Chroma DB
    # --------------------------------------------------

    vector_store = Chroma(
        collection_name="equity_research",
        embedding_function=embeddings,
        persist_directory=str(index_dir)
    )


    # --------------------------------------------------
    # 11. Add this report's chunks
    # --------------------------------------------------

    chunk_ids = [
        f"{report_id}-{i}"
        for i in range(len(chunks))
    ]


    vector_store.add_documents(
        documents=chunks,
        ids=chunk_ids
    )


    # --------------------------------------------------
    # 12. Save report metadata
    # --------------------------------------------------

    report_metadata = {
        "report_id": report_id,

        "filename": filename,

        "fiscal_year": metadata.fiscal_year,

        "report_type": metadata.report_type,

        "pages": total_pages,

        "chunks": len(chunks)
    }


    company_metadata["reports"].append(
        report_metadata
    )


    with open(
        metadata_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            company_metadata,
            f,
            indent=4
        )


    # --------------------------------------------------
    # 13. Return result
    # --------------------------------------------------

    return {
        "report_id": report_id,

        "index_dir": str(index_dir),

        "filename": filename,

        "company_name": metadata.company_name,

        "fiscal_year": metadata.fiscal_year,

        "report_type": metadata.report_type,

        "pages": total_pages,

        "chunks": len(chunks),

        "reused": False
    }