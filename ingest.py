import hashlib
import json
import shutil
from pathlib import Path
import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from report_metadata import extract_report_metadata


def ingest_pdf(
    file_bytes: bytes,
    filename: str
):

    # ----------------------------------
    # 1. Generate unique report ID
    # ----------------------------------

    report_id = hashlib.sha256(
        file_bytes
    ).hexdigest()[:12]


    index_dir = (
        Path("chroma_db") / report_id
    )


    marker_file = (
        index_dir / ".indexed"
    )


    metadata_file = (
        index_dir / "metadata.json"
    )


    # ----------------------------------
    # 2. Reuse existing index + metadata
    # ----------------------------------

    if marker_file.exists() and metadata_file.exists():

        with open(
            metadata_file,
            "r",
            encoding="utf-8"
        ) as f:

            saved_metadata = json.load(f)


        return {
            "report_id": report_id,
            "index_dir": str(index_dir),

            "filename": saved_metadata["filename"],

            "company_name": saved_metadata["company_name"],
            "fiscal_year": saved_metadata["fiscal_year"],
            "report_type": saved_metadata["report_type"],

            "pages": saved_metadata.get("pages"),
            "chunks": saved_metadata.get("chunks"),

            "reused": True
        }


    # ----------------------------------
    # 3. Remove incomplete previous index
    # ----------------------------------

    if index_dir.exists():

        shutil.rmtree(
            index_dir
        )


    index_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # ----------------------------------
    # 4. Detect report metadata
    # ----------------------------------

    metadata = extract_report_metadata(
        file_bytes
    )


    # ----------------------------------
    # 5. Extract PDF text
    # ----------------------------------

    pdf = pymupdf.open(
        stream=file_bytes,
        filetype="pdf"
    )


    total_pages = len(pdf)

    documents = []


    for page_number, page in enumerate(
        pdf,
        start=1
    ):

        text = page.get_text("text")

        if text.strip():

            documents.append(
                Document(
                    page_content=text,

                    metadata={
                        "source": filename,
                        "page": page_number
                    }
                )
            )


    # ----------------------------------
    # 6. Split into chunks
    # ----------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )


    chunks = splitter.split_documents(
        documents
    )


    # ----------------------------------
    # 7. Create embeddings
    # ----------------------------------

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )


    # ----------------------------------
    # 8. Create Chroma database
    # ----------------------------------

    vector_store = Chroma(
        collection_name="equity_research",
        embedding_function=embeddings,
        persist_directory=str(index_dir)
    )


    vector_store.add_documents(
        chunks
    )


    # ----------------------------------
    # 9. Save report metadata
    # ----------------------------------

    metadata_to_save = {
        "filename": filename,

        "company_name": metadata.company_name,
        "fiscal_year": metadata.fiscal_year,
        "report_type": metadata.report_type,

        "pages": total_pages,
        "chunks": len(chunks)
    }


    with open(
        metadata_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata_to_save,
            f,
            indent=4
        )


    # ----------------------------------
    # 10. Mark indexing complete
    # ----------------------------------

    marker_file.touch()


    pdf.close()


    # ----------------------------------
    # 11. Return report information
    # ----------------------------------

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