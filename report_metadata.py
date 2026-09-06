import pymupdf

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


# Load OPENAI_API_KEY before creating ChatOpenAI
load_dotenv()


class ReportMetadata(BaseModel):
    company_name: str = Field(
        description="Company that issued the financial report"
    )

    fiscal_year: str = Field(
        description="Fiscal year covered by the report"
    )

    report_type: str = Field(
        description="Type of financial report such as Annual Report, 10-K, or Quarterly Report"
    )


metadata_llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
).with_structured_output(ReportMetadata)


def extract_report_metadata(file_bytes: bytes) -> ReportMetadata:

    pdf = pymupdf.open(
        stream=file_bytes,
        filetype="pdf"
    )

    first_pages = []

    for page_number in range(
        min(5, len(pdf))
    ):

        text = pdf[page_number].get_text("text")

        if text.strip():
            first_pages.append(text)

    pdf.close()

    sample_text = "\n\n".join(first_pages)

    result = metadata_llm.invoke(
        f"""
Identify the financial report below.

Determine:
- company name
- fiscal year
- report type

Use ONLY information explicitly present in the document.

If any field cannot be determined, return "Unknown".

DOCUMENT:

{sample_text}
"""
    )

    return result