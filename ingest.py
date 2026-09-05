from pathlib import Path
from dotenv import load_dotenv
import fitz
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


load_dotenv()


# --------------------------------
# 1. Load PDF
# --------------------------------

pdf_path = Path("data/annual_report.pdf")

pdf = fitz.open(pdf_path)

documents = []

for page_number, page in enumerate(pdf, start=1):

    text = page.get_text("text")

    if text.strip():

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": "annual_report.pdf",
                    "page": page_number
                }
            )
        )

print(f"Pages loaded: {len(pdf)}")
print(f"Pages containing text: {len(documents)}")


# --------------------------------
# 2. Split PDF into chunks
# --------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)

chunks = text_splitter.split_documents(documents)

print(f"Chunks created: {len(chunks)}")


# --------------------------------
# 3. Create embeddings
# --------------------------------

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


# --------------------------------
# 4. Create Chroma vector database
# --------------------------------

vector_store = Chroma(
    collection_name="equity_research",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)


# --------------------------------
# 5. Store chunks
# --------------------------------

vector_store.add_documents(chunks)

print("Annual report indexed successfully.")