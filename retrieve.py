from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


load_dotenv()


# -----------------------------
# 1. Load embedding model
# -----------------------------

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


# -----------------------------
# 2. Connect to existing Chroma DB
# -----------------------------

vector_store = Chroma(
    collection_name="equity_research",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)


# -----------------------------
# 3. Ask a question
# -----------------------------

question = "What are the major risks faced by the company?"


# -----------------------------
# 4. Search vector database
# -----------------------------

results = vector_store.similarity_search(
    question,
    k=4
)


# -----------------------------
# 5. Print retrieved chunks
# -----------------------------

for index, doc in enumerate(results, start=1):

    print("\n" + "=" * 60)
    print(f"RESULT {index}")
    print("=" * 60)

    print(f"Page: {doc.metadata.get('page')}")
    print(f"Source: {doc.metadata.get('source')}")

    print("\nContent:\n")
    print(doc.page_content)