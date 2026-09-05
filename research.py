from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# 2. Embedding model
# --------------------------------------------------

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


# --------------------------------------------------
# 3. Connect to existing Chroma database
# --------------------------------------------------

vector_store = Chroma(
    collection_name="equity_research",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)


# --------------------------------------------------
# 4. LLM
# --------------------------------------------------

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)

class ResearchAnswer(BaseModel):
    answer: str = Field(
        description="Final grounded equity research answer"
    )

    used_pages: list[int] = Field(
        description="Page numbers actually used to support the answer"
    )


# --------------------------------------------------
# 5. Output parser
# --------------------------------------------------

structured_llm = llm.with_structured_output(
    ResearchAnswer
)


# --------------------------------------------------
# 6. Research prompt
# --------------------------------------------------

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an equity research assistant analyzing a company's annual report.

Answer the user's question using ONLY the supplied annual report context.

Rules:

1. Do not use outside knowledge.
2. Do not invent facts or financial figures.
3. Include only information that directly helps answer the question.
4. Cite supporting pages naturally in the answer, for example:
   (Page 44)
5. Every page cited in the answer MUST also appear in used_pages.
6. used_pages must contain ONLY pages actually used to support the final answer.
7. Do not include a page merely because it was retrieved.
8. If the context is insufficient, clearly state that.

ANNUAL REPORT CONTEXT:

{context}

QUESTION:

{question}
"""
)

# --------------------------------------------------
# 7. Main RAG chain
# --------------------------------------------------

chain = prompt | structured_llm

# --------------------------------------------------
# 8. Reranking function
# --------------------------------------------------

def rerank_documents(question, documents, top_k=5):

    candidates = []

    for i, doc in enumerate(documents):

        candidates.append(
            f"""
CHUNK_ID: {i}

PAGE:
{doc.metadata.get("page")}

CONTENT:
{doc.page_content[:1200]}
"""
        )

    candidates_text = "\n\n".join(candidates)

    rerank_prompt = f"""
You are ranking annual-report document chunks for an equity research question.

QUESTION:

{question}

Select the {top_k} most relevant chunks that directly help answer the question.

Rules:

- Prefer direct factual evidence.
- Prefer financial figures and explicit disclosures.
- Avoid unrelated company descriptions.
- Avoid generic information.
- Avoid generic disclaimers unless directly relevant.

Return ONLY chunk IDs separated by commas.

Example:

2,5,1,8,3

CANDIDATES:

{candidates_text}
"""

    response = llm.invoke(rerank_prompt)

    selected_ids = []

    for value in response.content.strip().split(","):

        try:
            selected_ids.append(int(value.strip()))

        except ValueError:
            pass


    reranked_documents = []

    for i in selected_ids:

        if 0 <= i < len(documents):

            reranked_documents.append(
                documents[i]
            )


    return reranked_documents


# --------------------------------------------------
# 9. Main equity research function
# --------------------------------------------------

def ask_equity_question(question):

    # ----------------------------------------------
    # A. Query expansion
    # ----------------------------------------------

    search_queries = generate_search_queries(question)

    # ----------------------------------------------
    # B. Retrieve candidate documents
    # ----------------------------------------------

    unique_documents = {}


    for search_query in search_queries:

        results = vector_store.similarity_search(
            search_query,
            k=4
        )


        for doc in results:

            key = (
                doc.metadata.get("source"),
                doc.metadata.get("page"),
                doc.metadata.get("start_index")
            )

            unique_documents[key] = doc


    documents = list(
        unique_documents.values()
    )


    # ----------------------------------------------
    # C. Rerank documents
    # ----------------------------------------------

    documents = rerank_documents(
        question,
        documents,
        top_k=5
    )


    # ----------------------------------------------
    # D. Build context
    # ----------------------------------------------

    context_parts = []


    for doc in documents:

        page = doc.metadata.get(
            "page",
            "Unknown"
        )

        context_parts.append(
            f"""
PAGE: {page}

{doc.page_content}
"""
        )


    context = "\n\n".join(
        context_parts
    )


    # ----------------------------------------------
    # E. Generate answer
    # ----------------------------------------------

    response = chain.invoke(
        {
            "context": context,
            "question": question
        }
    )
    answer = response.answer
    used_pages = response.used_pages

    # ----------------------------------------------
    # F. Prepare sources
    # ----------------------------------------------


    sources = []

    seen = set()

    for doc in documents:

        source = doc.metadata.get(
            "source",
            "Unknown"
        )

        page = doc.metadata.get(
            "page",
            "Unknown"
        )

    # Only show sources actually used by the LLM
        if page not in used_pages:
            continue

        key = (
            source,
            page
        )

        if key not in seen:

            sources.append({
                "source": source,
                "page": page,
                "content": doc.page_content
            })

            seen.add(key)


    # ----------------------------------------------
    # G. Return result
    # ----------------------------------------------

    return {
        "answer": answer,
        "sources": sources,
        "used_pages": used_pages
    }


def generate_search_queries(question):

    query_prompt = f"""
You are helping retrieve information from a company's annual report.

Create 3 different search queries that would help find
evidence needed to answer the user's question.

User question:
{question}

Rules:
- Keep each query concise.
- Use financial-report terminology where useful.
- Cover different ways the annual report may describe the same concept.
- Do not answer the question.
- Return exactly 3 queries.
- Put each query on a separate line.
"""

    response = llm.invoke(query_prompt)

    queries = [
        line.strip().lstrip("-1234567890. ")
        for line in response.content.splitlines()
        if line.strip()
    ]

    return [question] + queries[:3]