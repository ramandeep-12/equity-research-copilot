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

def get_vector_store(index_dir):

    return Chroma(
        collection_name="equity_research",
        embedding_function=embeddings,
        persist_directory=index_dir
    )


# --------------------------------------------------
# 4. LLM
# --------------------------------------------------

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)

def rewrite_follow_up_question(
    question: str,
    chat_history: list
) -> str:

    if not chat_history:
        return question

    history_text = "\n".join(
        f"{message['role'].upper()}: {message['content']}"
        for message in chat_history[-6:]
    )

    prompt = f"""
You are helping with financial research over company reports.

Convert the user's latest question into a standalone research question
using the conversation history.

Rules:
- Resolve references such as "that", "it", "this", "the next quarter",
  "those results", etc.
- Preserve the user's actual intent.
- Do not answer the question.
- Return ONLY the rewritten standalone question.

Conversation:
{history_text}

Latest question:
{question}
"""

    response = llm.invoke(prompt)

    return response.content.strip()

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

def ask_equity_question(question, index_dir, chat_history=None):
    if chat_history is None:
        chat_history = []


    standalone_question = rewrite_follow_up_question(
        question,
        chat_history
    )
    vector_store = get_vector_store(
        index_dir
    )

    # ----------------------------------------------
    # A. Query expansion
    # ----------------------------------------------

    search_queries = generate_search_queries(standalone_question)

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
        standalone_question,
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
            "question": standalone_question
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
You are generating search queries for information retrieval
from a company's annual report.

User question:
{question}

Generate exactly 3 concise search queries that could locate
the evidence required to answer the question.

Use terminology commonly found in annual reports when useful.

Do not answer the question.

Return one query per line.
"""

    response = llm.invoke(query_prompt)

    queries = [
        line.strip().lstrip("-1234567890. ")
        for line in response.content.splitlines()
        if line.strip()
    ]

    return [question] + queries[:3]