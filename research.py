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


# --------------------------------------------------
# 5. Rewrite conversational follow-up question
# --------------------------------------------------

def rewrite_follow_up_question(
    question: str,
    chat_history: list
) -> str:

    # If this is the first question,
    # no rewriting is required
    if not chat_history:
        return question


    # Use recent conversation only
    history_text = "\n".join(
        f"{message['role'].upper()}: {message['content']}"
        for message in chat_history[-6:]
    )


    rewrite_prompt = f"""
You are helping with financial research over company reports.

Convert the user's latest question into a standalone research question
using the conversation history.

Rules:

- Resolve conversational references such as:
  "that",
  "it",
  "this",
  "those results",
  "that growth",
  "the next quarter",
  "the previous year",
  "why did that happen",
  etc.

- Preserve the user's actual intent.

- Do not answer the question.

- Do not introduce facts that were not mentioned
  in the conversation.

- Return ONLY the rewritten standalone question.

CONVERSATION:

{history_text}

LATEST QUESTION:

{question}
"""


    response = llm.invoke(
        rewrite_prompt
    )


    return response.content.strip()


# --------------------------------------------------
# 6. Structured output models
# --------------------------------------------------

class UsedSource(BaseModel):

    report_id: str = Field(
        description=(
            "Exact REPORT_ID of the financial report "
            "used to support the answer"
        )
    )

    page: int = Field(
        description=(
            "Exact page number used to support the answer"
        )
    )


class ResearchAnswer(BaseModel):

    answer: str = Field(
        description=(
            "Final grounded equity research answer"
        )
    )

    used_sources: list[UsedSource] = Field(
        description=(
            "Exact report IDs and page numbers actually "
            "used to support the final answer"
        )
    )


# --------------------------------------------------
# 7. Structured LLM
# --------------------------------------------------

structured_llm = llm.with_structured_output(
    ResearchAnswer
)


# --------------------------------------------------
# 8. Final research prompt
# --------------------------------------------------

prompt = PromptTemplate(
    input_variables=[
        "context",
        "question"
    ],

    template="""
You are an equity research assistant analyzing company financial reports.

The supplied context may contain information from:

- Annual Reports
- Quarterly Reports
- Earnings Reports
- Financial Statements
- Other company financial disclosures

Answer the user's question using ONLY the supplied context.

RULES:

1. Do not use outside knowledge.

2. Do not invent:
   - facts
   - financial figures
   - percentages
   - dates
   - fiscal years
   - report IDs
   - page numbers

3. Include only information that directly helps answer the question.

4. Every important factual or financial claim must be supported
   by the supplied report context.

5. When useful, cite the report naturally using:

   (Report Type, FY Year, Page X)

   Example:

   (Annual Report, FY 2024, Page 32)

6. used_sources must contain ONLY sources that were actually
   used to support the final answer.

7. For every source in used_sources, return:

   - the EXACT REPORT_ID shown in the context
   - the EXACT PAGE number shown in the context

8. Do not include a source merely because it was retrieved.

9. Multiple financial reports may contain the same page number.
   Therefore, always distinguish evidence using:

   REPORT_ID + PAGE

10. When comparing multiple reports, clearly identify which
    information comes from which report.

11. Do not describe something as the "main", "primary",
    "largest", or "most important" driver unless the supplied
    context explicitly supports that characterization.

12. If the supplied context does not contain enough evidence
    to answer the question, clearly state that.

13. If only one report contains relevant evidence for a
    multi-report comparison question, state that the available
    context is insufficient for a complete comparison.

FINANCIAL REPORT CONTEXT:

{context}

QUESTION:

{question}
"""
)


# --------------------------------------------------
# 9. Main RAG chain
# --------------------------------------------------

chain = (
    prompt
    | structured_llm
)


# --------------------------------------------------
# 10. Generate retrieval queries
# --------------------------------------------------

def generate_search_queries(
    question: str
):

    query_prompt = f"""
You are generating search queries for retrieving evidence
from company financial reports.

USER QUESTION:

{question}

Generate exactly 3 concise search queries that could locate
the evidence required to answer the question.

Use terminology commonly found in financial reports when useful.

Examples of relevant terminology include:

- revenue
- operating income
- net income
- gross margin
- cloud revenue
- segment revenue
- earnings
- risk factors
- cash flow
- fiscal year
- quarterly results
- year-over-year growth

Rules:

- Do not answer the question.
- Do not invent facts.
- Return one search query per line.
"""


    response = llm.invoke(
        query_prompt
    )


    queries = []


    for line in response.content.splitlines():

        cleaned_line = line.strip()

        if not cleaned_line:
            continue

        # Remove common numbered/bullet prefixes
        cleaned_line = cleaned_line.lstrip(
            "-•1234567890. "
        )

        if cleaned_line:
            queries.append(
                cleaned_line
            )


    # Always include the original standalone question
    return [
        question
    ] + queries[:3]


# --------------------------------------------------
# 11. Rerank retrieved documents
# --------------------------------------------------

def rerank_documents(
    question,
    documents,
    top_k=5
):

    # Nothing to rerank
    if not documents:
        return []


    candidates = []


    for i, doc in enumerate(
        documents
    ):

        candidates.append(
            f"""
CHUNK_ID: {i}

REPORT_ID:
{doc.metadata.get("report_id")}

REPORT_TYPE:
{doc.metadata.get("report_type")}

FISCAL_YEAR:
{doc.metadata.get("fiscal_year")}

SOURCE:
{doc.metadata.get("source")}

PAGE:
{doc.metadata.get("page")}

CONTENT:

{doc.page_content[:1200]}
"""
        )


    candidates_text = "\n\n".join(
        candidates
    )


    rerank_prompt = f"""
You are ranking financial-report document chunks
for an equity research question.

QUESTION:

{question}

Select up to {top_k} chunks that most directly help
answer the question.

RULES:

- Prefer direct factual evidence.

- Prefer explicit financial disclosures.

- Prefer financial figures when the question
  is asking about financial performance.

- If the question compares periods or reports,
  select evidence from multiple relevant reports
  when available.

- Consider REPORT_TYPE and FISCAL_YEAR.

- Avoid unrelated company descriptions.

- Avoid generic information.

- Avoid generic disclaimers unless they are
  directly relevant to the question.

Return ONLY the selected CHUNK_ID values
separated by commas.

Example:

2,5,1,8,3

CANDIDATES:

{candidates_text}
"""


    response = llm.invoke(
        rerank_prompt
    )


    selected_ids = []


    for value in (
        response.content
        .strip()
        .split(",")
    ):

        try:

            chunk_id = int(
                value.strip()
            )

            if chunk_id not in selected_ids:

                selected_ids.append(
                    chunk_id
                )

        except ValueError:

            pass


    reranked_documents = []


    for i in selected_ids:

        if 0 <= i < len(documents):

            reranked_documents.append(
                documents[i]
            )


    # Fallback in case LLM returns unusable IDs
    if not reranked_documents:

        return documents[:top_k]


    return reranked_documents[:top_k]


# --------------------------------------------------
# 12. Main equity research function
# --------------------------------------------------

def ask_equity_question(
    question,
    index_dir,
    chat_history=None,
    report_ids=None
):

    # ----------------------------------------------
    # A. Initialize chat history
    # ----------------------------------------------

    if chat_history is None:

        chat_history = []


    # ----------------------------------------------
    # B. Rewrite follow-up question
    # ----------------------------------------------

    standalone_question = (
        rewrite_follow_up_question(
            question,
            chat_history
        )
    )


    # ----------------------------------------------
    # C. Connect to vector database
    # ----------------------------------------------

    vector_store = get_vector_store(
        index_dir
    )


    # ----------------------------------------------
    # D. Query expansion
    # ----------------------------------------------

    search_queries = (
        generate_search_queries(
            standalone_question
        )
    )


    # ----------------------------------------------
    # E. Retrieve candidate documents
    # ----------------------------------------------

    unique_documents = {}


    for search_query in search_queries:

    # --------------------------------------------------
    # Compare mode:
    # retrieve evidence separately from each
    # selected report
    # --------------------------------------------------

        if report_ids:

            for report_id in report_ids:

                results = vector_store.similarity_search(
                    search_query,
                    k=3,
                    filter={
                        "report_id": report_id
                    }
                )


                for doc in results:

                    key = (
                        doc.metadata.get("report_id"),
                        doc.metadata.get("page"),
                        doc.metadata.get("start_index")
                    )

                    unique_documents[key] = doc


    # --------------------------------------------------
    # Normal Ask Question mode:
    # search all reports
    # --------------------------------------------------

        else:

            results = vector_store.similarity_search(
                search_query,
                k=4
            )   


            for doc in results:

                key = (
                    doc.metadata.get("report_id"),
                    doc.metadata.get("page"),
                    doc.metadata.get("start_index")
                )

                unique_documents[key] = doc


    documents = list(
        unique_documents.values()
    )


    # ----------------------------------------------
    # F. Handle no retrieval results
    # ----------------------------------------------

    if not documents:

        return {
            "answer": (
                "I could not find relevant evidence "
                "in the indexed financial reports."
            ),

            "sources": [],

            "used_sources": []
        }


    # ----------------------------------------------
    # G. Rerank retrieved documents
    # ----------------------------------------------

    rerank_top_k = (
        6
        if report_ids
        else 5
    )


    documents = rerank_documents(
        standalone_question,
        documents,
        top_k=rerank_top_k
    )


    # ----------------------------------------------
    # H. Build financial-report context
    # ----------------------------------------------

    context_parts = []


    for doc in documents:

        context_parts.append(
            f"""
REPORT_ID:
{doc.metadata.get("report_id")}

REPORT_TYPE:
{doc.metadata.get("report_type")}

FISCAL_YEAR:
{doc.metadata.get("fiscal_year")}

SOURCE:
{doc.metadata.get("source")}

PAGE:
{doc.metadata.get("page")}

CONTENT:

{doc.page_content}
"""
        )


    context = "\n\n".join(
        context_parts
    )


    # ----------------------------------------------
    # I. Generate final answer
    # ----------------------------------------------

    response = chain.invoke(
        {
            "context": context,
            "question": standalone_question
        }
    )


    answer = response.answer


    # ----------------------------------------------
    # J. Get used REPORT_ID + PAGE pairs
    # ----------------------------------------------

    used_source_keys = {
        (
            source.report_id,
            source.page
        )

        for source in response.used_sources
    }


    # ----------------------------------------------
    # K. Prepare supporting source excerpts
    # ----------------------------------------------

    sources = []

    seen_sources = set()


    for doc in documents:

        report_id = doc.metadata.get(
            "report_id"
        )

        page = doc.metadata.get(
            "page"
        )


        source_key = (
            report_id,
            page
        )


        # ------------------------------------------
        # Only show sources actually used
        # ------------------------------------------

        if source_key not in used_source_keys:

            continue


        # ------------------------------------------
        # Avoid duplicate source boxes
        # ------------------------------------------

        if source_key in seen_sources:

            continue


        sources.append(
            {
                "report_id": report_id,

                "source": doc.metadata.get(
                    "source",
                    "Unknown"
                ),

                "page": page,

                "report_type": doc.metadata.get(
                    "report_type",
                    "Unknown"
                ),

                "fiscal_year": doc.metadata.get(
                    "fiscal_year",
                    "Unknown"
                ),

                "content": doc.page_content
            }
        )


        seen_sources.add(
            source_key
        )


    # ----------------------------------------------
    # L. Optional debugging
    # ----------------------------------------------

    print(
        "Original question:",
        question
    )

    print(
        "Standalone question:",
        standalone_question
    )

    print(
        "Used sources:",
        response.used_sources
    )

    print(
        "Returned sources:",
        [
            (
                source["report_type"],
                source["fiscal_year"],
                source["page"]
            )
            for source in sources
        ]
    )


    # ----------------------------------------------
    # M. Return final result
    # ----------------------------------------------

    return {
        "answer": answer,

        "sources": sources,

        "used_sources": [
            {
                "report_id": source.report_id,
                "page": source.page
            }

            for source in response.used_sources
        ]
    }