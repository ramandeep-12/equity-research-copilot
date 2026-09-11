"""Configuration and lazy API clients; importing the app never requires a key."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def data_root():
    return Path(os.getenv("EQUITY_DATA_DIR", "chroma_db"))


def get_llm():
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
                      temperature=0, timeout=90, max_retries=2)


def get_embeddings():
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model="text-embedding-3-small", request_timeout=90, max_retries=2)
