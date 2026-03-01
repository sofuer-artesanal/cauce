from __future__ import annotations
import os
from dotenv import load_dotenv
from typing import Final
load_dotenv()

MAX_WINDOW_SIZE: Final[int] = 20
COMPRESSION_BATCH_SIZE: Final[int] = 10
RETRIEVAL_TOP_K: Final[int] = 3
MAX_FACTS: Final[int] = 8
MAX_AGREEMENTS: Final[int] = 8
NARRATIVE_MAX_CHARS: Final[int] = 1200
TIMEZONE: Final[str] = "UTC"

# Env vars
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "")

# Persistence / runtime
DATA_DIR: str = os.getenv("DATA_DIR", "./data")
FAISS_DIR: str = os.getenv("FAISS_DIR", f"{DATA_DIR}/faiss")
BUFFER_STATE_PATH: str = os.getenv("BUFFER_STATE_PATH", f"{DATA_DIR}/short_term_buffer.json")
RUNTIME_STATE_PATH: str = os.getenv("RUNTIME_STATE_PATH", f"{DATA_DIR}/runtime_state.json")

AUTO_PERSIST_ON_INGEST: bool = True
PERSIST_EVERY_N_INGESTS: int = int(os.getenv("PERSIST_EVERY_N_INGESTS", "1"))


def validate_persistence_config() -> None:
    if PERSIST_EVERY_N_INGESTS < 1:
        raise ValueError("PERSIST_EVERY_N_INGESTS must be >= 1")


validate_persistence_config()

def validate_config() -> None:
    if MAX_WINDOW_SIZE <= 0:
        raise ValueError("MAX_WINDOW_SIZE must be > 0")
    if not (0 < COMPRESSION_BATCH_SIZE <= MAX_WINDOW_SIZE):
        raise ValueError("COMPRESSION_BATCH_SIZE must be >0 and <= MAX_WINDOW_SIZE")
    if RETRIEVAL_TOP_K < 1:
        raise ValueError("RETRIEVAL_TOP_K must be >= 1")
    # Note: env vars are required at runtime by the LLM/embeddings code paths.
    # If you need strict enforcement, call `ensure_env()` where appropriate.


def ensure_env() -> None:
    if not LLM_PROVIDER or not LLM_MODEL or not EMBEDDING_MODEL:
        raise EnvironmentError("Missing required LLM or embedding environment variables. See .env.example")


validate_config()

