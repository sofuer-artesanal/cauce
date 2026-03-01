from __future__ import annotations
import json
import os
import tempfile
from datetime import datetime
from typing import List
import logging

from .config import DATA_DIR, FAISS_DIR, BUFFER_STATE_PATH, RUNTIME_STATE_PATH
from .models import ChatMessage

logger = logging.getLogger(__name__)


def ensure_data_dirs() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FAISS_DIR, exist_ok=True)


def _atomic_write(path: str, data: str) -> None:
    dirpath = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=dirpath)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


def save_buffer(messages: List[ChatMessage]) -> None:
    ensure_data_dirs()
    payload = {"messages": [m.model_dump() for m in messages]}
    data = json.dumps(payload, ensure_ascii=False, indent=2)
    _atomic_write(BUFFER_STATE_PATH, data)
    logger.info("Saved buffer to %s", BUFFER_STATE_PATH)


def load_buffer() -> List[ChatMessage]:
    if not os.path.exists(BUFFER_STATE_PATH):
        return []
    try:
        with open(BUFFER_STATE_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
        out = []
        for item in payload.get("messages", []):
            try:
                out.append(ChatMessage.model_validate(item))
            except Exception:
                logger.exception("Failed to validate ChatMessage from persistence")
        return out
    except Exception:
        logger.exception("Failed to load buffer from %s", BUFFER_STATE_PATH)
        return []


def save_runtime_state(next_memory_id: int) -> None:
    ensure_data_dirs()
    payload = {"next_memory_id": int(next_memory_id), "last_persisted_at": datetime.utcnow().isoformat() + "Z"}
    data = json.dumps(payload, ensure_ascii=False, indent=2)
    _atomic_write(RUNTIME_STATE_PATH, data)
    logger.info("Saved runtime state to %s", RUNTIME_STATE_PATH)


def load_runtime_state() -> dict:
    if not os.path.exists(RUNTIME_STATE_PATH):
        return {}
    try:
        with open(RUNTIME_STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logger.exception("Failed to load runtime state from %s", RUNTIME_STATE_PATH)
        return {}


def save_faiss(store) -> None:
    ensure_data_dirs()
    try:
        store.save_local(FAISS_DIR)
        logger.info("Saved FAISS index to %s", FAISS_DIR)
    except Exception:
        logger.exception("Failed to save FAISS store to %s", FAISS_DIR)


def load_faiss_if_exists(store) -> None:
    # if FAISS_DIR doesn't exist or is empty, do nothing
    if not os.path.isdir(FAISS_DIR):
        return
    try:
        # Some FAISS wrappers expect files like index.faiss or similar
        has_files = any(os.path.isfile(os.path.join(FAISS_DIR, p)) for p in os.listdir(FAISS_DIR))
        if not has_files:
            return
        try:
            store.load_local(FAISS_DIR)
            logger.info("Loaded FAISS index from %s", FAISS_DIR)
        except TypeError:
            # older versions might expect embeddings param
            try:
                store.load_local(FAISS_DIR, None)
                logger.info("Loaded FAISS index (alt) from %s", FAISS_DIR)
            except Exception:
                logger.exception("Failed to load FAISS (alternate) from %s", FAISS_DIR)
    except Exception:
        logger.exception("Error while attempting to load FAISS from %s", FAISS_DIR)
