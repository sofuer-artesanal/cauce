from __future__ import annotations
import threading
import logging
from typing import Optional

from .memory_engine import MemoryEngine
from .persistence import load_buffer, load_faiss_if_exists, load_runtime_state, save_buffer, save_faiss, save_runtime_state, ensure_data_dirs

logger = logging.getLogger(__name__)


class MemoryRuntime:
    def __init__(self) -> None:
        self.engine: Optional[MemoryEngine] = None
        self.ingest_lock = threading.RLock()
        self.compression_lock = threading.Lock()
        self.compression_in_progress = False
        self._ingest_count = 0

    def start(self) -> None:
        ensure_data_dirs()
        self.engine = MemoryEngine()
        # restore buffer
        msgs = load_buffer()
        if msgs:
            self.engine.stm.replace_all(msgs)
            logger.info("Restored %d messages into ShortTermBuffer", len(msgs))

        # restore faiss if exists
        try:
            load_faiss_if_exists(self.engine.vector)
        except Exception:
            logger.exception("Failed to load faiss on startup")

        # restore runtime state (next_memory_id)
        state = load_runtime_state()
        try:
            nid = int(state.get("next_memory_id", 0))
            if nid and getattr(self.engine.vector, "_next_id", None) is not None:
                self.engine.vector._next_id = nid
                logger.info("Restored next_memory_id=%d", nid)
        except Exception:
            logger.exception("Failed to restore next_memory_id")

    def shutdown(self) -> None:
        if not self.engine:
            return
        try:
            save_buffer(self.engine.stm.to_list())
        except Exception:
            logger.exception("Failed to save buffer on shutdown")
        try:
            save_faiss(self.engine.vector)
        except Exception:
            logger.exception("Failed to save faiss on shutdown")
        try:
            save_runtime_state(getattr(self.engine.vector, "_next_id", 1))
        except Exception:
            logger.exception("Failed to save runtime state on shutdown")

    def persist_after_ingest(self) -> None:
        if not self.engine:
            return
        try:
            save_buffer(self.engine.stm.to_list())
        except Exception:
            logger.exception("Failed to save buffer after ingest")
        try:
            save_runtime_state(getattr(self.engine.vector, "_next_id", 1))
        except Exception:
            logger.exception("Failed to save runtime state after ingest")


# module-level singleton
runtime = MemoryRuntime()
