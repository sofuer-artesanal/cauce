import logging
from .short_term_buffer import ShortTermBuffer
from .compression_engine import CompressionEngine, CompressionValidationError
from .vector_memory import VectorMemoryStore
from .context_assembler import ContextAssembler
from .models import ChatMessage
from .config import MAX_WINDOW_SIZE, COMPRESSION_BATCH_SIZE, RETRIEVAL_TOP_K

logger = logging.getLogger(__name__)


class MemoryEngine:
    def __init__(self, compression_engine: CompressionEngine | None = None) -> None:
        self.stm = ShortTermBuffer()
        # allow injection of a compression engine (so we don't hardcode providers)
        self.compression = compression_engine if compression_engine is not None else CompressionEngine()
        self.vector = VectorMemoryStore()
        self.assembler = ContextAssembler()

    def ingest(self, message: ChatMessage) -> None:
        # maintain backward-compatible ingest by delegating to ingest_with_result
        self.ingest_with_result(message)

    def ingest_with_result(self, message: ChatMessage) -> dict:
        logger.info("Ingesting message %s", message.id)
        rollover_triggered = False
        compressed_batch_id = None
        memory_id = None

        self.stm.append(message)
        if self.stm.size() >= MAX_WINDOW_SIZE:
            rollover_triggered = True
            logger.info("Buffer reached MAX_WINDOW_SIZE (%d). Triggering compression.", MAX_WINDOW_SIZE)
            batch = self.stm.pop_oldest_batch(COMPRESSION_BATCH_SIZE)
            try:
                summary = self.compression.compress(batch)
            except CompressionValidationError:
                logger.error("Compression failed for batch")
                raise
            memory_id = self.vector.add(summary)
            compressed_batch_id = summary.batch_id
            logger.info("Added memory %s", memory_id)

        return {
            "rollover_triggered": rollover_triggered,
            "compressed_batch_id": compressed_batch_id,
            "memory_id": memory_id,
            "buffer_size": self.stm.size(),
            "memories": self.vector.count(),
        }

    def build_context_for_query(self, user_query: str) -> str:
        retrieved = self.vector.search(user_query, RETRIEVAL_TOP_K)
        recent = self.stm.snapshot()
        logger.info("Built context for query (retrieved=%d, recent=%d)", len(retrieved), len(recent))
        return self.assembler.build(user_query, recent, retrieved)

    def stats(self) -> dict:
        return {
            "buffer_size": self.stm.size(),
            "memories": self.vector.count(),
        }
