from __future__ import annotations
import logging

from fastapi import FastAPI, HTTPException

from .api_models import IngestResponse, RetrieveRequest, RetrieveResponse, HealthResponse
from .models import ChatMessage
from .memory_runtime import runtime
from .config import AUTO_PERSIST_ON_INGEST, PERSIST_EVERY_N_INGESTS, MAX_WINDOW_SIZE

logger = logging.getLogger(__name__)

app = FastAPI()


@app.on_event("startup")
def on_startup() -> None:
    runtime.start()


@app.on_event("shutdown")
def on_shutdown() -> None:
    runtime.shutdown()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    eng = runtime.engine
    if eng is None:
        raise HTTPException(status_code=500, detail="Engine not initialized")
    s = eng.stats()
    return HealthResponse(status="ok", buffer_size=s["buffer_size"], memories=s["memories"], compression_in_progress=runtime.compression_in_progress)


@app.post("/ingest", response_model=IngestResponse)
def ingest(message: ChatMessage) -> IngestResponse:
    eng = runtime.engine
    if eng is None:
        raise HTTPException(status_code=500, detail="Engine not initialized")

    try:
        # acquire ingest lock for all mutations
        with runtime.ingest_lock:
            rollover = False
            compressed_batch_id = None
            memory_id = None

            will_rollover = (eng.stm.size() + 1) >= MAX_WINDOW_SIZE
            if will_rollover:
                # acquire compression lock while holding ingest_lock
                runtime.compression_lock.acquire()
                try:
                    runtime.compression_in_progress = True
                    res = eng.ingest_with_result(message)
                    if isinstance(res, dict):
                        rollover = res.get("rollover_triggered", False)
                        compressed_batch_id = res.get("compressed_batch_id")
                        memory_id = res.get("memory_id")
                finally:
                    runtime.compression_in_progress = False
                    runtime.compression_lock.release()
            else:
                res = eng.ingest_with_result(message)
                if isinstance(res, dict):
                    rollover = res.get("rollover_triggered", False)
                    compressed_batch_id = res.get("compressed_batch_id")
                    memory_id = res.get("memory_id")

            # persist after ingest per config
            if AUTO_PERSIST_ON_INGEST:
                runtime.persist_after_ingest()

            stats = eng.stats()
            return IngestResponse(status="ingested", buffer_size=stats["buffer_size"], memories=stats["memories"], rollover_triggered=rollover, compressed_batch_id=compressed_batch_id, memory_id=memory_id)

    except Exception as e:
        logger.exception("Error during ingest")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest) -> RetrieveResponse:
    eng = runtime.engine
    if eng is None:
        raise HTTPException(status_code=500, detail="Engine not initialized")
    try:
        context = eng.build_context_for_query(req.query)
        stats = eng.stats()
        retrieved = 0
        # ContextAssembler doesn't return count; keep simple
        return RetrieveResponse(query=req.query, context=context, retrieved_count=retrieved, buffer_size=stats["buffer_size"])
    except Exception as e:
        logger.exception("Error during retrieve")
        raise HTTPException(status_code=500, detail=str(e))
