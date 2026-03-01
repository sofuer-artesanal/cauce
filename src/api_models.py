from __future__ import annotations
from typing import Literal
from pydantic import BaseModel


class IngestResponse(BaseModel):
    status: Literal["ingested"]
    buffer_size: int
    memories: int
    rollover_triggered: bool
    compressed_batch_id: str | None = None
    memory_id: str | None = None


class RetrieveRequest(BaseModel):
    query: str


class RetrieveResponse(BaseModel):
    query: str
    context: str
    retrieved_count: int
    buffer_size: int


class HealthResponse(BaseModel):
    status: Literal["ok"]
    buffer_size: int
    memories: int
    compression_in_progress: bool
