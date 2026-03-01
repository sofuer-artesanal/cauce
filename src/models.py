from __future__ import annotations
from pydantic import BaseModel, field_validator, Field
from typing import Literal
from datetime import datetime


class ChatMessage(BaseModel):
    id: str
    role: Literal["user", "assistant", "system"]
    author: str
    content: str
    timestamp: datetime


class CompressionInput(BaseModel):
    batch_id: str
    messages: list[ChatMessage]


class MemorySummary(BaseModel):
    batch_id: str
    from_ts: datetime
    to_ts: datetime
    participants: list[str]
    topics: list[str]
    facts: list[str]
    agreements: list[str]
    narrative: str

    @field_validator("facts")
    def validate_facts_len(cls, v):
        if len(v) > 8:
            raise ValueError("facts must have at most 8 items")
        return v

    @field_validator("agreements")
    def validate_agreements_len(cls, v):
        if len(v) > 8:
            raise ValueError("agreements must have at most 8 items")
        return v

    @field_validator("narrative")
    def validate_narrative_len(cls, v):
        if not (1 <= len(v) <= 1200):
            raise ValueError("narrative must be 1..1200 chars")
        return v

    @field_validator("participants")
    def uniq_sort_participants(cls, v: list[str]):
        unique = sorted(set(v))
        return unique

    @field_validator("topics")
    def lowercase_topics(cls, v: list[str]):
        return [t.lower() for t in v]

    @field_validator("to_ts")
    def check_ts_order(cls, to_ts, info):
        values = info.data
        from_ts = values.get("from_ts")
        if from_ts and from_ts > to_ts:
            raise ValueError("from_ts must be <= to_ts")
        return to_ts


class RetrievedMemory(BaseModel):
    memory_id: str
    score: float
    summary: MemorySummary
