import json
import pytest
from datetime import datetime, timezone
from src.compression_engine import CompressionEngine, CompressionValidationError
from src.models import ChatMessage, MemorySummary


def make_batch():
    base = datetime(2025,1,1,12,0,tzinfo=timezone.utc)
    return [ChatMessage(id="m1", role="user", author="Luis", content="hello", timestamp=base), ChatMessage(id="m2", role="user", author="Francis", content="ok", timestamp=base)]


def test_compression_success(monkeypatch):
    ce = CompressionEngine()
    batch = make_batch()

    def fake_llm(prompt):
        return json.dumps({
            "batch_id": "batch_1_2",
            "from_ts": batch[0].timestamp.isoformat(),
            "to_ts": batch[-1].timestamp.isoformat(),
            "participants": ["Luis","Francis"],
            "topics": ["meeting"],
            "facts": ["f1"],
            "agreements": ["a1"],
            "narrative": "short",
        })

    monkeypatch.setattr(ce, "_call_llm", fake_llm)
    summary = ce.compress(batch)
    assert isinstance(summary, MemorySummary)


def test_compression_retry_and_fail(monkeypatch):
    ce = CompressionEngine()
    batch = make_batch()
    calls = {"n":0}

    def bad_then_bad(prompt):
        calls["n"] += 1
        return "not a json"

    monkeypatch.setattr(ce, "_call_llm", bad_then_bad)
    with pytest.raises(CompressionValidationError):
        ce.compress(batch)
