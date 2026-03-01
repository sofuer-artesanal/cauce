import pytest
from datetime import datetime, timezone, timedelta
from src.memory_engine import MemoryEngine
from src.models import ChatMessage


class DummyCompression:
    def __init__(self, engine):
        self.engine = engine

    def compress(self, batch):
        # return a minimal valid MemorySummary-like dict via engine's compression._normalize_summary
        first = batch[0].timestamp
        last = batch[-1].timestamp
        data = {
            "batch_id": f"batch_{int(first.timestamp())}_{int(last.timestamp())}",
            "from_ts": first.isoformat(),
            "to_ts": last.isoformat(),
            "participants": [m.author for m in batch],
            "topics": ["test"],
            "facts": ["fact1"],
            "agreements": ["agree1"],
            "narrative": "A short narrative.",
        }
        # Use underlying compression engine normalization + pydantic validation
        return self.engine.compression._normalize_summary(data) if hasattr(self.engine.compression, '_normalize_summary') else data


def test_buffer_rollover(monkeypatch):
    me = MemoryEngine()
    # monkeypatch compression engine to return a MemorySummary object
    class C:
        def compress(self, batch):
            from src.models import MemorySummary
            first = batch[0].timestamp
            last = batch[-1].timestamp
            return MemorySummary(batch_id=f"batch_{int(first.timestamp())}_{int(last.timestamp())}", from_ts=first, to_ts=last, participants=[m.author for m in batch], topics=["t"], facts=["f"], agreements=["a"], narrative="narr")

    me.compression = C()

    base = datetime(2025,1,1,12,0,tzinfo=timezone.utc)
    # ingest 20 messages
    for i in range(20):
        msg = ChatMessage(id=f"msg_{i:04d}", role="user", author="Luis" if i%2==0 else "Francis", content="hello", timestamp=base + timedelta(minutes=i))
        me.ingest(msg)

    assert me.vector.count() == 1
    assert me.stm.size() == 10
