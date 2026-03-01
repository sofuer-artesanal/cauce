from datetime import datetime, timezone
from src.vector_memory import VectorMemoryStore
from src.models import MemorySummary, ChatMessage
from src.context_assembler import ContextAssembler


def test_hybrid_retrieval_and_assembler():
    vm = VectorMemoryStore()
    # create two summaries
    s1 = MemorySummary(batch_id="b1", from_ts=datetime(2025,1,1,12,0,tzinfo=timezone.utc), to_ts=datetime(2025,1,1,13,0,tzinfo=timezone.utc), participants=["Luis"], topics=["meeting"], facts=["prefers morning"], agreements=["budget ok"], narrative="Luis prefers meetings in the morning.")
    s2 = MemorySummary(batch_id="b2", from_ts=datetime(2025,1,2,12,0,tzinfo=timezone.utc), to_ts=datetime(2025,1,2,13,0,tzinfo=timezone.utc), participants=["Francis"], topics=["budget"], facts=["agreed budget"], agreements=["agree"], narrative="Francis agreed on the budget of $5000.")
    vm.add(s1)
    vm.add(s2)

    # query for budget
    retrieved = vm.search("budget", top_k=3)
    assert any("budget" in m.summary.narrative.lower() for m in retrieved)

    # assembler should include historical and recent
    ca = ContextAssembler()
    recent = [ChatMessage(id="m3", role="user", author="ExEx", content="recent note", timestamp=datetime(2025,1,3,12,0,tzinfo=timezone.utc))]
    ctx = ca.build("What about the budget?", recent, retrieved)
    assert "[HISTORICAL_MEMORIES]" in ctx
    assert "[RECENT_CONTEXT]" in ctx
    assert "[USER_QUERY]" in ctx
