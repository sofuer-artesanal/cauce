import logging
from datetime import datetime, timedelta, timezone
from .memory_engine import MemoryEngine
from .models import ChatMessage, MemorySummary

logging.basicConfig(format='%(asctime)s | %(levelname)s | %(name)s | %(message)s', level=logging.INFO)


def make_messages() -> list[ChatMessage]:
    base = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
    msgs = []
    authors = ["Luis", "Francis", "ExEx"]
    contents = [
        "Luis prefers meetings in the morning.",
        "Francis agreed on the budget of $5000.",
        "We decided the delivery date is 2025-02-15.",
        "A follow-up note about progress.",
        "Random chat content.",
    ]
    for i in range(1, 31):
        author = authors[i % len(authors)]
        content = contents[i % len(contents)]
        role = "user" if author != "ExEx" else "assistant"
        msg = ChatMessage(id=f"msg_{i:04d}", role=role, author=author, content=content, timestamp=base + timedelta(minutes=i))
        msgs.append(msg)
    return msgs


def run_demo():
    # Choose compression engine: local summarizer or an external LLM via OpenRouter.
    # If environment variable LLM_PROVIDER=openrouter and LLM_API_KEY/LLM_MODEL are set,
    # use the OpenRouterLLM implementation; otherwise default to local summarizer.
    try:
        from .llm import OpenRouterLLM
    except Exception:
        OpenRouterLLM = None

    def local_compress(batch):
        from datetime import timezone
        from collections import Counter

        from_ts = batch[0].timestamp
        to_ts = batch[-1].timestamp
        participants = sorted({m.author for m in batch})
        # simple heuristics
        topics = set()
        facts = []
        agreements = []
        narrative_lines = []
        for m in batch:
            t = m.content.lower()
            narrative_lines.append(f"[{m.author}] {m.content}")
            if "prefer" in t or "prefers" in t or "morning" in t:
                facts.append(m.content)
                topics.add("meeting")
            if "agreed" in t or "agree" in t or "budget" in t:
                agreements.append(m.content)
                topics.add("budget")
            if "decid" in t or "delivery" in t or "date" in t:
                facts.append(m.content)
                topics.add("delivery")

        facts = facts[:3]
        agreements = agreements[:3]
        topics = sorted(list(topics)) or ["general"]
        narrative = " ".join(narrative_lines)[:1200]
        batch_id = f"batch_{int(from_ts.timestamp())}_{int(to_ts.timestamp())}"
        return MemorySummary(batch_id=batch_id, from_ts=from_ts, to_ts=to_ts, participants=participants, topics=topics, facts=facts, agreements=agreements, narrative=narrative)

    # Build a compression engine with provider or use local compression
    compression_engine = None
    import os
    if os.getenv("LLM_PROVIDER", "").lower() == "openrouter" and OpenRouterLLM is not None:
        try:
            provider = OpenRouterLLM()
            from .compression_engine import CompressionEngine
            compression_engine = CompressionEngine(provider=provider)
        except Exception as e:
            logging.warning("Failed to initialize OpenRouterLLM: %s — falling back to local summarizer", e)

    if compression_engine is None:
        # create a compression engine object and override compress with local_compress
        from .compression_engine import CompressionEngine
        compression_engine = CompressionEngine()
        compression_engine.compress = local_compress

    me = MemoryEngine(compression_engine=compression_engine)
    messages = make_messages()
    for m in messages:
        try:
            me.ingest(m)
        except Exception as e:
            logging.error("Error ingesting: %s", e)

    print("STATS:", me.stats())

    queries = [
        "When did we decide the delivery date?",
        "What did Luis prefer for meeting time?",
        "What was agreed about budget?",
    ]
    for q in queries:
        ctx = me.build_context_for_query(q)
        print("\n--- Context for query ---\n")
        print(ctx)


if __name__ == "__main__":
    run_demo()
