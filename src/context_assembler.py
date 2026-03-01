from typing import List
from .models import ChatMessage, RetrievedMemory
from .config import RETRIEVAL_TOP_K


class ContextAssembler:
    def build(self, user_query: str, stm_messages: List[ChatMessage], retrieved: List[RetrievedMemory]) -> str:
        # Historical memories
        hist_lines = ["[HISTORICAL_MEMORIES]"]
        # ensure up to RETRIEVAL_TOP_K
        for rm in sorted(retrieved, key=lambda r: r.score, reverse=True)[:RETRIEVAL_TOP_K]:
            hist_lines.append(f"- (score={rm.score:.4f}) {rm.memory_id}: {rm.summary.narrative}")

        # Recent context
        recent_lines = ["", "[RECENT_CONTEXT]"]
        for m in stm_messages:
            ts = m.timestamp.isoformat()
            recent_lines.append(f"- [{ts}] {m.author}({m.role}): {m.content}")

        # User query
        user_lines = ["", "[USER_QUERY]", user_query]

        return "\n".join(hist_lines + recent_lines + user_lines)
