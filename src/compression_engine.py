from __future__ import annotations
import json
import logging
import re
from typing import List
from datetime import timezone
from .models import ChatMessage, MemorySummary
from .prompts import COMPRESSION_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


class CompressionValidationError(Exception):
    pass


from .llm import LLMProvider


class CompressionEngine:
    def __init__(self, provider: LLMProvider | None = None) -> None:
        # provider is an injectable LLM implementation; if None, _call_llm must be monkeypatched in tests or replaced
        self.provider = provider

    def _serialize_batch(self, batch: List[ChatMessage]) -> str:
        # compact JSON
        def msg_to_dict(m: ChatMessage):
            return {
                "id": m.id,
                "role": m.role,
                "author": m.author,
                "content": m.content,
                "timestamp": m.timestamp.replace(tzinfo=timezone.utc).isoformat(),
            }

        return json.dumps([msg_to_dict(m) for m in batch], separators=(',', ':'))

    def _call_llm(self, prompt: str) -> str:
        # Use injected provider if available
        if self.provider is not None:
            return self.provider.call(prompt)
        # Hook for tests to monkeypatch. Default: raise until real provider is wired.
        raise RuntimeError("LLM provider not configured. Provide an LLMProvider or patch _call_llm in tests.")

    def _parse_llm_json(self, raw_response) -> dict:
        text = "" if raw_response is None else str(raw_response)
        stripped = text.strip()
        if not stripped:
            raise ValueError("Empty LLM response")

        candidates: list[str] = [stripped]

        # If wrapped in markdown code fences, extract inner content.
        fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL | re.IGNORECASE)
        if fence_match:
            candidates.append(fence_match.group(1).strip())

        # Greedy object slice from first { to last } as an additional candidate.
        first_obj = stripped.find("{")
        last_obj = stripped.rfind("}")
        if first_obj != -1 and last_obj != -1 and last_obj > first_obj:
            candidates.append(stripped[first_obj:last_obj + 1])

        for candidate in candidates:
            if not candidate:
                continue
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

        # Final fallback: find first decodable JSON object inside mixed text.
        decoder = json.JSONDecoder()
        for idx, char in enumerate(stripped):
            if char != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(stripped[idx:])
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue

        raise ValueError("No valid JSON object found in LLM response")

    def compress(self, batch: List[ChatMessage]) -> MemorySummary:
        if not batch:
            raise ValueError("Empty batch")
        from_ts = batch[0].timestamp
        to_ts = batch[-1].timestamp
        batch_id = f"batch_{int(from_ts.timestamp())}_{int(to_ts.timestamp())}"
        serialized = self._serialize_batch(batch)
        prompt = COMPRESSION_PROMPT_TEMPLATE.replace("{serialized_batch}", serialized)

        # attempt 1
        try:
            resp = self._call_llm(prompt)
            data = self._parse_llm_json(resp)
            summary = MemorySummary(**self._normalize_summary(data))
            return summary
        except Exception as e:
            logger.warning("Compression failed on first attempt: %s", e)
            # retry once
            try:
                logger.warning("Retrying compression asking for strict JSON")
                prompt2 = prompt + "\nRespond with strict valid JSON."
                resp2 = self._call_llm(prompt2)
                data2 = self._parse_llm_json(resp2)
                summary2 = MemorySummary(**self._normalize_summary(data2))
                return summary2
            except Exception as e2:
                logger.error("Compression failed after retry: %s", e2)
                raise CompressionValidationError("Compression failed: invalid JSON or schema")

    def _normalize_summary(self, data: dict) -> dict:
        # Ensure keys exist and types coerced minimally
        return {
            "batch_id": data.get("batch_id", ""),
            "from_ts": data.get("from_ts"),
            "to_ts": data.get("to_ts"),
            "participants": data.get("participants", []),
            "topics": data.get("topics", []),
            "facts": data.get("facts", []),
            "agreements": data.get("agreements", []),
            "narrative": data.get("narrative", ""),
        }
