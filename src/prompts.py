COMPRESSION_PROMPT_TEMPLATE = """
You are a memory compression engine for conversations.
You will receive a batch of messages with author, role, timestamp, and content.
Return EXCLUSIVELY valid JSON, with no markdown and no additional text.

Goal:
1) Extract important verifiable facts.
2) Extract explicit agreements/decisions.
3) Write a brief narrative summary useful for semantic retrieval.

Strict rules:
- Do not invent information.
- If something is not explicit, do not include it as a fact.
- facts: maximum 8 items.
- agreements: maximum 8 items.
- narrative: maximum 1200 characters.
- topics: short lowercase labels.
- participants: unique names.

Required JSON schema:
{
  "batch_id": "string",
  "from_ts": "ISO-8601",
  "to_ts": "ISO-8601",
  "participants": ["string"],
  "topics": ["string"],
  "facts": ["string"],
  "agreements": ["string"],
  "narrative": "string"
}

Batch:
{serialized_batch}
"""
