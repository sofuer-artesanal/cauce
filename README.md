# Cauce

Pronunciation (IPA): */ˈkaw.se/*

Cauce is a lightweight conversational memory engine for assistants and chat-based systems.
It ingests messages as they arrive, keeps a short-term working window for immediate context, compresses older exchanges into structured summaries, and retrieves relevant long-term memories for new queries.
The goal is to provide a simple base that is easy to run locally and easy to adapt as your memory strategy evolves.

## Usage examples

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Run the API

```bash
uvicorn src.api:app --reload
```

Then try:

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "id": "msg_001",
    "role": "user",
    "author": "Luis",
    "content": "Let\'s schedule the review for Monday morning.",
    "timestamp": "2026-03-01T10:00:00Z"
  }'
```

```bash
curl -X POST http://127.0.0.1:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "When did we schedule the review?"}'
```

### 3) Run the local simulator

```bash
python -m src.simulator
```

## Extendability

This project is designed to be modular:

- **Persistence**: swap or adapt functions in `src/persistence.py` to store data in any backend (files, DB, cloud, etc.).
- **Embeddings / vector store**: inject your own embeddings into `VectorMemoryStore` in `src/vector_memory.py`.
- **Compression / summarization**: replace the compression provider (`src/compression_engine.py`, `src/llm.py`) with the model or API you prefer.

In short, Cauce gives you the memory workflow and API surface, while letting you plug in the storage, embedding, and LLM stack that fits your needs.
