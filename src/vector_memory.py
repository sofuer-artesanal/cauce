from __future__ import annotations

import hashlib
import json
from typing import List, Dict, Optional

from langchain_community.vectorstores import FAISS

from .models import MemorySummary, RetrievedMemory


class _LocalEmbeddings:
    """Lightweight deterministic local embeddings for tests/PoC.

    Produces fixed-size vectors by hashing tokens into a small dense vector.
    """

    def __init__(self, dimension: int = 128):
        self.dimension = dimension

    def _embed_text(self, text: str) -> List[float]:
        vec = [0.0] * self.dimension
        for tok in text.lower().split():
            tok = ''.join(ch for ch in tok if ch.isalnum())
            if not tok:
                continue
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dimension
            vec[idx] += 1.0
        # normalize
        norm = sum(x * x for x in vec) ** 0.5
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def __call__(self, text: str) -> List[float]:
        return self._embed_text(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_text(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_text(text)


class VectorMemoryStore:
    def __init__(self, embeddings: Optional[object] = None) -> None:
        self._embeddings = embeddings or _LocalEmbeddings()
        self._store: Optional[FAISS] = None
        self._next_id = 1
        self._count = 0

    def add(self, summary: MemorySummary) -> str:
        memory_id = f"mem_{self._next_id}"
        self._next_id += 1

        metadata = {
            "memory_id": memory_id,
            "batch_id": summary.batch_id,
            "from_ts": summary.from_ts.isoformat(),
            "to_ts": summary.to_ts.isoformat(),
            "participants": ",".join(summary.participants),
            "topics": ",".join(summary.topics),
            "facts": summary.facts,
            "agreements": summary.agreements,
            # store the pydantic-serializable dict so we can reconstruct
            "summary_obj": summary.model_dump(),
        }

        text = summary.narrative

        if self._store is None:
            # initialize FAISS with the first document
            self._store = FAISS.from_texts([text], self._embeddings, metadatas=[metadata], ids=[memory_id])
            self._count = 1
        else:
            # add_texts will accept metadatas and ids
            try:
                self._store.add_texts([text], metadatas=[metadata], ids=[memory_id])
            except TypeError:
                # older langchain API may expect add_documents; craft a duck-typed doc
                class _Doc:
                    def __init__(self, page_content, metadata):
                        self.page_content = page_content
                        self.metadata = metadata

                doc = _Doc(page_content=text, metadata=metadata)
                self._store.add_documents([doc], ids=[memory_id])
            self._count += 1

        return memory_id

    def search(self, query: str, top_k: int) -> List[RetrievedMemory]:
        if self._store is None:
            return []

        # prefer methods that return scores if available
        docs_and_scores = None
        if hasattr(self._store, "similarity_search_with_score"):
            docs_and_scores = self._store.similarity_search_with_score(query, k=top_k)
        elif hasattr(self._store, "similarity_search_with_relevance_scores"):
            docs_and_scores = self._store.similarity_search_with_relevance_scores(query, k=top_k)
        else:
            docs = self._store.similarity_search(query, k=top_k)
            docs_and_scores = [(d, 0.0) for d in docs]

        results: List[RetrievedMemory] = []
        for item in docs_and_scores:
            # item may be (Document, score) or (Document, float)
            if isinstance(item, tuple) and len(item) == 2:
                doc, raw_score = item
            else:
                doc = item
                raw_score = 0.0

            # convert raw_score (distance) to similarity-like score if it seems to be a distance
            try:
                score = float(raw_score)
                if score > 1.0:
                    score = 1.0 / (1.0 + score)
            except Exception:
                score = 0.0

            md = doc.metadata or {}
            summary_data = md.get("summary_obj") or {}
            try:
                summary = MemorySummary.model_validate(summary_data)
            except Exception:
                # attempt to reconstruct from fields if validation fails
                summary = MemorySummary(
                    batch_id=md.get("batch_id", ""),
                    from_ts=md.get("from_ts"),
                    to_ts=md.get("to_ts"),
                    participants=md.get("participants", "").split(",") if md.get("participants") else [],
                    topics=md.get("topics", "").split(",") if md.get("topics") else [],
                    facts=md.get("facts", []),
                    agreements=md.get("agreements", []),
                    narrative=doc.page_content,
                )

            results.append(RetrievedMemory(memory_id=md.get("memory_id", ""), score=score, summary=summary))

        return results

    def count(self) -> int:
        return int(self._count)

    def save_local(self, folder_path: str) -> None:
        if self._store is None:
            raise ValueError("No FAISS store to save")
        # FAISS wrapper exposes save_local
        self._store.save_local(folder_path)

    def load_local(self, folder_path: str, embeddings: Optional[object] = None) -> None:
        emb = embeddings or self._embeddings or _LocalEmbeddings()
        try:
            self._store = FAISS.load_local(folder_path, emb)
        except TypeError:
            # older API
            try:
                self._store = FAISS.load_local(folder_path)
            except Exception:
                self._store = None
        except Exception:
            self._store = None

        # attempt to set count and next id
        ntotal = 0
        try:
            if self._store is not None:
                ntotal = int(getattr(self._store, "index").ntotal)
        except Exception:
            try:
                ntotal = len(getattr(self._store, "docstore")._dict)
            except Exception:
                ntotal = 0

        self._count = ntotal
        self._next_id = ntotal + 1

    def is_initialized(self) -> bool:
        return self._store is not None

    def next_id(self) -> int:
        return int(self._next_id)
