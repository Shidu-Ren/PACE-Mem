from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests


_CLIENT_CACHE: dict[tuple[str, str, str, int, float], "OpenAIEmbeddingClient"] = {}
_CLIENT_CACHE_LOCK = threading.Lock()


def _normalize_rows(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return array / norms


def _safe_name(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value or "unknown")).strip("_")
    return value[:120] or "unknown"


def _hash_texts(texts: list[str], extra: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(extra, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    for text in texts:
        digest.update(b"\0")
        digest.update(str(text or "").encode("utf-8"))
    return digest.hexdigest()


class OpenAIEmbeddingClient:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = "",
        batch_size: int = 64,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.base_url = str(base_url or "").rstrip("/")
        self.api_key = api_key or os.environ.get("EMBEDDING_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        self.batch_size = max(1, int(batch_size))
        self.timeout = float(timeout)
        self._memory_cache: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

    @property
    def embeddings_url(self) -> str:
        if not self.base_url:
            raise ValueError(
                "Dense retrieval requires --embedding_base_url, for example "
                "http://127.0.0.1:30012/v1"
            )
        if self.base_url.endswith("/embeddings"):
            return self.base_url
        return f"{self.base_url}/embeddings"

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        texts = [str(text or "") for text in texts]
        output: list[np.ndarray | None] = [None] * len(texts)
        missing_indices: list[int] = []
        missing_texts: list[str] = []

        with self._lock:
            for idx, text in enumerate(texts):
                cached = self._memory_cache.get(text)
                if cached is None:
                    missing_indices.append(idx)
                    missing_texts.append(text)
                else:
                    output[idx] = cached

        for start in range(0, len(missing_texts), self.batch_size):
            batch_texts = missing_texts[start : start + self.batch_size]
            batch_indices = missing_indices[start : start + self.batch_size]
            batch_embeddings = self._request_embeddings(batch_texts)
            with self._lock:
                for idx, text, embedding in zip(batch_indices, batch_texts, batch_embeddings):
                    self._memory_cache[text] = embedding
                    output[idx] = embedding

        return _normalize_rows(np.vstack(output))

    def _request_embeddings(self, texts: list[str]) -> np.ndarray:
        payload = {"model": self.model, "input": texts}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = requests.post(self.embeddings_url, headers=headers, json=payload, timeout=self.timeout)
                if response.status_code >= 400:
                    raise RuntimeError(f"HTTP {response.status_code}: {response.text[:1000]}")
                data = response.json()["data"]
                data = sorted(data, key=lambda row: row.get("index", 0))
                embeddings = np.asarray([row["embedding"] for row in data], dtype=np.float32)
                return _normalize_rows(embeddings)
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Embedding request failed after 3 attempts: {last_error}") from last_error


def get_embedding_client(args: Any) -> OpenAIEmbeddingClient:
    model = str(getattr(args, "embedding_model", "") or "Qwen/Qwen3-Embedding-4B")
    base_url = str(getattr(args, "embedding_base_url", "") or "")
    api_key = str(getattr(args, "embedding_api_key", "") or "")
    batch_size = int(getattr(args, "embedding_batch_size", 64) or 64)
    timeout = float(getattr(args, "embedding_timeout", 120.0) or 120.0)
    key = (model, base_url, api_key, batch_size, timeout)
    with _CLIENT_CACHE_LOCK:
        client = _CLIENT_CACHE.get(key)
        if client is None:
            client = OpenAIEmbeddingClient(model, base_url, api_key, batch_size, timeout)
            _CLIENT_CACHE[key] = client
        return client


def load_or_create_doc_embeddings(
    docs: list[dict[str, Any]],
    item: dict[str, Any],
    args: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    texts = [str(doc.get("text") or "") for doc in docs]
    client = get_embedding_client(args)
    cache_root = str(getattr(args, "embedding_cache_dir", "") or "")
    meta_extra = {
        "model": client.model,
        "memory": getattr(args, "selector_memory", ""),
        "clip_length": getattr(args, "clip_length", ""),
        "doc_count": len(texts),
    }
    text_hash = _hash_texts(texts, meta_extra)

    if not cache_root:
        return client.embed_texts(texts), {"cache": "memory_only", "text_hash": text_hash, "embedding_model": client.model}

    video_id = _safe_name(str(item.get("video_id") or item.get("id") or item.get("global_idx") or "unknown"))
    model_name = _safe_name(client.model)
    cache_dir = Path(cache_root) / model_name / _safe_name(str(getattr(args, "selector_memory", "memory")))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{video_id}_{text_hash[:16]}.npz"

    if cache_path.exists():
        loaded = np.load(cache_path, allow_pickle=False)
        embeddings = np.asarray(loaded["embeddings"], dtype=np.float32)
        if embeddings.shape[0] == len(texts):
            return _normalize_rows(embeddings), {
                "cache": "hit",
                "cache_path": str(cache_path),
                "text_hash": text_hash,
                "embedding_model": client.model,
            }

    embeddings = client.embed_texts(texts)
    tmp_path = cache_path.with_suffix(cache_path.suffix + f".tmp.{os.getpid()}")
    with tmp_path.open("wb") as f:
        np.savez_compressed(f, embeddings=embeddings)
    os.replace(tmp_path, cache_path)
    return embeddings, {
        "cache": "miss",
        "cache_path": str(cache_path),
        "text_hash": text_hash,
        "embedding_model": client.model,
    }
