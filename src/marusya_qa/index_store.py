from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import json

import numpy as np

from marusya_qa.data import Passage


def require_faiss():
    try:
        import faiss  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Установите faiss-cpu: pip install faiss-cpu") from exc
    return faiss


class FaissPassageIndex:
    def __init__(self, index: object, passages: list[Passage]) -> None:
        self.index = index
        self.passages = passages

    @classmethod
    def create(cls, dimension: int) -> "FaissPassageIndex":
        faiss = require_faiss()
        return cls(faiss.IndexFlatIP(dimension), [])

    def add(self, vectors: np.ndarray, passages: Iterable[Passage]) -> None:
        if vectors.ndim != 2:
            raise ValueError("vectors must be a 2D matrix")
        self.index.add(np.ascontiguousarray(vectors.astype("float32")))  # type: ignore[attr-defined]
        self.passages.extend(passages)

    def search(self, query_vectors: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if query_vectors.ndim != 2:
            raise ValueError("query_vectors must be a 2D matrix")
        return self.index.search(  # type: ignore[attr-defined]
            np.ascontiguousarray(query_vectors.astype("float32")),
            top_k,
        )

    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        faiss = require_faiss()
        faiss.write_index(self.index, str(path / "passages.faiss"))
        with (path / "passages.jsonl").open("w", encoding="utf-8") as file:
            for passage in self.passages:
                file.write(
                    json.dumps(
                        {"docid": passage.docid, "title": passage.title, "text": passage.text},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    @classmethod
    def load(cls, directory: str | Path) -> "FaissPassageIndex":
        path = Path(directory)
        faiss = require_faiss()
        index = faiss.read_index(str(path / "passages.faiss"))
        passages: list[Passage] = []
        with (path / "passages.jsonl").open(encoding="utf-8") as file:
            for line in file:
                row = json.loads(line)
                passages.append(Passage(docid=row["docid"], title=row["title"], text=row["text"]))
        return cls(index, passages)
