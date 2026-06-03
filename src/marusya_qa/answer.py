from __future__ import annotations

import argparse
import json
from pathlib import Path

from marusya_qa.config import RetrieverConfig
from marusya_qa.embedding import DenseEncoder
from marusya_qa.index_store import FaissPassageIndex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Answer a Russian user query using the passage index.")
    parser.add_argument("query", help="User question in Russian.")
    parser.add_argument("--index-dir", default="artifacts/ru_miracl_index")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--device", default="auto", help="auto, mps, cuda, cpu or any torch device.")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def resolve_min_score(index_dir: str, explicit_value: float | None) -> float:
    if explicit_value is not None:
        return explicit_value
    threshold_file = Path(index_dir) / "threshold.json"
    if threshold_file.exists():
        data = json.loads(threshold_file.read_text(encoding="utf-8"))
        return float(data["threshold"])
    return 0.25


def answer_query(
    query: str,
    *,
    index_dir: str = "artifacts/ru_miracl_index",
    top_k: int = 3,
    min_score: float | None = None,
    device: str | None = "auto",
) -> dict[str, object]:
    resolved_min_score = resolve_min_score(index_dir, min_score)
    config = RetrieverConfig.load(index_dir)
    encoder = DenseEncoder(config, device=device)
    index = FaissPassageIndex.load(index_dir)
    return answer_query_with_components(
        query,
        encoder=encoder,
        index=index,
        top_k=top_k,
        min_score=resolved_min_score,
    )


def answer_query_with_components(
    query: str,
    *,
    encoder: DenseEncoder,
    index: FaissPassageIndex,
    top_k: int,
    min_score: float,
) -> dict[str, object]:
    query_vector = encoder.encode_queries([query], batch_size=1)
    scores, indices = index.search(query_vector, top_k)

    results = []
    for score, idx in zip(scores[0].tolist(), indices[0].tolist(), strict=True):
        if idx < 0:
            continue
        passage = index.passages[idx]
        results.append(
            {
                "score": float(score),
                "docid": passage.docid,
                "title": passage.title,
                "answer": passage.text,
            }
        )

    accepted = results[0] if results and results[0]["score"] >= min_score else None
    return {
        "query": query,
        "accepted": accepted is not None,
        "min_score": min_score,
        "answer": accepted["answer"] if accepted else "Не удалось найти релевантный ответ.",
        "results": results,
    }


def main() -> None:
    args = parse_args()
    min_score = resolve_min_score(args.index_dir, args.min_score)
    config = RetrieverConfig.load(args.index_dir)
    encoder = DenseEncoder(config, device=args.device)
    index = FaissPassageIndex.load(args.index_dir)
    payload = answer_query_with_components(
        args.query,
        encoder=encoder,
        index=index,
        top_k=args.top_k,
        min_score=min_score,
    )

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print(payload["answer"])
    if results:
        print(f"\nscore={results[0]['score']:.4f}; title={results[0]['title']}; docid={results[0]['docid']}")


if __name__ == "__main__":
    main()
