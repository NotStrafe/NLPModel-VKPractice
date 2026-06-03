from __future__ import annotations

from math import log2


def hit_at_k(ranked_docids: list[str], relevant_docids: set[str], k: int) -> float:
    if not relevant_docids:
        return 0.0
    return float(any(docid in relevant_docids for docid in ranked_docids[:k]))


def reciprocal_rank(ranked_docids: list[str], relevant_docids: set[str], k: int) -> float:
    if not relevant_docids:
        return 0.0
    for rank, docid in enumerate(ranked_docids[:k], start=1):
        if docid in relevant_docids:
            return 1.0 / rank
    return 0.0


def dcg_at_k(ranked_docids: list[str], relevant_docids: set[str], k: int) -> float:
    score = 0.0
    for rank, docid in enumerate(ranked_docids[:k], start=1):
        if docid in relevant_docids:
            score += 1.0 / log2(rank + 1)
    return score


def ndcg_at_k(ranked_docids: list[str], relevant_docids: set[str], k: int) -> float:
    if not relevant_docids:
        return 0.0
    ideal_hits = min(len(relevant_docids), k)
    ideal = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if ideal == 0.0:
        return 0.0
    return dcg_at_k(ranked_docids, relevant_docids, k) / ideal


def aggregate_metrics(rankings: list[tuple[list[str], set[str]]], k: int) -> dict[str, float]:
    if not rankings:
        return {f"hit@{k}": 0.0, f"mrr@{k}": 0.0, f"ndcg@{k}": 0.0, "queries": 0.0}
    return {
        f"hit@{k}": sum(hit_at_k(rank, rel, k) for rank, rel in rankings) / len(rankings),
        f"mrr@{k}": sum(reciprocal_rank(rank, rel, k) for rank, rel in rankings) / len(rankings),
        f"ndcg@{k}": sum(ndcg_at_k(rank, rel, k) for rank, rel in rankings) / len(rankings),
        "queries": float(len(rankings)),
    }
