from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from marusya_qa.config import DEFAULT_LANG, DEFAULT_QUERY_DATASET, RetrieverConfig
from marusya_qa.data import (
    iter_miracl_corpus_rows,
    load_miracl_queries,
    load_passages_by_docid,
    negative_passages,
    positive_passages,
    row_to_passage,
)
from marusya_qa.embedding import DenseEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate relevance threshold on MIRACL dev labels.")
    parser.add_argument("--model", default="checkpoints/marusya-miracl-e5")
    parser.add_argument("--lang", default=DEFAULT_LANG)
    parser.add_argument("--query-dataset", default=DEFAULT_QUERY_DATASET)
    parser.add_argument(
        "--query-config",
        default=None,
        help="Dataset config. Use an empty string for datasets without configs.",
    )
    parser.add_argument("--split", default="dev")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--max-negatives-per-query", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", help="auto, mps, cuda, cpu or any torch device.")
    parser.add_argument("--output", default="artifacts/ru_miracl_index/threshold.json")
    return parser.parse_args()


def best_f1_threshold(scores: list[float], labels: list[int]) -> dict[str, float]:
    if not scores or not labels or len(scores) != len(labels):
        raise ValueError("scores and labels must be non-empty lists with equal length")

    best = {"threshold": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    for threshold in sorted(set(scores)):
        predicted = [score >= threshold for score in scores]
        tp = sum(int(pred and label == 1) for pred, label in zip(predicted, labels, strict=True))
        fp = sum(int(pred and label == 0) for pred, label in zip(predicted, labels, strict=True))
        fn = sum(int((not pred) and label == 1) for pred, label in zip(predicted, labels, strict=True))

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1 > best["f1"]:
            best = {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
    return best


def main() -> None:
    args = parse_args()
    config = RetrieverConfig.load(args.model, fallback_model=args.model)
    encoder = DenseEncoder(config, device=args.device)
    rows = load_miracl_queries(
        split=args.split,
        lang=args.lang,
        dataset_name=args.query_dataset,
        config_name=args.query_config,
        cache_dir=args.cache_dir,
        streaming=args.streaming,
        limit=args.max_queries,
    )
    rows = list(rows)
    needed_docids = {
        str(passage["docid"])
        for row in rows
        for passage in row.get("positive_passages", [])
        if passage.get("docid") and not passage.get("text")
    }
    if needed_docids:
        print(f"Loading {len(needed_docids)} positive passages from MIRACL corpus...")
        passages = load_passages_by_docid(needed_docids, lang=args.lang, cache_dir=args.cache_dir)
        for row in rows:
            for passage in row.get("positive_passages", []):
                resolved = passages.get(str(passage.get("docid")))
                if resolved is not None:
                    passage["title"] = resolved.title
                    passage["text"] = resolved.text

    positive_docids = {
        str(passage.get("docid"))
        for row in rows
        for passage in row.get("positive_passages", [])
        if passage.get("docid")
    }
    negative_pool = []
    target_negatives = max(len(rows) * args.max_negatives_per_query, args.max_negatives_per_query)
    for corpus_row in iter_miracl_corpus_rows(
        lang=args.lang,
        cache_dir=args.cache_dir,
        limit=target_negatives + len(positive_docids) + 100,
    ):
        passage = row_to_passage(corpus_row)
        if passage.docid and passage.docid not in positive_docids and passage.full_text:
            negative_pool.append(passage.full_text)
            if len(negative_pool) >= target_negatives:
                break

    scores: list[float] = []
    labels: list[int] = []
    negative_offset = 0
    for row in tqdm(rows, desc="Calibrating threshold"):
        query = str(row.get("query") or "").strip()
        positives = [passage.full_text for passage in positive_passages(row) if passage.full_text]
        negatives = [
            passage.full_text
            for passage in negative_passages(row)[: args.max_negatives_per_query]
            if passage.full_text
        ]
        if not negatives and negative_pool:
            negatives = [
                negative_pool[(negative_offset + idx) % len(negative_pool)]
                for idx in range(args.max_negatives_per_query)
            ]
            negative_offset += args.max_negatives_per_query
        if not query or not positives or not negatives:
            continue

        query_vector = encoder.encode_queries([query], batch_size=1)
        passage_texts = positives + negatives
        passage_vectors = encoder.encode_passages(passage_texts, batch_size=args.batch_size)
        row_scores = np.dot(passage_vectors, query_vector[0]).tolist()
        scores.extend(float(score) for score in row_scores)
        labels.extend([1] * len(positives) + [0] * len(negatives))

    result = best_f1_threshold(scores, labels)
    result["pairs"] = float(len(labels))
    result["positive_pairs"] = float(sum(labels))
    result["negative_pairs"] = float(len(labels) - sum(labels))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
