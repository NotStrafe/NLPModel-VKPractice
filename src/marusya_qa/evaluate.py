from __future__ import annotations

import argparse
import json

from tqdm import tqdm

from marusya_qa.config import DEFAULT_LANG, DEFAULT_QUERY_DATASET, RetrieverConfig
from marusya_qa.data import load_miracl_queries, relevant_docids
from marusya_qa.embedding import DenseEncoder
from marusya_qa.index_store import FaissPassageIndex
from marusya_qa.metrics import aggregate_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retriever on MIRACL dev queries.")
    parser.add_argument("--index-dir", default="artifacts/ru_miracl_index")
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
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", help="auto, mps, cuda, cpu or any torch device.")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RetrieverConfig.load(args.index_dir)
    encoder = DenseEncoder(config, device=args.device)
    index = FaissPassageIndex.load(args.index_dir)
    rows = load_miracl_queries(
        split=args.split,
        lang=args.lang,
        dataset_name=args.query_dataset,
        config_name=args.query_config,
        cache_dir=args.cache_dir,
        streaming=args.streaming,
        limit=args.max_queries,
    )

    rankings: list[tuple[list[str], set[str]]] = []
    batch_queries: list[str] = []
    batch_relevant: list[set[str]] = []

    def flush_batch() -> None:
        if not batch_queries:
            return
        print(f"Encoding/searching {len(batch_queries)} queries...", flush=True)
        vectors = encoder.encode_queries(batch_queries, batch_size=args.batch_size)
        _, indices = index.search(vectors, args.top_k)
        for row_indices, rel in zip(indices.tolist(), batch_relevant, strict=True):
            ranked_docids = [
                index.passages[idx].docid for idx in row_indices if idx >= 0 and idx < len(index.passages)
            ]
            rankings.append((ranked_docids, rel))
        batch_queries.clear()
        batch_relevant.clear()

    for row in tqdm(rows, desc="Evaluating queries"):
        query = str(row.get("query") or "").strip()
        rel = relevant_docids(row)
        if not query or not rel:
            continue
        batch_queries.append(query)
        batch_relevant.append(rel)
        if len(batch_queries) >= args.batch_size:
            flush_batch()
    flush_batch()

    metrics = aggregate_metrics(rankings, args.top_k)
    text = json.dumps(metrics, ensure_ascii=False, indent=2)
    print(text, flush=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as file:
            file.write(text + "\n")


if __name__ == "__main__":
    main()
