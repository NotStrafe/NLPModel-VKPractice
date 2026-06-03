from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader
from sentence_transformers import InputExample, SentenceTransformer
from sentence_transformers.sentence_transformer import losses

from marusya_qa.config import (
    DEFAULT_BASE_MODEL,
    DEFAULT_LANG,
    DEFAULT_PASSAGE_PREFIX,
    DEFAULT_QUERY_PREFIX,
    DEFAULT_QUERY_DATASET,
    RetrieverConfig,
)
from marusya_qa.data import iter_training_pairs, load_miracl_queries, load_passages_by_docid
from marusya_qa.device import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train dense retriever on Russian MIRACL pairs.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output-dir", default="checkpoints/marusya-miracl-e5")
    parser.add_argument("--lang", default=DEFAULT_LANG)
    parser.add_argument("--query-dataset", default=DEFAULT_QUERY_DATASET)
    parser.add_argument(
        "--query-config",
        default=None,
        help="Dataset config. Use an empty string for datasets without configs.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--max-train-queries", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--query-prefix", default=DEFAULT_QUERY_PREFIX)
    parser.add_argument("--passage-prefix", default=DEFAULT_PASSAGE_PREFIX)
    parser.add_argument("--device", default="auto", help="auto, mps, cuda, cpu or any torch device.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_miracl_queries(
        split=args.split,
        lang=args.lang,
        dataset_name=args.query_dataset,
        config_name=args.query_config,
        cache_dir=args.cache_dir,
        streaming=args.streaming,
        limit=args.max_train_queries,
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

    examples = [
        InputExample(texts=[f"{args.query_prefix}{query}", f"{args.passage_prefix}{passage}"])
        for query, passage in iter_training_pairs(rows)
    ]
    if not examples:
        raise RuntimeError("No training examples were created from MIRACL rows.")

    device = resolve_device(args.device)
    model = SentenceTransformer(args.base_model, device=device)
    print(f"Training device: {device or 'sentence-transformers default'}")
    train_loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    warmup_steps = int(len(train_loader) * args.epochs * args.warmup_ratio)

    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.learning_rate},
        output_path=args.output_dir,
        show_progress_bar=True,
    )

    RetrieverConfig(
        model_name_or_path=str(Path(args.output_dir)),
        query_prefix=args.query_prefix,
        passage_prefix=args.passage_prefix,
    ).save(args.output_dir)
    print(f"Saved trained retriever to {args.output_dir}")


if __name__ == "__main__":
    main()
