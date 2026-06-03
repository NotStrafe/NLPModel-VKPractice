from __future__ import annotations

import argparse

from tqdm import tqdm

from marusya_qa.config import DEFAULT_CORPUS_DATASET, DEFAULT_LANG, RetrieverConfig
from marusya_qa.data import load_miracl_corpus, row_to_passage
from marusya_qa.embedding import DenseEncoder
from marusya_qa.index_store import FaissPassageIndex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS index for MIRACL passages.")
    parser.add_argument("--model", default="checkpoints/marusya-miracl-e5")
    parser.add_argument("--index-dir", default="artifacts/ru_miracl_index")
    parser.add_argument("--lang", default=DEFAULT_LANG)
    parser.add_argument("--corpus-dataset", default=DEFAULT_CORPUS_DATASET)
    parser.add_argument(
        "--corpus-config",
        default=None,
        help="Dataset config. Use an empty string for datasets without configs.",
    )
    parser.add_argument("--corpus-split", default="train")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto", help="auto, mps, cuda, cpu or any torch device.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RetrieverConfig.load(args.model, fallback_model=args.model)
    encoder = DenseEncoder(config, device=args.device)
    rows = load_miracl_corpus(
        lang=args.lang,
        dataset_name=args.corpus_dataset,
        config_name=args.corpus_config,
        split=args.corpus_split,
        cache_dir=args.cache_dir,
        streaming=args.streaming,
        limit=args.max_docs,
    )

    index: FaissPassageIndex | None = None
    batch_passages = []
    batch_texts = []
    total = 0

    for row in tqdm(rows, desc="Indexing passages"):
        passage = row_to_passage(row)
        if not passage.docid or not passage.full_text:
            continue
        batch_passages.append(passage)
        batch_texts.append(passage.full_text)
        if len(batch_texts) >= args.batch_size:
            vectors = encoder.encode_passages(batch_texts, batch_size=args.batch_size)
            if index is None:
                index = FaissPassageIndex.create(vectors.shape[1])
            index.add(vectors, batch_passages)
            total += len(batch_passages)
            batch_passages = []
            batch_texts = []

    if batch_texts:
        vectors = encoder.encode_passages(batch_texts, batch_size=args.batch_size)
        if index is None:
            index = FaissPassageIndex.create(vectors.shape[1])
        index.add(vectors, batch_passages)
        total += len(batch_passages)

    if index is None or total == 0:
        raise RuntimeError("No passages were indexed.")

    index.save(args.index_dir)
    config.save(args.index_dir)
    print(f"Indexed {total} passages into {args.index_dir}")


if __name__ == "__main__":
    main()
