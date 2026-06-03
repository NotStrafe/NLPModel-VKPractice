from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
import gzip
import json
from pathlib import Path
from typing import Any

from marusya_qa.config import DEFAULT_CORPUS_DATASET, DEFAULT_LANG, DEFAULT_QUERY_DATASET


@dataclass(frozen=True)
class Passage:
    docid: str
    title: str
    text: str

    @property
    def full_text(self) -> str:
        if self.title and self.title not in self.text[: max(len(self.title) + 10, 30)]:
            return f"{self.title}\n{self.text}".strip()
        return self.text.strip()


def _limited(dataset: Any, limit: int | None) -> Any:
    if limit is None:
        return dataset
    if hasattr(dataset, "take"):
        return dataset.take(limit)
    return dataset.select(range(min(limit, len(dataset))))


def _is_default_miracl(dataset_name: str) -> bool:
    return dataset_name == DEFAULT_QUERY_DATASET


def _is_default_miracl_corpus(dataset_name: str) -> bool:
    return dataset_name == DEFAULT_CORPUS_DATASET


def _hf_download(repo_id: str, filename: str, cache_dir: str | None) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        cache_dir=cache_dir,
    )


def _read_topics(path: str) -> dict[str, str]:
    topics: dict[str, str] = {}
    with open(path, encoding="utf-8") as file:
        for line in file:
            line = line.rstrip("\n")
            if not line:
                continue
            qid, query = line.split("\t", maxsplit=1)
            topics[qid] = query
    return topics


def _read_qrels(path: str) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = {}
    with open(path, encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            if len(parts) >= 4:
                qid, docid, relevance = parts[0], parts[2], parts[3]
            else:
                qid, docid, relevance = parts[0], parts[1], parts[2]
            try:
                if int(float(relevance)) <= 0:
                    continue
            except ValueError:
                continue
            qrels.setdefault(qid, set()).add(docid)
    return qrels


def _miracl_split(split: str) -> str:
    aliases = {"validation": "dev", "valid": "dev"}
    return aliases.get(split, split)


def _miracl_query_rows(
    *,
    split: str,
    lang: str,
    cache_dir: str | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    split = _miracl_split(split)
    topics_file = _hf_download(
        DEFAULT_QUERY_DATASET,
        f"miracl-v1.0-{lang}/topics/topics.miracl-v1.0-{lang}-{split}.tsv",
        cache_dir,
    )
    topics = _read_topics(topics_file)

    qrels: dict[str, set[str]] = {}
    if split in {"train", "dev"}:
        qrels_file = _hf_download(
            DEFAULT_QUERY_DATASET,
            f"miracl-v1.0-{lang}/qrels/qrels.miracl-v1.0-{lang}-{split}.tsv",
            cache_dir,
        )
        qrels = _read_qrels(qrels_file)

    rows: list[dict[str, Any]] = []
    for qid, query in topics.items():
        positive = [
            {"docid": docid, "title": "", "text": ""}
            for docid in sorted(qrels.get(qid, set()))
        ]
        rows.append({"query_id": qid, "query": query, "positive_passages": positive})
        if limit is not None and len(rows) >= limit:
            break
    return rows


def _corpus_files(lang: str, cache_dir: str | None) -> list[str]:
    from huggingface_hub import list_repo_files

    prefix = f"miracl-corpus-v1.0-{lang}/docs-"
    files = [
        file
        for file in list_repo_files(DEFAULT_CORPUS_DATASET, repo_type="dataset")
        if file.startswith(prefix) and file.endswith(".jsonl.gz")
    ]
    return sorted(files, key=lambda name: int(Path(name).stem.split("-")[-1].split(".")[0]))


def iter_miracl_corpus_rows(
    *,
    lang: str = DEFAULT_LANG,
    cache_dir: str | None = None,
    limit: int | None = None,
) -> Iterator[dict[str, Any]]:
    yielded = 0
    for filename in _corpus_files(lang, cache_dir):
        path = _hf_download(DEFAULT_CORPUS_DATASET, filename, cache_dir)
        with gzip.open(path, "rt", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                yield json.loads(line)
                yielded += 1
                if limit is not None and yielded >= limit:
                    return


def load_passages_by_docid(
    docids: set[str],
    *,
    lang: str = DEFAULT_LANG,
    cache_dir: str | None = None,
) -> dict[str, Passage]:
    remaining = set(docids)
    found: dict[str, Passage] = {}
    if not remaining:
        return found

    for row in iter_miracl_corpus_rows(lang=lang, cache_dir=cache_dir):
        passage = row_to_passage(row)
        if passage.docid in remaining:
            found[passage.docid] = passage
            remaining.remove(passage.docid)
            if not remaining:
                break
    return found


def load_miracl_queries(
    *,
    split: str,
    lang: str = DEFAULT_LANG,
    dataset_name: str = DEFAULT_QUERY_DATASET,
    config_name: str | None = None,
    cache_dir: str | None = None,
    streaming: bool = False,
    limit: int | None = None,
) -> Iterable[dict[str, Any]]:
    if _is_default_miracl(dataset_name):
        if streaming:
            raise ValueError("Streaming is not supported for direct MIRACL file loading.")
        return _miracl_query_rows(split=split, lang=lang, cache_dir=cache_dir, limit=limit)

    from datasets import load_dataset

    config = lang if config_name is None else config_name
    args = [dataset_name] if config == "" else [dataset_name, config]
    dataset = load_dataset(
        *args,
        split=split,
        cache_dir=cache_dir,
        streaming=streaming,
    )
    return _limited(dataset, limit)


def load_miracl_corpus(
    *,
    lang: str = DEFAULT_LANG,
    dataset_name: str = DEFAULT_CORPUS_DATASET,
    config_name: str | None = None,
    split: str = "train",
    cache_dir: str | None = None,
    streaming: bool = False,
    limit: int | None = None,
) -> Iterable[dict[str, Any]]:
    if _is_default_miracl_corpus(dataset_name):
        if streaming:
            raise ValueError("Streaming is not supported for direct MIRACL corpus loading.")
        return iter_miracl_corpus_rows(lang=lang, cache_dir=cache_dir, limit=limit)

    from datasets import load_dataset

    config = lang if config_name is None else config_name
    args = [dataset_name] if config == "" else [dataset_name, config]
    dataset = load_dataset(
        *args,
        split=split,
        cache_dir=cache_dir,
        streaming=streaming,
    )
    return _limited(dataset, limit)


def row_to_passage(row: dict[str, Any]) -> Passage:
    docid = str(row.get("docid") or row.get("id") or row.get("_id") or "")
    title = str(row.get("title") or "")
    text = str(row.get("text") or row.get("passage") or row.get("contents") or "")
    return Passage(docid=docid, title=title, text=text)


def positive_passages(row: dict[str, Any]) -> list[Passage]:
    passages = row.get("positive_passages") or row.get("positives") or []
    return [row_to_passage(passage) for passage in passages if passage]


def negative_passages(row: dict[str, Any]) -> list[Passage]:
    passages = row.get("negative_passages") or row.get("negatives") or []
    return [row_to_passage(passage) for passage in passages if passage]


def iter_training_pairs(rows: Iterable[dict[str, Any]]) -> Iterator[tuple[str, str]]:
    for row in rows:
        query = str(row.get("query") or row.get("question") or "").strip()
        if not query:
            continue
        for passage in positive_passages(row):
            text = passage.full_text
            if text:
                yield query, text


def relevant_docids(row: dict[str, Any]) -> set[str]:
    return {passage.docid for passage in positive_passages(row) if passage.docid}
