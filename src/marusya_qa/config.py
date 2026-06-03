from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json


DEFAULT_LANG = "ru"
DEFAULT_QUERY_DATASET = "miracl/miracl"
DEFAULT_CORPUS_DATASET = "miracl/miracl-corpus"
DEFAULT_BASE_MODEL = "intfloat/multilingual-e5-small"
DEFAULT_QUERY_PREFIX = "query: "
DEFAULT_PASSAGE_PREFIX = "passage: "


@dataclass(frozen=True)
class RetrieverConfig:
    model_name_or_path: str = DEFAULT_BASE_MODEL
    query_prefix: str = DEFAULT_QUERY_PREFIX
    passage_prefix: str = DEFAULT_PASSAGE_PREFIX
    normalize_embeddings: bool = True

    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "retriever_config.json").write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path, fallback_model: str | None = None) -> "RetrieverConfig":
        path = Path(directory) / "retriever_config.json"
        if not path.exists():
            model = fallback_model or str(directory)
            if not Path(str(directory)).exists() and str(directory).startswith("checkpoints/"):
                model = DEFAULT_BASE_MODEL
            return cls(model_name_or_path=model)
        data = json.loads(path.read_text(encoding="utf-8"))
        if fallback_model is not None:
            data["model_name_or_path"] = fallback_model
        return cls(**data)
