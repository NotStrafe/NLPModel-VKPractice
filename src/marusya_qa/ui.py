from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field


DEFAULT_INDEX_DIR = "artifacts/ru_miracl_index"
DEFAULT_TOP_K = 3
DEFAULT_MIN_SCORE = 0.25
DEFAULT_DEVICE = "auto"


def _load_env_file() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


_load_env_file()


def _clean_env_value(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    while cleaned.endswith("\\"):
        cleaned = cleaned[:-1].strip()
    return cleaned or None


def _env_value(name: str, default: str | None = None) -> str | None:
    return _clean_env_value(os.getenv(name, default))


def _env_int(name: str, default: int) -> int:
    value = _env_value(name, str(default))
    try:
        return int(value or default)
    except ValueError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Некорректное значение {name}={value!r}. Ожидается целое число.",
        ) from exc


def _env_float(name: str) -> float | None:
    value = _env_value(name)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Некорректное значение {name}={value!r}. Ожидается число.",
        ) from exc


class AnswerRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)


class Candidate(BaseModel):
    score: float
    docid: str
    title: str
    answer: str


class AnswerResponse(BaseModel):
    query: str
    accepted: bool
    min_score: float
    answer: str
    results: list[Candidate]


class StatusResponse(BaseModel):
    status: str
    index_ready: bool


def _settings() -> dict[str, object]:
    return {
        "index_dir": _env_value("MARUSYA_QA_INDEX_DIR", DEFAULT_INDEX_DIR),
        "top_k": _env_int("MARUSYA_QA_TOP_K", DEFAULT_TOP_K),
        "min_score": _env_float("MARUSYA_QA_MIN_SCORE"),
        "device": _env_value("MARUSYA_QA_DEVICE", DEFAULT_DEVICE),
    }


def _index_ready(index_dir: str) -> bool:
    path = Path(index_dir)
    return (path / "passages.faiss").exists() and (path / "passages.jsonl").exists()


def _public_status_payload() -> dict[str, object]:
    settings = _settings()
    index_dir = str(settings["index_dir"])
    return {
        "status": "ok",
        "index_ready": _index_ready(index_dir),
    }


def _health_payload() -> dict[str, object]:
    settings = _settings()
    index_dir = str(settings["index_dir"])
    return {
        "status": "ok",
        "index_ready": _index_ready(index_dir),
        "index_dir": index_dir,
        "top_k": int(settings["top_k"]),
        "device": str(settings["device"]),
    }


def _resolve_min_score(index_dir: str, explicit_value: float | None) -> float:
    if explicit_value is not None:
        return explicit_value
    threshold_file = Path(index_dir) / "threshold.json"
    if threshold_file.exists():
        data = json.loads(threshold_file.read_text(encoding="utf-8"))
        return float(data["threshold"])
    return DEFAULT_MIN_SCORE


@lru_cache(maxsize=4)
def _load_components(index_dir: str, device: str):
    from marusya_qa.config import RetrieverConfig
    from marusya_qa.embedding import DenseEncoder
    from marusya_qa.index_store import FaissPassageIndex

    config = RetrieverConfig.load(index_dir)
    encoder = DenseEncoder(config, device=device)
    index = FaissPassageIndex.load(index_dir)
    return encoder, index


def _answer(query: str) -> dict[str, object]:
    from marusya_qa.answer import answer_query_with_components

    settings = _settings()
    index_dir = str(settings["index_dir"])
    top_k = int(settings["top_k"])
    device = str(settings["device"])

    if not _index_ready(index_dir):
        raise HTTPException(
            status_code=503,
            detail=(
                "Индекс не найден. Сначала обучите модель и соберите индекс "
                f"в папке {index_dir}."
            ),
        )

    min_score = _resolve_min_score(index_dir, settings["min_score"])  # type: ignore[arg-type]
    encoder, index = _load_components(index_dir, device)
    return answer_query_with_components(
        query.strip(),
        encoder=encoder,
        index=index,
        top_k=top_k,
        min_score=min_score,
    )


def _page() -> str:
    return """
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MIRACL QA</title>
  <style>
    :root {
      --bg: #edf2f5;
      --surface: #ffffff;
      --surface-soft: #f7fafb;
      --ink: #182230;
      --muted: #667085;
      --line: #d5dde5;
      --accent: #176b87;
      --accent-dark: #0f5167;
      --ok: #146c43;
      --warn: #9a5b13;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }

    main {
      width: min(980px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 32px 0 52px;
    }

    .header,
    .search,
    .answer,
    .source {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
    }

    .header {
      padding: 24px 26px;
      margin-bottom: 16px;
    }

    h1 {
      margin: 0 0 8px;
      font-size: clamp(28px, 4vw, 40px);
      line-height: 1.08;
      letter-spacing: 0;
    }

    .subtitle {
      margin: 0;
      color: var(--muted);
      font-size: 16px;
      line-height: 1.55;
    }

    .search {
      padding: 20px;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
      align-items: start;
    }

    textarea {
      width: 100%;
      min-height: 92px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 13px 14px;
      font: inherit;
      font-size: 16px;
      line-height: 1.45;
      color: var(--ink);
      background: var(--surface-soft);
      outline: none;
    }

    textarea:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(23, 107, 135, 0.14);
      background: #ffffff;
    }

    button {
      min-height: 44px;
      min-width: 132px;
      border: 1px solid var(--accent-dark);
      border-radius: 6px;
      background: var(--accent);
      color: #ffffff;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
      padding: 0 18px;
    }

    button:hover {
      background: var(--accent-dark);
    }

    button:disabled {
      cursor: wait;
      opacity: 0.68;
    }

    .answer {
      display: none;
      margin-top: 18px;
      padding: 20px 22px;
      border-left-width: 5px;
    }

    .answer.accepted {
      display: block;
      border-left-color: var(--ok);
    }

    .answer.rejected {
      display: block;
      border-left-color: var(--warn);
    }

    .label {
      color: var(--muted);
      font-size: 13px;
      font-weight: 750;
      text-transform: uppercase;
      margin-bottom: 8px;
    }

    .answer-text {
      font-size: 18px;
      line-height: 1.62;
      white-space: pre-wrap;
    }

    .sources {
      display: none;
      margin-top: 24px;
    }

    .sources.visible {
      display: block;
    }

    h2 {
      font-size: 20px;
      margin: 0 0 12px;
      letter-spacing: 0;
    }

    .source {
      padding: 16px 18px;
      margin: 12px 0;
    }

    .source-title {
      font-weight: 750;
      margin-bottom: 7px;
    }

    .source-meta {
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 10px;
    }

    .source-text {
      color: #303946;
      font-size: 14px;
      line-height: 1.56;
      white-space: pre-wrap;
    }

    .error {
      display: none;
      margin-top: 16px;
      border: 1px solid #efc7a7;
      border-radius: 8px;
      background: #fff7ed;
      color: #7c3f12;
      padding: 14px 16px;
      line-height: 1.5;
    }

    .error.visible {
      display: block;
    }

    .status {
      margin: 16px 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #ffffff;
      padding: 14px 16px;
      color: var(--muted);
      line-height: 1.5;
    }

    .status.ready {
      border-color: #b7ddc6;
      background: #f1fbf5;
      color: #1f5132;
    }

    .status.missing {
      border-color: #efc7a7;
      background: #fff7ed;
      color: #7c3f12;
    }

    code {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 0.92em;
      background: rgba(255, 255, 255, 0.7);
      border: 1px solid rgba(0, 0, 0, 0.08);
      border-radius: 4px;
      padding: 1px 4px;
    }

    @media (max-width: 720px) {
      main {
        width: min(100vw - 24px, 980px);
        padding-top: 18px;
      }

      .search {
        grid-template-columns: 1fr;
      }

      button {
        width: 100%;
      }
    }
  </style>
</head>
<body>
  <main>
    <section class="header">
      <h1>MIRACL QA</h1>
      <p class="subtitle">Введите вопрос на русском языке. Сервис найдет релевантный фрагмент в индексе и отсечет ответ, если уверенность ниже порога.</p>
    </section>

    <section class="search">
      <textarea id="query" maxlength="500" placeholder="Например: Кто такой Юрий Гагарин?">Кто такой Юрий Гагарин?</textarea>
      <button id="submit" type="button">Найти ответ</button>
    </section>

    <div id="status" class="status">Проверяю индекс...</div>
    <div id="error" class="error"></div>

    <section id="answer" class="answer">
      <div id="answerLabel" class="label"></div>
      <div id="answerText" class="answer-text"></div>
    </section>

    <section id="sources" class="sources">
      <h2>Источники</h2>
      <div id="sourceList"></div>
    </section>
  </main>

  <script>
    const query = document.getElementById("query");
    const submit = document.getElementById("submit");
    const statusBox = document.getElementById("status");
    const error = document.getElementById("error");
    const answer = document.getElementById("answer");
    const answerLabel = document.getElementById("answerLabel");
    const answerText = document.getElementById("answerText");
    const sources = document.getElementById("sources");
    const sourceList = document.getElementById("sourceList");

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    function setError(message) {
      error.textContent = message;
      error.classList.add("visible");
      answer.className = "answer";
      sources.className = "sources";
    }

    function clearError() {
      error.textContent = "";
      error.classList.remove("visible");
    }

    function renderStatus(payload) {
      if (payload.index_ready) {
        statusBox.className = "status ready";
        statusBox.textContent = "Индекс готов.";
        submit.disabled = false;
        return;
      }

      statusBox.className = "status missing";
      statusBox.innerHTML = `
        Индекс не готов.
        Сначала выполните обучение и индексацию: <code>python3 -m marusya_qa.train --max-train-queries 200 --epochs 1 --batch-size 8</code>,
        затем <code>python3 -m marusya_qa.index --max-docs 10000</code>.
      `;
      submit.disabled = true;
    }

    function render(payload) {
      answer.className = `answer ${payload.accepted ? "accepted" : "rejected"}`;
      answerLabel.textContent = payload.accepted ? "Ответ найден" : "Ответ отсечен";
      answerText.textContent = payload.answer;

      sourceList.innerHTML = payload.results.map((item) => `
        <article class="source">
          <div class="source-title">${escapeHtml(item.title || "Без названия")}</div>
          <div class="source-meta">score=${Number(item.score).toFixed(4)} · docid=${escapeHtml(item.docid)}</div>
          <div class="source-text">${escapeHtml(item.answer)}</div>
        </article>
      `).join("");
      sources.classList.toggle("visible", payload.results.length > 0);
    }

    async function ask() {
      const text = query.value.trim();
      if (!text) {
        setError("Введите вопрос.");
        return;
      }

      clearError();
      submit.disabled = true;
      submit.textContent = "Ищу...";

      try {
        const response = await fetch("/api/answer", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({query: text})
        });
        const rawBody = await response.text();
        let payload = {};
        try {
          payload = rawBody ? JSON.parse(rawBody) : {};
        } catch {
          payload = {detail: rawBody || "Сервер вернул ответ в неизвестном формате."};
        }
        if (!response.ok) {
          throw new Error(payload.detail || "Не удалось получить ответ.");
        }
        render(payload);
      } catch (err) {
        setError(err.message);
      } finally {
        submit.disabled = false;
        submit.textContent = "Найти ответ";
      }
    }

    async function loadStatus() {
      try {
        const response = await fetch("/api/status");
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.detail || "Не удалось проверить индекс.");
        }
        renderStatus(payload);
      } catch (err) {
        statusBox.className = "status missing";
        statusBox.textContent = err.message;
        submit.disabled = true;
      }
    }

    submit.addEventListener("click", ask);
    query.addEventListener("keydown", (event) => {
      if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
        ask();
      }
    });
    loadStatus();
  </script>
</body>
</html>
"""


app = FastAPI(title="MIRACL QA", version="0.1.0")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _page()


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/health")
def health() -> dict[str, object]:
    return _health_payload()


@app.get("/api/status", response_model=StatusResponse)
def api_status() -> dict[str, object]:
    return _public_status_payload()


@app.post("/api/answer", response_model=AnswerResponse)
def api_answer(request: AnswerRequest) -> dict[str, object]:
    return _answer(request.query)


def main() -> None:
    import uvicorn

    host = _env_value("MARUSYA_QA_HOST", "127.0.0.1") or "127.0.0.1"
    port = _env_int("MARUSYA_QA_PORT", 8000)
    uvicorn.run("marusya_qa.ui:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
