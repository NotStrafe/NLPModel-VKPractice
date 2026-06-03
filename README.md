# NLPModel-VKPractice

Проект решает задачу из ТЗ: по открытому датасету MIRACL обучается retrieval-модель,
которая выбирает наиболее релевантный русскоязычный фрагмент Wikipedia для вопроса
пользователя и отсекает нерелевантные ответы по порогу score.

## Цель

Собрать воспроизводимый NLP-пайплайн для голосового помощника в стиле Маруси:

- загрузить русскую часть MIRACL;
- обучить dense retriever на парах `query -> positive_passage`;
- построить FAISS-индекс корпуса;
- оценить качество ранжирования на `dev`;
- откалибровать порог отсечения нерелевантных ответов;
- дать веб-интерфейс и CLI для ответа на пользовательский вопрос.

## Данные

- Запросы и разметка: `miracl/miracl`, конфигурация `ru`.
- Корпус passages: `miracl/miracl-corpus`, конфигурация `ru`.
- Указанные в ТЗ embeddings `Cohere/miracl-ru-queries-22-12` можно использовать как
  внешний baseline, но основной код проекта обучает собственный retriever на MIRACL.

Для датасетов без конфигурации передайте пустой `--query-config ""` или
`--corpus-config ""`, например для `Cohere/miracl-ru-queries-22-12`.
MIRACL загружается напрямую из файлов Hugging Face Hub (`topics`, `qrels`,
`docs-*.jsonl.gz`), потому что новые версии `datasets` больше не поддерживают старые
dataset loading scripts вроде `miracl.py`.

## Подход

Модель формулируется как задача информационного поиска:

1. Вопрос пользователя кодируется bi-encoder моделью.
2. Все passages корпуса кодируются той же моделью и складываются в FAISS `IndexFlatIP`.
3. Для вопроса выбираются top-k passages по cosine similarity.
4. Порог релевантности калибруется на `positive_passages` и `negative_passages`.
5. Если лучший score ниже `--min-score`, ответ отклоняется как нерелевантный.

Базовая модель по умолчанию: `intfloat/multilingual-e5-small`. Она поддерживает русский
язык и подходит для семантического поиска. Дообучение выполняется через
`MultipleNegativesRankingLoss`.

Устройство выбирается автоматически в порядке `mps -> cuda -> cpu`. При необходимости
его можно переопределить аргументом `--device`.

## Конфигурация `.env`

Для локального запуска можно скопировать `.env.example` в `.env`. Файл `.env`
игнорируется git, поэтому в него можно класть локальные настройки без риска случайно
запушить их в репозиторий.

```bash
cp .env.example .env
```

Содержимое `.env.example`:

```dotenv
MARUSYA_QA_HOST=127.0.0.1
MARUSYA_QA_PORT=8000
MARUSYA_QA_INDEX_DIR=artifacts/ru_miracl_index
MARUSYA_QA_TOP_K=3
MARUSYA_QA_DEVICE=auto

# Optional. If unset, the service reads threshold.json near the index.
# If threshold.json is absent, fallback 0.25 is used.
# MARUSYA_QA_MIN_SCORE=0.25
```

Назначение переменных:

- `MARUSYA_QA_HOST`: адрес FastAPI-сервера.
- `MARUSYA_QA_PORT`: порт FastAPI-сервера.
- `MARUSYA_QA_INDEX_DIR`: папка с `passages.faiss`, `passages.jsonl` и опциональным `threshold.json`.
- `MARUSYA_QA_TOP_K`: сколько кандидатов извлекать из индекса.
- `MARUSYA_QA_DEVICE`: устройство для encoder, обычно `auto`.
- `MARUSYA_QA_MIN_SCORE`: ручной порог отсечения; если не задан, используется `threshold.json`.

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
```

Если `pip install -e ".[dev]"` в локальной среде пытается заново скачать build-зависимости
и падает, установите проект без build isolation:

```bash
pip install -e . --no-build-isolation
```

Без editable-установки команды вида `python3 -m marusya_qa.train` не найдут пакет и
упадут с `ModuleNotFoundError: No module named 'marusya_qa'`.

## Быстрый smoke run

Полная русская часть MIRACL большая, поэтому для проверки пайплайна можно ограничить
число запросов и документов. Такой запуск проверяет, что обучение, индекс, оценка и UI
работают технически. Для демонстрации качества и ответа на произвольные вопросы нужен
полный индекс без `--max-docs`.

```bash
python3 -m marusya_qa.train \
  --max-train-queries 200 \
  --epochs 1 \
  --batch-size 8 \
  --output-dir checkpoints/marusya-miracl-e5

python3 -m marusya_qa.index \
  --model checkpoints/marusya-miracl-e5 \
  --max-docs 10000 \
  --index-dir artifacts/ru_miracl_index

python3 -m marusya_qa.evaluate \
  --index-dir artifacts/ru_miracl_index \
  --max-queries 100 \
  --top-k 10

python3 -m marusya_qa.calibrate \
  --model checkpoints/marusya-miracl-e5 \
  --max-queries 100 \
  --output artifacts/ru_miracl_index/threshold.json

python3 -m marusya_qa.answer \
  "Кто такой Юрий Гагарин?" \
  --index-dir artifacts/ru_miracl_index \
  --top-k 3 \
  --min-score 0.25

marusya-qa-ui
```

## Полный запуск

Для итогового результата уберите `--max-train-queries`, `--max-docs` и `--max-queries`.
На CPU полный индекс будет строиться долго; для ускорения укажите GPU/MPS-устройство,
если оно доступно для `sentence-transformers`.

```bash
python3 -m marusya_qa.train --epochs 1 --batch-size 16
python3 -m marusya_qa.index --batch-size 128
python3 -m marusya_qa.evaluate --top-k 10 --output artifacts/eval_dev.json
python3 -m marusya_qa.calibrate --output artifacts/ru_miracl_index/threshold.json
marusya-qa-ui
```

## Веб-интерфейс

После установки проекта командой `pip install -e ".[dev]"` запустите:

```bash
marusya-qa-ui
```

Откройте `http://127.0.0.1:8000`. В интерфейсе есть только пользовательский сценарий:
ввести вопрос и получить ответ с источниками. Параметры поиска не выводятся на экран,
потому что ТЗ требует модель для выбора релевантных ответов, а не ручной подбор
настроек пользователем.

Серверные настройки можно задать через переменные окружения:

```bash
MARUSYA_QA_INDEX_DIR=artifacts/ru_miracl_index \
MARUSYA_QA_TOP_K=3 \
MARUSYA_QA_DEVICE=auto \
marusya-qa-ui
```

Если `MARUSYA_QA_MIN_SCORE` не задан, сервис использует `threshold.json` рядом с
индексом или fallback `0.25`.

## Формат результата

CLI `answer` возвращает лучший фрагмент, если он прошел порог релевантности:

```bash
python3 -m marusya_qa.answer "Кто такой Юрий Гагарин?" --json
```

Если рядом с индексом есть `threshold.json`, команда использует откалиброванный порог.
Иначе применяется fallback `0.25`. Порог можно переопределить через `--min-score`.

Пример JSON:

```json
{
  "query": "Кто такой Юрий Гагарин?",
  "accepted": true,
  "min_score": 0.25,
  "answer": "Лётчик-космонавт СССР...",
  "results": [
    {
      "score": 0.82,
      "docid": "123",
      "title": "Гагарин, Юрий Алексеевич",
      "answer": "..."
    }
  ]
}
```

## Метрики

Оценка считает стандартные retrieval-метрики:

- `hit@k`: есть ли хотя бы один релевантный passage в top-k;
- `mrr@k`: средняя обратная позиция первого релевантного passage;
- `ndcg@k`: качество ранжирования с учетом позиции релевантных passages.

## Структура

```text
src/marusya_qa/
  data.py         # загрузка MIRACL и преобразование passages
  train.py        # обучение bi-encoder retriever
  index.py        # построение FAISS-индекса корпуса
  evaluate.py     # расчет hit@k, mrr@k, ndcg@k
  calibrate.py    # подбор порога отсечения нерелевантных ответов
  ui.py           # веб-интерфейс и API FastAPI
  answer.py       # CLI для ответа на вопрос
  metrics.py      # чистые функции метрик
tests/
  test_metrics.py
```
