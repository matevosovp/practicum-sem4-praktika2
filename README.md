# Ecommerce Recommender

Репозиторий содержит законченный проект рекомендательной системы для e-commerce с акцентом на локальную воспроизводимость, `time-based` evaluation и memory-safe preprocessing.

## Задача

Цель системы: рекомендовать товары, которые пользователь с высокой вероятностью добавит в корзину.

Целевой сигнал:

- основной: `addtocart`
- дополнительный сильный: `transaction`
- контекстный: `view`

## Данные

Используются три исходных источника:

- `events.csv` — 2 756 101 событий
- `category_tree.csv` — 1 669 строк иерархии категорий
- `item_properties_part1.csv` + `item_properties_part2.csv` — 20 275 902 строк свойств товаров

Ключевые особенности данных, влияющие на решение:

- 1 407 580 уникальных пользователей
- 235 061 товаров в логах событий
- 96.67% событий — `view`
- медианное число событий на пользователя — `1`
- `transaction_after_cart_rate = 30.71%`
- часть товаров из `events` отсутствует в `item_properties`
- случайный split приводит к temporal leakage

## EDA Summary

EDA оформлен в [notebooks/01_eda.ipynb](/home/what/praktika/practicum-sem4-praktika2/notebooks/01_eda.ipynb) и сводится к следующим инженерным выводам:

- данные крайне разреженные, поэтому dense user-item матрицы и full-join с item properties не подходят
- popularity baseline обязателен, но одного его недостаточно
- короткая история пользователя делает repeat-interest baseline очень сильным
- item-to-item co-occurrence полезен как retrieval-компонент с большей каталоговой широтой
- свойства товаров используются точечно: берутся только последние `categoryid` и `available`
- весь offline-контур строится только на `time-based split`

## Стратегия решения

### Offline setup

- train/validation разбиваются по времени, validation window = `14` дней
- в validation учитываются только будущие `addtocart` и `transaction`
- товары, купленные пользователем в train-окне, исключаются из рекомендаций

Основная метрика:

- `Recall@10`

Дополнительные метрики:

- `HitRate@10`
- `NDCG@10`

### Сравниваемые модели

- `global_popularity`
- `history_baseline`
- `weighted_item2item`
- `hybrid_history_item2item`

### Почему итоговая модель — `history_baseline`

На этом датасете большинство пользователей имеют очень короткую историю, поэтому лучшую offline-метрику дает повторное ранжирование уже проявленного интереса. Более сложный item-to-item retrieval улучшает каталоговое покрытие, но уступает по `Recall@10`. Этот результат оставлен осознанно: итоговый сервис использует реально лучший offline-артефакт, а не номинально более сложную модель.

## Результаты экспериментов

Результаты зафиксированы в [artifacts/reports/evaluation_summary.json](/home/what/praktika/practicum-sem4-praktika2/artifacts/reports/evaluation_summary.json).

| model | Recall@10 | HitRate@10 | NDCG@10 | Catalog coverage |
|---|---:|---:|---:|---:|
| `history_baseline` | 0.1496 | 0.1906 | 0.1320 | 1794 |
| `hybrid_history_item2item` | 0.1377 | 0.1729 | 0.0838 | 3456 |
| `weighted_item2item` | 0.0828 | 0.1120 | 0.0468 | 3310 |
| `global_popularity` | 0.0085 | 0.0216 | 0.0091 | 15 |

Итоговый сериализованный артефакт:

- [models/recommender.joblib](/home/what/praktika/practicum-sem4-praktika2/models/recommender.joblib)

Дополнительный serving-артефакт:

- [models/serving_history.parquet](/home/what/praktika/practicum-sem4-praktika2/models/serving_history.parquet)

## Архитектура проекта

```text
raw csv
  -> src/ecommerce_recsys/data.py
  -> src/ecommerce_recsys/features.py
  -> scripts/run_etl.py / Airflow ETL DAG
  -> Postgres transformed tables
  -> time split + offline evaluation
  -> model selection + MLflow logging
  -> models/recommender.joblib
  -> FastAPI service + Prometheus metrics
  -> Airflow DAG for scheduled retraining
```

Ключевой принцип реализации: сначала агрегация до уровня `user-item`, затем ограниченный candidate generation, без dense-представлений и без полного расплющивания `item_properties`.

## Структура репозитория

```text
app/                     FastAPI сервис и monitoring hooks
airflow/dags/            DAG переобучения
configs/                 пример train-конфига
docs/                    описание мониторинга
models/                  сериализованная модель и serving history
notebooks/               EDA и modeling notebooks
scripts/                 CLI для EDA, train, notebooks и MLflow
src/ecommerce_recsys/    reusable код пайплайна
```

## Установка

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Airflow вынесен в отдельный файл зависимостей, чтобы не утяжелять основной стек:

```bash
pip install -r requirements-airflow.txt
```

## Запуск EDA

```bash
python scripts/run_eda.py
python scripts/build_notebooks.py
```

Основные ноутбуки:

- [notebooks/01_eda.ipynb](/home/what/praktika/practicum-sem4-praktika2/notebooks/01_eda.ipynb)
- [notebooks/02_modeling.ipynb](/home/what/praktika/practicum-sem4-praktika2/notebooks/02_modeling.ipynb)

## Запуск обучения

```bash
python scripts/train_model.py \
  --data-dir . \
  --artifacts-dir artifacts \
  --models-dir models \
  --run-name local-train
```

Промежуточные артефакты сохраняются в parquet:

- `artifacts/data/train_events.parquet`
- `artifacts/data/validation_events.parquet`
- `artifacts/data/aggregated_train.parquet`
- `artifacts/data/evaluation_frame.parquet`
- `artifacts/data/item_snapshot.parquet`

## ETL В Postgres

Для ETL поддерживаются два способа подключения к Postgres:

- Airflow connection через `conn_id`, например `recsys_postgres`
- `.env`-переменные `DB_DESTINATION_HOST`, `DB_DESTINATION_PORT`, `DB_DESTINATION_NAME`, `DB_DESTINATION_USER`, `DB_DESTINATION_PASSWORD`

Если передан `--conn-id` или задан `RECSYS_AIRFLOW_CONN_ID`, скрипты используют Airflow connection. Иначе они продолжают читать `.env`.

В локальном `docker-compose` для Airflow автоматически пробрасывается `conn_id=recsys_postgres` через `AIRFLOW_CONN_RECSYS_POSTGRES`.

Локальный запуск полного ETL-контура:

```bash
python scripts/run_etl.py full \
  --data-dir . \
  --artifacts-dir artifacts \
  --staging-schema recsys_staging \
  --mart-schema recsys_mart \
  --run-id local-etl
```

Пошаговый запуск:

```bash
python scripts/run_etl.py transform --data-dir . --artifacts-dir artifacts --run-id local-transform
python scripts/run_etl.py load --artifacts-dir artifacts --staging-schema recsys_staging --mart-schema recsys_mart
python scripts/run_etl.py validate --artifacts-dir artifacts --staging-schema recsys_staging --mart-schema recsys_mart
```

Запуск через Airflow connection:

```bash
python scripts/run_etl.py load \
  --artifacts-dir artifacts \
  --staging-schema recsys_staging \
  --mart-schema recsys_mart \
  --conn-id recsys_postgres
```

Загружаются только transformed tables, без raw CSV в базе:

- `recsys_staging.events_prepared`
- `recsys_mart.user_item_features`
- `recsys_mart.item_snapshot`
- `recsys_mart.validation_targets`
- `recsys_mart.etl_runs`

Подробное описание слоев есть в [docs/etl.md](/home/what/praktika/practicum-sem4-praktika2/docs/etl.md).

## MLflow

Локальный запуск:

```bash
./scripts/start_mlflow.sh
```

Поддерживаются два режима artifact storage:

- локальный каталог `artifacts/mlruns`
- S3-compatible storage через переменные из `.env` (`MLFLOW_S3_ENDPOINT_URL`, `S3_BUCKET_NAME`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)

Если `MLFLOW_TRACKING_URI` не задан, `scripts/train_model.py` пишет в локальный файл-стор MLflow.

## API

Запуск:

```bash
PYTHONPATH=src uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /health`
- `GET /metrics`
- `POST /recommendations`

Пример запроса:

```json
{
  "user_id": 257597,
  "k": 5,
  "filter_bought": true
}
```

Пример cold-start запроса с историей прямо в payload:

```json
{
  "user_id": null,
  "k": 5,
  "user_history": [
    {"item_id": 355908, "event": "view", "timestamp": 1433221332117}
  ]
}
```

## Docker

Сборка:

```bash
docker build -t ecommerce-recsys .
```

Запуск:

```bash
docker run --rm -p 8000:8000 ecommerce-recsys
```

Docker-образ использует уже сериализованную модель из каталога `models/`.

## Airflow DAG

Как запустить Airflow:

Требование для этого репозитория:

- `docker-compose.yaml` находится в корне репозитория
- код Airflow лежит в каталоге `airflow/`
- папки `airflow/dags/` и `plugins/` монтируются в контейнер Airflow

Подготовка окружения:

```bash
sudo apt-get update
sudo apt-get install python3.10-venv
python3.10 -m venv .venv_project_name

source .venv_project_name/bin/activate

pip install -r requirements.txt
pip install -r requirements-airflow.txt
```

Подготовка переменных окружения:

```bash
# заполните .env_template и переименуйте файл в .env
export $(grep -v '^#' .env | xargs)
```

Запуск Airflow:

```bash
# docker-compose.yaml уже лежит в корне репозитория, дополнительный curl не нужен

# первый запуск из корня репозитория
docker compose up airflow-init

# опционально: полностью очистить локальное состояние контейнеров
# это удалит volumes и кэш, поэтому использовать только для "чистого" старта
docker compose down --volumes --remove-orphans

# основной запуск
docker compose up --build
```

После запуска UI Airflow будет доступен на `http://localhost:8080`.

Файл DAG:

- [airflow/dags/retrain_recommender.py](/home/what/praktika/practicum-sem4-praktika2/airflow/dags/retrain_recommender.py)
- [airflow/dags/recsys_etl_to_postgres.py](/home/what/praktika/practicum-sem4-praktika2/airflow/dags/recsys_etl_to_postgres.py)

Что делает DAG:

- `ecommerce_recommender_etl_to_postgres`:
  - готовит runtime-директории
  - делает local transform из CSV в parquet
  - загружает transformed tables в Postgres
  - валидирует, что таблицы доступны
- `ecommerce_recommender_retrain`:
  - запускает обучение модели
  - пишет метрики в MLflow
  - сохраняет reporting-артефакт с ключевыми метриками для последующего анализа
  - публикует итоговые метрики и реестр ключевых артефактов в Postgres

Для локального теста:

```bash
export RECSYS_PROJECT_ROOT=$(pwd)
export RECSYS_PYTHON_BIN=$(which python)
airflow dags test ecommerce_recommender_etl_to_postgres 2024-01-08
airflow dags test ecommerce_recommender_retrain 2024-01-08
```

## Мониторинг

Prometheus-compatible метрики экспортируются прямо из сервиса.

Основные метрики:

- `recsys_api_requests_total`
- `recsys_api_recommendation_latency_seconds`
- `recsys_api_fallback_total`

Подробности описаны в [docs/monitoring.md](/home/what/praktika/practicum-sem4-praktika2/docs/monitoring.md).

## Ограничения решения

- текущая финальная модель сильнее всего на repeat-interest сценариях, а не на discovery
- item features используются только в lightweight-виде (`available`, `categoryid`)
- online feature store не используется; сервис работает на сериализованном offline snapshot
- ETL загружает только transformed tables, а не raw-слой
- Airflow dependencies вынесены отдельно, чтобы не утяжелять основной runtime сервиса

## Воспроизводимость

- все версии зависимостей зафиксированы в `requirements.txt`
- `time-based split` жестко зашит в pipeline
- промежуточные артефакты сохраняются на диск
- модель и serving history сериализуются отдельно
- notebooks строятся из уже рассчитанных артефактов, без обязательного повторного тяжелого preprocessing при открытии
- Airflow не генерирует notebooks; вместо этого он сохраняет [artifacts/reports/reporting_metrics.json](/home/what/praktika/practicum-sem4-praktika2/artifacts/reports/reporting_metrics.json) как компактный слой метрик для последующего анализа
- в конце retrain-контура метрики и registry артефактов также записываются в таблицы `recsys_mart.reporting_runs`, `recsys_mart.model_metrics_history`, `recsys_mart.artifact_registry`
