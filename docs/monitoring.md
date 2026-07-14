# Monitoring

Документ описывает минимально разумный monitoring-контур для текущего FastAPI-сервиса. Он разделён на два слоя:

- online health: доступность, latency и fallback-поведение API
- offline quality control: метрики моделей, артефакты retrain-контура и признаки деградации recommendation quality

## Что отдаёт сервис

Prometheus-compatible метрики доступны на `GET /metrics`.

Текущие метрики:

- `recsys_api_requests_total{endpoint, status}`: счетчик запросов с разбиением по endpoint и HTTP status
- `recsys_api_recommendation_latency_seconds_bucket|count|sum`: histogram времени ответа для `/recommendations`
- `recsys_api_fallback_total`: число recommendation-запросов, где пришлось использовать fallback-логику

Endpoint `GET /health` нужен для liveness/readiness-проверок и возвращает текущий `model_name`.

## Как читать `/metrics`

Несколько практических правил интерпретации:

- `recsys_api_requests_total` лучше читать как rate, а не как абсолютный counter
- histogram latency анализируется через перцентили, а не по `sum/count` в изоляции
- `recsys_api_fallback_total` нужно соотносить только с трафиком на `/recommendations`

Примеры PromQL-запросов:

```promql
sum(rate(recsys_api_requests_total{endpoint="/recommendations"}[5m]))
```

Трафик на рекомендации за последние 5 минут.

```promql
sum(rate(recsys_api_requests_total{status=~"5.."}[5m]))
/
sum(rate(recsys_api_requests_total[5m]))
```

Доля ошибок уровня `5xx`.

```promql
histogram_quantile(
  0.95,
  sum(rate(recsys_api_recommendation_latency_seconds_bucket[5m])) by (le)
)
```

P95 latency для `/recommendations`.

```promql
sum(rate(recsys_api_fallback_total[5m]))
/
sum(rate(recsys_api_requests_total{endpoint="/recommendations",status=~"2.."}[5m]))
```

Доля успешных recommendation-запросов, завершившихся через fallback.

## Что смотреть на дашборде

Минимальный dashboard стоит собрать из следующих панелей:

- Request rate: общий поток запросов и отдельно поток `/recommendations`
- Error rate: доля `5xx` и `4xx`, отдельно для recommendation endpoint
- Latency: p50, p95 и p99 по histogram buckets
- Fallback share: отношение `fallback_total` к успешным recommendation-запросам
- Health status: доступность `/health` и совпадение ожидаемого `model_name` с текущим rollout

Практическая интерпретация:

- рост трафика без роста latency означает, что сервис масштабируется нормально
- рост fallback share при стабильном трафике обычно указывает не на инфраструктуру, а на качество входного user context
- рост `5xx` при стабильном fallback чаще говорит о runtime-ошибках или проблемах с model artifact

## Какие сигналы говорят о деградации качества

Сервис не меряет online recommendation quality напрямую, поэтому quality drift собирается из косвенных признаков:

- fallback share растёт относительно базового уровня
- после retrain ухудшается `Recall@10` или `NDCG@10` в `artifacts/reports/evaluation_summary.json`
- MLflow показывает последовательное падение offline quality между запусками
- свежий retrain резко увеличивает catalog coverage при заметном падении recall: это сигнал не улучшения discovery, а вероятного размытия релевантности
- новый model artifact перестаёт обновляться по расписанию Airflow DAG

Для этого проекта особенно важно следить за двумя сценариями:

- quality regression: recall падает, хотя сервис технически жив
- context regression: сервис отвечает быстро, но чаще проваливается в fallback из-за короткой или отсутствующей истории

## Offline quality artifacts

Ключевые точки контроля качества находятся вне online path:

- [artifacts/reports/evaluation_summary.json](../artifacts/reports/evaluation_summary.json)
- [artifacts/reports/modeling_notebook_report.json](../artifacts/reports/modeling_notebook_report.json)
- MLflow experiment history
- Airflow logs и статус DAG-задач retrain / ETL

Если online-метрики выглядят нормально, а бизнес-качество ухудшается, первую проверку стоит начинать именно с этих артефактов.

## Разумные алерты

Базовый набор:

- `5xx` error rate выше `2%` в течение `10m`
- `p95` latency выше `500ms` в течение `10m`
- fallback share заметно выше типичного baseline проекта
- `/health` недоступен или возвращает неожиданный `model_name`
- retrain DAG не завершался успешно дольше ожидаемого окна
- новый retrain завершился, но offline quality в `evaluation_summary.json` ухудшилась сильнее допустимого порога

Пример operational alert:

- сервис жив, но `fallback share` вырос кратно при стабильном трафике и latency

Это типичный индикатор того, что recommendation path деградировал содержательно, даже если инфраструктура формально здорова.

## Что делать при инциденте

Короткий порядок проверки:

1. Проверить `/health` и базовую доступность `/recommendations`.
2. Посмотреть request rate, error rate и p95/p99 latency.
3. Сравнить текущий fallback share с обычным уровнем.
4. Проверить свежесть model artifact и статус последнего Airflow retrain.
5. Сверить offline metrics в `evaluation_summary.json` и MLflow с предыдущим стабильным запуском.

Такой порядок позволяет быстро отделить инфраструктурную проблему от деградации самой recommendation-логики.
