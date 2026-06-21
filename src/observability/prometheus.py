from prometheus_client import Counter, Histogram, Gauge, start_http_server


# Счётчики запросов
crag_requests_total = Counter(
    "crag_requests_total", "Всего запросов к CRAG", ["source"]
)
crag_fallback_total = Counter(
    "crag_fallback_total", "Количество fallback на web search"
)

# Латентность по этапам
crag_retrieve_duration = Histogram(
    "crag_retrieve_duration_seconds", "Время retrieval"
)
crag_grade_duration = Histogram(
    "crag_grade_duration_seconds", "Время grading"
)
crag_generate_duration = Histogram(
    "crag_generate_duration_seconds", "Время генерации ответа"
)

# Качество retrieval
crag_relevant_chunks_ratio = Gauge(
    "crag_relevant_chunks_ratio", "Доля релевантных чанков от общего числа"
)


def start_metrics_server(port: int = 8001):
    start_http_server(port)
    print(f"Prometheus метрики доступны на http://localhost:{port}/metrics")