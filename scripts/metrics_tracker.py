import time

from src.observability.prometheus import start_metrics_server

start_metrics_server(port=8001)
print("Сервер метрик запущен. Нажми Ctrl+C для остановки.")

while True:
    time.sleep(1)
