import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

from src.crag_core.query_rewriter import rewrite_query, rewrite_query_multiple

queries = [
    "что такое БЖД",
    "расскажи про экологию",
    "опасные штуки на производстве",
]

for q in queries:
    print(f"\nОригинал:       {q}")
    print(f"Переформулировка: {rewrite_query(q)}")

print("\n--- Множественные варианты ---")
variants = rewrite_query_multiple("пожарная безопасность", n=3)
for i, v in enumerate(variants, 1):
    print(f"{i}. {v}")