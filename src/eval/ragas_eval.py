import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage

from src.agent.graph import build_graph
from src.config.setting import settings


def load_eval_dataset(path: str = "src/eval/datasets/eval_dataset.json") -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_pipeline_for_eval(graph, questions: list[str]) -> tuple[list[str], list[list[str]]]:
    answers = []
    contexts = []

    for q in questions:
        print(f"  Обрабатываю: {q}")
        result = graph.invoke({
            "query": q,
            "chunks": [], "relevant_chunks": [], "irrelevant_chunks": [],
            "rewritten_query": None, "rewrite_attempts": 0,
            "web_results": [], "used_fallback": False,
            "answer": "", "source": "",
            "messages": [HumanMessage(content=q)],
            "_trace_id": None,
        })

        answers.append(result["answer"])
        ctx = [c.text for c in result.get("relevant_chunks", [])]
        contexts.append(ctx if ctx else ["нет контекста"])

    return answers, contexts


def run_ragas_evaluation(model, client):
    eval_data = load_eval_dataset()
    questions = [item["question"] for item in eval_data]
    ground_truths = [item["ground_truth"] for item in eval_data]

    print(f"Прогоняем {len(questions)} вопросов через CRAG pipeline...")
    graph = build_graph(model, client)
    answers, contexts = run_pipeline_for_eval(graph, questions)

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    print("Запускаем RAGAS оценку (это займёт время)...")

    ragas_llm = LangchainLLMWrapper(ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=0.0,
    ))
    ragas_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
    ))

    # Увеличенный таймаут и последовательное выполнение — Ollama не умеет
    # параллельно обрабатывать несколько запросов на одной локальной модели
    run_config = RunConfig(
        timeout=300,
        max_workers=1,
    )

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=run_config,
    )

    print("\n=== RAGAS результаты по каждому вопросу ===")
    df = result.to_pandas()
    print("Колонки в результате:", df.columns.tolist())

    available_cols = [c for c in ["user_input", "faithfulness", "answer_relevancy", "context_precision", "context_recall"] if c in df.columns]
    print(df[available_cols].to_string())

    metric_cols = [c for c in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"] if c in df.columns]
    avg = df[metric_cols].mean()
    print("\n=== Средние метрики ===")
    print(avg)

    return result, df