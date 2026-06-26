from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver


def get_memory() -> MemorySaver:
    """In-memory checkpointer для хранения истории диалога."""
    return MemorySaver()


def format_history(messages: list) -> str:
    """Форматирует историю сообщений в текст для контекста."""
    parts = []
    for msg in messages[-6:]:  # последние 3 обмена
        if isinstance(msg, HumanMessage):
            parts.append(f"Пользователь: {msg.content}")
        elif isinstance(msg, AIMessage):
            parts.append(f"Ассистент: {msg.content}")
    return "\n".join(parts)
