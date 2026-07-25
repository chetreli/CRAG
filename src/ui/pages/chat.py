import json
import uuid

import requests
import streamlit as st

from src.ui.config import API_BASE_URL

st.set_page_config(page_title="Чат — CRAG", page_icon="💬")
st.title("💬 Чат с CRAG")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# Отображаем историю
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "meta" in msg:
            badge = {
                "local": "📚 локальная база",
                "web": "🌐 веб-поиск",
                "no_context": "⚠️ без контекста",
            }
            st.caption(badge.get(msg["meta"]["source"], msg["meta"]["source"]))

if prompt := st.chat_input("Задай вопрос..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        answer_placeholder = st.empty()  # ← используем для показа во время загрузки

        try:
            with requests.post(
                f"{API_BASE_URL}/chat/stream",
                json={"message": prompt, "session_id": st.session_state.session_id},
                stream=True,
                timeout=300,
            ) as response:
                response.raise_for_status()
                final_data = None

                for line in response.iter_lines():
                    if not line:
                        continue
                    line = line.decode("utf-8")
                    if not line.startswith("data: "):
                        continue

                    data = json.loads(line[6:])
                    progress_bar.progress(data.get("progress", 0))
                    status_text.markdown(f"_{data.get('message', '')}_")

                    if data.get("stage") == "done":
                        final_data = data

                if final_data:
                    answer = final_data["answer"]
                    source = final_data["source"]
                    used_fallback = final_data["used_fallback"]

                    # Очищаем прогрессбар и статус
                    progress_bar.empty()
                    status_text.empty()
                    answer_placeholder.empty()  # ← очищаем placeholder

                    # Сохраняем в историю
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "meta": {"source": source, "used_fallback": used_fallback},
                    })

                    # Перерисовываем страницу — ответ появится из истории
                    st.rerun()

        except requests.exceptions.RequestException as e:
            progress_bar.empty()
            status_text.empty()
            answer_placeholder.empty()
            st.error(f"Ошибка соединения с API: {e}")

if st.sidebar.button("🗑️ Очистить историю"):
    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())
    st.rerun()
