import uuid

import requests
import streamlit as st

from ui.config import API_BASE_URL

st.set_page_config(page_title="Чат — CRAG", page_icon="💬")
st.title("💬 Чат с CRAG")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "meta" in msg:
            meta = msg["meta"]
            badge = {"local": "📚 локальная база", "web": "🌐 веб-поиск", "no_context": "⚠️ без контекста"}
            st.caption(badge.get(meta["source"], meta["source"]))

if prompt := st.chat_input("Задай вопрос..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Думаю..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/chat",
                    json={"message": prompt, "session_id": st.session_state.session_id},
                    timeout=180,
                )
                response.raise_for_status()
                data = response.json()

                st.markdown(data["answer"])
                badge = {"local": "📚 локальная база", "web": "🌐 веб-поиск", "no_context": "⚠️ без контекста"}
                st.caption(badge.get(data["source"], data["source"]))

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": data["answer"],
                    "meta": {"source": data["source"], "used_fallback": data["used_fallback"]},
                })
            except requests.exceptions.RequestException as e:
                st.error(f"Ошибка соединения с API: {e}")

if st.sidebar.button("🗑️ Очистить историю"):
    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())
    st.rerun()
