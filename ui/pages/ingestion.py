import requests
import streamlit as st

from ui.config import API_BASE_URL

st.set_page_config(page_title="Загрузка документов — CRAG", page_icon="📄")
st.title("📄 Загрузка документов в базу знаний")

st.markdown("Поддерживаемые форматы: **PDF**, **TXT**, **MD**")

uploaded_files = st.file_uploader(
    "Выбери файлы для загрузки",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True,
)

if uploaded_files and st.button("📤 Загрузить и проиндексировать"):
    progress = st.progress(0)
    results = []

    for i, file in enumerate(uploaded_files):
        with st.spinner(f"Обрабатываю {file.name}..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/ingest",
                    files={"file": (file.name, file.getvalue())},
                    timeout=300,
                )
                response.raise_for_status()
                data = response.json()
                results.append((file.name, "✅", data["chunks_indexed"]))
            except requests.exceptions.RequestException as e:
                results.append((file.name, "❌", str(e)))

        progress.progress((i + 1) / len(uploaded_files))

    st.subheader("Результаты")
    for name, status, info in results:
        st.write(f"{status} **{name}** — {info}")
