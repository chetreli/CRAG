import streamlit as st

st.set_page_config(page_title="CRAG Assistant", page_icon="🤖", layout="wide")

st.title("🤖 CRAG — корректирующая RAG система")
st.markdown("""
Выбери страницу в боковом меню:
- **💬 Чат** — задавай вопросы по загруженным документам
- **📄 Загрузка документов** — добавь новые документы в базу знаний
""")
