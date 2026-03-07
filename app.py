import streamlit as st
st.set_page_config(
    page_title="Predictive Maintenance System",
    layout="wide"
)

page1 = st.Page("analysis_and_model.py", title="Анализ и модель")
page2 = st.Page("presentation.py", title="Презентация")

pages = [page1, page2]

current_page = st.navigation(pages, position="sidebar", expanded=True)
current_page.run()

st.sidebar.markdown("---")
st.sidebar.write("**Студент:** Садыков Булат")
st.sidebar.write("**Группа:** 4319")
st.sidebar.write("**Год:** 2026")
