import streamlit as st

# Настройка страницы
st.set_page_config(
    page_title="Predictive Maintenance System",
    layout="wide"
)

# Создание страниц
page1 = st.Page("analysis_and_model.py", title="Анализ и модель")
page2 = st.Page("presentation.py", title="Презентация")

# Настройка навигации
pages = [page1, page2]

# Отображение навигации
current_page = st.navigation(pages, position="sidebar", expanded=True)
current_page.run()

# Информация о студенте
st.sidebar.markdown("---")
st.sidebar.write("**Студент:** Садыков Булат")
st.sidebar.write("**Группа:** 4319")
st.sidebar.write("**Год:** 2026")