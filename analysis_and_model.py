import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_curve,
                             roc_auc_score, precision_recall_curve)
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# Настройка страницы
st.set_page_config(page_title="Анализ и модель", layout="wide")


def load_data(uploaded_file=None):
    """Загрузка данных из файла или через ucimlrepo"""
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        try:
            from ucimlrepo import fetch_ucirepo
            with st.spinner('Загрузка данных из UCI ML Repository...'):
                dataset = fetch_ucirepo(id=601)
                data = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
            st.success('Данные успешно загружены из репозитория!')
        except Exception as e:
            st.error(f"Ошибка загрузки данных: {e}")
            # Создаем демо-данные для тестирования
            data = create_sample_data()
            st.warning("Используются демо-данные для тестирования")
    return data


def create_sample_data():
    """Создание демо-данных для тестирования"""
    np.random.seed(42)
    n_samples = 1000

    data = pd.DataFrame({
        'Type': np.random.choice(['L', 'M', 'H'], n_samples),
        'Air temperature [K]': np.random.normal(300, 2, n_samples),
        'Process temperature [K]': np.random.normal(310, 1, n_samples),
        'Rotational speed [rpm]': np.random.normal(1500, 100, n_samples),
        'Torque [Nm]': np.random.normal(40, 10, n_samples),
        'Tool wear [min]': np.random.uniform(0, 250, n_samples),
        'Machine failure': np.random.choice([0, 1], n_samples, p=[0.97, 0.03])
    })
    return data


def preprocess_data(data):
    """Предобработка данных"""
    # Копируем данные
    df = data.copy()

    # Удаление ненужных столбцов
    columns_to_drop = ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    existing_columns = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=existing_columns)

    # Преобразование категориальной переменной Type
    if 'Type' in df.columns:
        le = LabelEncoder()
        df['Type'] = le.fit_transform(df['Type'])
        st.session_state['label_encoder'] = le

    # Проверка на пропущенные значения
    missing_values = df.isnull().sum()

    return df, missing_values


def scale_features(X_train, X_test, numerical_features):
    """Масштабирование числовых признаков"""
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])

    return X_train_scaled, X_test_scaled, scaler


def train_models(X_train, y_train):
    """Обучение различных моделей"""
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42, probability=True)
    }

    trained_models = {}
    for name, model in models.items():
        with st.spinner(f'Обучение модели {name}...'):
            model.fit(X_train, y_train)
            trained_models[name] = model

    return trained_models


def evaluate_model(model, X_test, y_test, model_name):
    """Оценка модели"""
    # Предсказания
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Метрики
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Classification report в виде словаря
    report = classification_report(y_test, y_pred, output_dict=True)

    return {
        'name': model_name,
        'accuracy': accuracy,
        'conf_matrix': conf_matrix,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'report': report
    }


def plot_roc_curves(results, y_test):
    """Построение ROC-кривых для всех моделей"""
    fig = go.Figure()

    colors = ['blue', 'green', 'red', 'purple']
    for i, result in enumerate(results):
        fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f"{result['name']} (AUC = {result['roc_auc']:.3f})",
            line=dict(color=colors[i], width=2)
        ))

    # Добавляем диагональную линию
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Guess',
        line=dict(color='gray', width=2, dash='dash')
    ))

    fig.update_layout(
        title='ROC-кривые для всех моделей',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        showlegend=True,
        width=800,
        height=600
    )

    return fig


def plot_confusion_matrix(conf_matrix, model_name):
    """Построение матрицы ошибок"""
    fig = px.imshow(
        conf_matrix,
        text_auto=True,
        color_continuous_scale='Blues',
        labels=dict(x="Предсказанный класс", y="Истинный класс", color="Количество"),
        x=['Нет отказа (0)', 'Отказ (1)'],
        y=['Нет отказа (0)', 'Отказ (1)']
    )
    fig.update_layout(title=f'Матрица ошибок - {model_name}')
    return fig


def plot_feature_importance(model, feature_names, model_name):
    """Визуализация важности признаков"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        fig = go.Figure(data=[
            go.Bar(
                x=importances[indices][:10],
                y=[feature_names[i] for i in indices[:10]],
                orientation='h'
            )
        ])

        fig.update_layout(
            title=f'Важность признаков - {model_name}',
            xaxis_title='Важность',
            yaxis_title='Признаки',
            height=400
        )
        return fig
    return None


def main():
    st.title("Анализ данных и обучение модели")
    st.markdown("---")

    # Инициализация session state
    if 'models_trained' not in st.session_state:
        st.session_state['models_trained'] = False
    if 'data_loaded' not in st.session_state:
        st.session_state['data_loaded'] = False

    # Создание вкладок
    tab1, tab2, tab3, tab4 = st.tabs([
        "Загрузка данных",
        "Предобработка",
        "Обучение модели",
        "Оценка и предсказания"
    ])

    with tab1:
        st.header("Загрузка данных")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Вариант 1: Загрузка CSV файла")
            uploaded_file = st.file_uploader(
                "Выберите CSV файл",
                type="csv",
                help="Загрузите файл predictive_maintenance.csv"
            )

            if uploaded_file is not None:
                st.session_state['data'] = load_data(uploaded_file)
                st.session_state['data_loaded'] = True
                st.success("Данные успешно загружены из файла!")

        with col2:
            st.subheader("Вариант 2: Загрузка из UCI Repository")
            if st.button("Загрузить данные из репозитория", use_container_width=True):
                st.session_state['data'] = load_data()
                st.session_state['data_loaded'] = True

        # Отображение информации о данных
        if st.session_state['data_loaded']:
            st.markdown("---")
            st.subheader("Просмотр данных")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Количество записей", st.session_state['data'].shape[0])
            with col2:
                st.metric("Количество признаков", st.session_state['data'].shape[1])
            with col3:
                failure_rate = st.session_state['data']['Machine failure'].mean() * 100
                st.metric("Доля отказов", f"{failure_rate:.2f}%")

            st.dataframe(st.session_state['data'].head(10), use_container_width=True)

            # Статистика по данным
            with st.expander("Статистика по данным"):
                st.dataframe(st.session_state['data'].describe(), use_container_width=True)

    with tab2:
        st.header("Предобработка данных")

        if not st.session_state['data_loaded']:
            st.warning("Сначала загрузите данные во вкладке 'Загрузка данных'")
            return

        # Предобработка данных
        data = st.session_state['data']

        st.subheader("1. Информация о данных до предобработки")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Типы данных:**")
            dtypes_df = pd.DataFrame({
                'Столбец': data.dtypes.index,
                'Тип данных': data.dtypes.values
            })
            st.dataframe(dtypes_df, use_container_width=True)

        with col2:
            st.write("**Пропущенные значения:**")
            missing_df = pd.DataFrame({
                'Столбец': data.columns,
                'Пропуски': data.isnull().sum().values,
                'Процент': (data.isnull().sum() / len(data) * 100).values
            })
            st.dataframe(missing_df, use_container_width=True)

        # Применение предобработки
        st.subheader("2. Применение предобработки")

        col1, col2 = st.columns(2)

        with col1:
            remove_id = st.checkbox("Удалить идентификаторы (UDI, Product ID)", value=True)
            encode_type = st.checkbox("Преобразовать Type в числовой формат", value=True)

        with col2:
            scale_numerical = st.checkbox("Масштабировать числовые признаки", value=True)
            handle_missing = st.checkbox("Обработать пропущенные значения", value=True)

        if st.button("Применить предобработку", type="primary", use_container_width=True):
            with st.spinner("Выполняется предобработка данных..."):
                # Предобработка
                df_processed, missing_values = preprocess_data(data)

                # Сохраняем обработанные данные
                st.session_state['data_processed'] = df_processed

                # Определяем числовые признаки для масштабирования
                numerical_features = ['Air temperature [K]', 'Process temperature [K]',
                                      'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
                existing_num_features = [f for f in numerical_features if f in df_processed.columns]

                # Разделение на признаки и целевую переменную
                X = df_processed.drop(columns=['Machine failure'])
                y = df_processed['Machine failure']

                # Разделение на обучающую и тестовую выборки
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )

                # Масштабирование
                if scale_numerical and existing_num_features:
                    X_train_scaled, X_test_scaled, scaler = scale_features(
                        X_train, X_test, existing_num_features
                    )
                    st.session_state['X_train'] = X_train_scaled
                    st.session_state['X_test'] = X_test_scaled
                    st.session_state['scaler'] = scaler
                else:
                    st.session_state['X_train'] = X_train
                    st.session_state['X_test'] = X_test

                st.session_state['y_train'] = y_train
                st.session_state['y_test'] = y_test
                st.session_state['feature_names'] = X.columns.tolist()
                st.session_state['data_preprocessed'] = True

                st.success("Предобработка данных завершена!")

        # Отображение результата предобработки
        if 'data_preprocessed' in st.session_state and st.session_state['data_preprocessed']:
            st.markdown("---")
            st.subheader("3. Результат предобработки")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Обучающая выборка:**")
                st.write(f"Количество образцов: {len(st.session_state['X_train'])}")
                st.write(f"Количество признаков: {st.session_state['X_train'].shape[1]}")

            with col2:
                st.write("**Тестовая выборка:**")
                st.write(f"Количество образцов: {len(st.session_state['X_test'])}")
                st.write(f"Количество признаков: {st.session_state['X_test'].shape[1]}")

            st.write("**Первые строки обработанных данных:**")
            st.dataframe(st.session_state['X_train'].head(), use_container_width=True)

    with tab3:
        st.header("Обучение моделей")

        if 'data_preprocessed' not in st.session_state:
            st.warning("Сначала выполните предобработку данных во вкладке 'Предобработка'")
            return

        st.subheader("Выбор моделей для обучения")

        col1, col2 = st.columns(2)

        with col1:
            use_lr = st.checkbox("Logistic Regression", value=True)
            use_rf = st.checkbox("Random Forest", value=True)

        with col2:
            use_xgb = st.checkbox("XGBoost", value=True)
            use_svm = st.checkbox("SVM", value=False)

        if st.button("Обучить выбранные модели", type="primary", use_container_width=True):
            # Собираем выбранные модели
            models_to_train = {}
            if use_lr:
                models_to_train['Logistic Regression'] = LogisticRegression(random_state=42, max_iter=1000)
            if use_rf:
                models_to_train['Random Forest'] = RandomForestClassifier(n_estimators=100, random_state=42)
            if use_xgb:
                models_to_train['XGBoost'] = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
            if use_svm:
                models_to_train['SVM'] = SVC(kernel='rbf', random_state=42, probability=True)

            # Обучение моделей
            trained_models = {}
            results = []

            progress_bar = st.progress(0)
            for i, (name, model) in enumerate(models_to_train.items()):
                with st.spinner(f'Обучение модели {name}...'):
                    model.fit(st.session_state['X_train'], st.session_state['y_train'])
                    trained_models[name] = model

                    # Оценка модели
                    result = evaluate_model(
                        model,
                        st.session_state['X_test'],
                        st.session_state['y_test'],
                        name
                    )
                    results.append(result)

                    progress_bar.progress((i + 1) / len(models_to_train))

            st.session_state['trained_models'] = trained_models
            st.session_state['results'] = results
            st.session_state['models_trained'] = True

            st.success(f"Обучено {len(trained_models)} моделей!")

            # Отображение результатов обучения
            st.subheader("Результаты обучения")

            results_df = pd.DataFrame([
                {
                    'Модель': r['name'],
                    'Accuracy': f"{r['accuracy']:.4f}",
                    'ROC-AUC': f"{r['roc_auc']:.4f}"
                }
                for r in results
            ])

            st.dataframe(results_df, use_container_width=True)

    with tab4:
        st.header("Оценка моделей и предсказания")

        if 'models_trained' not in st.session_state:
            st.warning("Сначала обучите модели во вкладке 'Обучение модели'")
            return

        # Создание подвкладок
        eval_tab1, eval_tab2, eval_tab3 = st.tabs([
            "Сравнение моделей",
            "Детальный анализ",
            "Предсказание на новых данных"
        ])

        with eval_tab1:
            st.subheader("Сравнение всех моделей")

            # ROC-кривые
            fig_roc = plot_roc_curves(
                st.session_state['results'],
                st.session_state['y_test']
            )
            st.plotly_chart(fig_roc, use_container_width=True)

            # Сравнение метрик
            st.subheader("Сравнение метрик")

            comparison_data = []
            for result in st.session_state['results']:
                comparison_data.append({
                    'Модель': result['name'],
                    'Accuracy': result['accuracy'],
                    'ROC-AUC': result['roc_auc'],
                    'Precision (0)': result['report']['0']['precision'],
                    'Recall (0)': result['report']['0']['recall'],
                    'F1-score (0)': result['report']['0']['f1-score'],
                    'Precision (1)': result['report']['1']['precision'],
                    'Recall (1)': result['report']['1']['recall'],
                    'F1-score (1)': result['report']['1']['f1-score']
                })

            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df.style.highlight_max(axis=0), use_container_width=True)

        with eval_tab2:
            st.subheader("Детальный анализ моделей")

            # Выбор модели для детального анализа
            model_names = [r['name'] for r in st.session_state['results']]
            selected_model = st.selectbox("Выберите модель для анализа", model_names)

            # Находим результаты выбранной модели
            selected_result = next(
                r for r in st.session_state['results']
                if r['name'] == selected_model
            )
            selected_model_obj = st.session_state['trained_models'][selected_model]

            col1, col2 = st.columns(2)

            with col1:
                # Матрица ошибок
                fig_cm = plot_confusion_matrix(
                    selected_result['conf_matrix'],
                    selected_model
                )
                st.plotly_chart(fig_cm, use_container_width=True)

            with col2:
                # Важность признаков (для моделей, где это доступно)
                fig_importance = plot_feature_importance(
                    selected_model_obj,
                    st.session_state['feature_names'],
                    selected_model
                )
                if fig_importance:
                    st.plotly_chart(fig_importance, use_container_width=True)
                else:
                    st.info(f"Модель {selected_model} не поддерживает визуализацию важности признаков")

            # Classification Report
            st.subheader("Classification Report")
            report_df = pd.DataFrame(selected_result['report']).transpose()
            st.dataframe(report_df, use_container_width=True)

        with eval_tab3:
            st.subheader("Предсказание для новых данных")

            st.info("Введите параметры оборудования для предсказания вероятности отказа")

            # Создание формы для ввода данных
            with st.form("prediction_form"):
                col1, col2 = st.columns(2)

                with col1:
                    product_type = st.selectbox(
                        "Тип продукта",
                        options=['L', 'M', 'H'],
                        help="L - Low, M - Medium, H - High"
                    )
                    air_temp = st.number_input(
                        "Температура воздуха (K)",
                        min_value=280.0,
                        max_value=320.0,
                        value=300.0,
                        step=0.1
                    )
                    process_temp = st.number_input(
                        "Температура процесса (K)",
                        min_value=290.0,
                        max_value=330.0,
                        value=310.0,
                        step=0.1
                    )

                with col2:
                    rotational_speed = st.number_input(
                        "Скорость вращения (rpm)",
                        min_value=1000,
                        max_value=2000,
                        value=1500,
                        step=10
                    )
                    torque = st.number_input(
                        "Крутящий момент (Nm)",
                        min_value=10.0,
                        max_value=70.0,
                        value=40.0,
                        step=0.1
                    )
                    tool_wear = st.number_input(
                        "Износ инструмента (min)",
                        min_value=0,
                        max_value=300,
                        value=150,
                        step=1
                    )

                # Выбор модели для предсказания
                model_for_prediction = st.selectbox(
                    "Выберите модель для предсказания",
                    options=model_names,
                    index=0
                )

                submit_button = st.form_submit_button(
                    "Предсказать",
                    type="primary",
                    use_container_width=True
                )

            if submit_button:
                # Создание DataFrame с входными данными
                input_data = pd.DataFrame({
                    'Type': [product_type],
                    'Air temperature [K]': [air_temp],
                    'Process temperature [K]': [process_temp],
                    'Rotational speed [rpm]': [rotational_speed],
                    'Torque [Nm]': [torque],
                    'Tool wear [min]': [tool_wear]
                })

                # Преобразование типа продукта
                if 'label_encoder' in st.session_state:
                    input_data['Type'] = st.session_state['label_encoder'].transform(input_data['Type'])

                # Масштабирование (если применялось)
                if 'scaler' in st.session_state:
                    numerical_features = ['Air temperature [K]', 'Process temperature [K]',
                                          'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
                    input_data[numerical_features] = st.session_state['scaler'].transform(
                        input_data[numerical_features]
                    )

                # Получение модели
                model = st.session_state['trained_models'][model_for_prediction]

                # Предсказание
                prediction = model.predict(input_data)[0]
                prediction_proba = model.predict_proba(input_data)[0]

                # Отображение результатов
                st.markdown("---")
                st.subheader("Результат предсказания")

                col1, col2, col3 = st.columns(3)

                with col1:
                    if prediction == 0:
                        st.success("Отказ НЕ произойдет")
                    else:
                        st.error("Отказ ПРОИЗОЙДЕТ")

                with col2:
                    st.metric(
                        "Вероятность отказа",
                        f"{prediction_proba[1]:.2%}"
                    )

                with col3:
                    st.metric(
                        "Вероятность нормальной работы",
                        f"{prediction_proba[0]:.2%}"
                    )

                # Визуализация вероятностей
                fig = go.Figure(data=[
                    go.Bar(
                        x=['Нормальная работа', 'Отказ'],
                        y=[prediction_proba[0], prediction_proba[1]],
                        marker_color=['green', 'red'],
                        text=[f'{prediction_proba[0]:.1%}', f'{prediction_proba[1]:.1%}'],
                        textposition='auto',
                    )
                ])

                fig.update_layout(
                    title='Вероятности предсказания',
                    yaxis_title='Вероятность',
                    yaxis=dict(tickformat='.0%'),
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()