import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt
import time


# Загружаем и обучаем модель + данные
@st.cache_data
def load_data_and_model():
    df = pd.read_csv('data/adult.csv')
    df.replace('?', pd.NA, inplace=True)
    df.dropna(inplace=True)
    df['income'] = df['income'].apply(lambda x: 1 if '>50K' in x else 0)

    X_raw = df.drop('income', axis=1)
    y = df['income']

    X_encoded = pd.get_dummies(X_raw, drop_first=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test, X_raw, scaler, X_encoded.columns.tolist()


# Загружаем модель + данные
model, X_train, X_test, y_train, y_test, X_raw_template, scaler, feature_columns = load_data_and_model()

# Sidebar — выбор страницы
page = st.sidebar.selectbox("Выберите страницу:", ["1. Настройка модели", "2. Предсказание дохода по параметрам"])

# --- Страница 1 ---
if page == "1. Настройка модели":
    st.title("📊 Настройка модели Gradient Boosting + ROC-кривая")

    # Гиперпараметры
    n_estimators = st.slider("Количество деревьев (n_estimators)", min_value=50, max_value=300, step=50, value=100)
    max_depth = st.slider("Максимальная глубина дерева (max_depth)", min_value=1, max_value=10, step=1, value=3)
    learning_rate = st.slider("Learning rate", min_value=0.01, max_value=0.5, step=0.01, value=0.1)

    # Кнопка обучения
    if st.button("Обучить модель"):
        start = time.time()

        model_gb = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42
        )
        model_gb.fit(X_train, y_train)

        duration = time.time() - start

        y_pred = model_gb.predict(X_test)
        y_proba = model_gb.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        st.success("✅ Модель успешно обучена!")

        st.info(f"🕒 Время обучения модели: {duration:.2f} сек")

        col1, col2, col3 = st.columns(3)
        col1.metric("🎯 Accuracy", f"{acc:.4f}")
        col2.metric("📌 F1-score", f"{f1:.4f}")
        col3.metric("📈 ROC AUC", f"{roc_auc:.4f}")

        st.subheader("📈 ROC-кривая модели")
        fig, ax = plt.subplots()
        RocCurveDisplay.from_estimator(model_gb, X_test, y_test, ax=ax)
        st.pyplot(fig)

# --- Страница 2 ---
elif page == "2. Предсказание дохода по параметрам":
    st.title("💼 Предсказание дохода (>50K$) по индивидуальным параметрам")

    # Ввод параметров пользователем
    age = st.slider("Возраст", 18, 90, 30)
    education = st.selectbox("Образование", X_raw_template['education'].unique())
    marital_status = st.selectbox("Семейное положение", X_raw_template['marital.status'].unique())
    occupation = st.selectbox("Профессия", X_raw_template['occupation'].unique())
    hours_per_week = st.slider("Часов работы в неделю", 1, 99, 40)
    sex = st.selectbox("Пол", X_raw_template['sex'].unique())
    native_country = st.selectbox("Страна", X_raw_template['native.country'].unique())

    # Кнопка предсказания
    if st.button("Предсказать доход"):
        # Собираем строку с параметрами
        input_dict = {
            'age': age,
            'education': education,
            'marital.status': marital_status,
            'occupation': occupation,
            'hours.per.week': hours_per_week,
            'sex': sex,
            'native.country': native_country
        }

        # Преобразуем в DataFrame
        input_df = pd.DataFrame([input_dict])

        # One-hot encoding (как было при обучении)
        input_encoded = pd.get_dummies(input_df)
        input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

        # Масштабируем
        input_scaled = scaler.transform(input_encoded)

        # Предсказание
        pred = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0][1]

        # Вывод
        st.subheader("Результат:")
        if pred == 1:
            st.success(f"✅ По заданным параметрам модель прогнозирует доход **> 50K$** (вероятность {proba:.2%})")
        else:
            st.error(f"❌ По заданным параметрам модель прогнозирует доход **<= 50K$** (вероятность {1 - proba:.2%})")
