import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import matplotlib.pyplot as plt
import pandas as pd

st.title("🧠 Сравнение моделей машинного обучения")

# Данные
data = fetch_california_housing(as_frame=True)
X, y = data.data, data.target
st.write("## 📊 Данные")
st.dataframe(X.head())

# Выбор моделей
st.sidebar.header("🧩 Выбор моделей")
models_to_train = st.sidebar.multiselect(
    "Выберите модели",
    ["Decision Tree", "Random Forest", "Gradient Boosting"],
    default=["Decision Tree"]
)

# Гиперпараметры (для дерева)
st.sidebar.header("⚙️ Гиперпараметры (для дерева решений)")
max_depth = st.sidebar.slider("Макс. глубина", 1, 20, 5)

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

results = {}

for name in models_to_train:
    if name == "Decision Tree":
        model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    elif name == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif name == "Gradient Boosting":
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        "mse": mean_squared_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
        "pred": y_pred
    }

# Отображение метрик
st.write("## 📈 Результаты")
for name, res in results.items():
    st.subheader(f"🔹 {name}")
    st.write(f"**MSE**: {res['mse']:.3f} | **R²**: {res['r2']:.3f}")
    fig, ax = plt.subplots()
    ax.scatter(y_test, res['pred'], alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red')
    ax.set_title(f"{name}: Предсказания vs Истина")
    st.pyplot(fig)
