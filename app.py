import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Random Forest Regression App",
    page_icon="🌲",
    layout="wide"
)

st.title("🌲 Random Forest Regression – Salary Prediction")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "📂 Upload CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        # Load data
        df = pd.read_csv(uploaded_file)
        st.success("✅ Dataset loaded successfully")

        st.subheader("📊 Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

        # ---------------- TARGET SELECTION ----------------
        st.subheader("🎯 Select Target Column")
        target_col = st.selectbox("Choose target column", df.columns)

        X = df.drop(columns=[target_col])
        y = df[target_col]

        # ---------------- TRAIN TEST SPLIT ----------------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # ---------------- MODEL TRAINING ----------------
        st.subheader("⚙️ Model Training")

        n_estimators = st.slider(
            "Number of Trees (n_estimators)",
            min_value=50,
            max_value=300,
            value=100,
            step=50
        )

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=42
        )

        model.fit(X_train, y_train)
        st.success("🌳 Random Forest model trained successfully")

        # ---------------- PREDICTIONS ----------------
        y_pred = model.predict(X_test)

        # ---------------- EVALUATION ----------------
        st.subheader("📈 Model Performance")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.2f}")
        col2.metric("MSE", f"{mean_squared_error(y_test, y_pred):.2f}")
        col3.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
        col4.metric("R² Score", f"{r2_score(y_test, y_pred):.2f}")

        # ---------------- FEATURE IMPORTANCE ----------------
        st.subheader("⭐ Feature Importance")

        importance_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        st.dataframe(importance_df, use_container_width=True)

        # ---------------- MANUAL PREDICTION ----------------
        st.subheader("🧮 Predict Using Custom Input")

        user_input = {}

        for col in X.columns:
            user_input[col] = st.number_input(
                f"Enter {col}",
                value=float(X[col].mean())
            )

        input_df = pd.DataFrame([user_input])

        if st.button("🔮 Predict"):
            prediction = model.predict(input_df)
            st.success(f"💰 Predicted Value: {prediction[0]:.2f}")

    except Exception as e:
        st.error("❌ Error occurred while processing the file")
        st.exception(e)

else:
    st.info("👆 Please upload a CSV file to continue")
