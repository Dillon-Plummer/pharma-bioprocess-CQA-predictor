# app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import the necessary functions
from synthetic_data import generate_synthetic_data
from feature_engineering import create_tabular_features

# --- Page Configuration ---
st.set_page_config(page_title="Bioprocess CQA Predictor", page_icon="🔬", layout="wide")

# --- Model Loading ---
@st.cache_resource
def load_all_models():
    models = {}
    model_files = {'gb': 'gb_model.pkl', 'rf': 'rf_model.pkl', 'xgb': 'xgb_model.pkl', 'lasso': 'lasso_model.pkl'}
    models_dir = Path(__file__).resolve().parent / 'models'
    for model_name, file_name in model_files.items():
        path = models_dir / file_name
        if path.exists():
            models[model_name] = joblib.load(path)
    return models

# --- Load Data ---
@st.cache_data
def load_demo_data():
    return generate_synthetic_data()

# --- Main App ---
models = load_all_models()
demo_df = load_demo_data()

st.title("🔬 Bioprocess Critical Quality Attribute (CQA) Predictor")
st.write("This application uses internally-generated synthetic data to predict the final protein titer of a cell culture batch.")

if not models:
    st.error("Models not found. Please run `train_models.py` first.")
else:
    st.sidebar.title("Select a Batch")
    batch_ids = demo_df['batch_id'].unique()
    selected_batch_id = st.sidebar.selectbox("Select a Demo Batch to Analyze", sorted(batch_ids))

    batch_df = demo_df[demo_df['batch_id'] == selected_batch_id].copy()

    tab_summary, tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Data Summary", "📊 SPC Results", "⛓️ Lasso Regression",
        "🌳 Random Forest", "🚀 XGBoost", "⚖️ Model Comparison",
    ])

    with tab_summary:
        st.header(f"Data Summary for: {selected_batch_id}")
        st.dataframe(batch_df.describe())
        st.subheader("Correlation Heatmap")
        corr = batch_df[['temperature', 'ph', 'dissolved_oxygen']].corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="Blues", ax=ax)
        st.pyplot(fig)

    with tab1:
        st.header("Statistical Process Control (SPC) Chart")
        data = batch_df['temperature']
        cl, std_dev = data.mean(), data.std()
        ucl, lcl = cl + 3 * std_dev, cl - 3 * std_dev
        fig, ax = plt.subplots()
        sns.lineplot(x=data.index, y=data.values, marker='o', ax=ax, color='dodgerblue', label='Temperature')
        ax.axhline(cl, color='green', linestyle='--', label='Center Line (CL)')
        ax.axhline(ucl, color='red', linestyle='--', label='Control Limit (UCL)')
        ax.axhline(lcl, color='red', linestyle='--', label='Control Limit (LCL)')
        ax.set_title("SPC Chart for Temperature", fontsize=16)
        ax.set_xlabel("Data Point Index")
        ax.set_ylabel("Temperature")
        ax.legend()
        st.pyplot(fig)

    # Use the single, imported function to calculate features for the selected batch
    features = create_tabular_features(batch_df, is_training=False)

    with tab2:
        st.header("Lasso Regression Prediction")
        prediction = models['lasso'].predict(features)[0]
        st.metric(label="Predicted Titer", value=f"{prediction:.4f}")
        with st.expander("View Model Coefficients"):
            coeffs = pd.DataFrame(models['lasso'].coef_, features.columns, columns=['Coefficient']).sort_values(by='Coefficient', ascending=False)
            st.dataframe(coeffs)

    with tab3:
        st.header("Random Forest Prediction")
        prediction = models['rf'].predict(features)[0]
        st.metric(label="Predicted Titer", value=f"{prediction:.4f}")
        with st.expander("Feature Importances"):
            importances = pd.Series(models['rf'].feature_importances_, index=features.columns).sort_values(ascending=False)
            st.dataframe(importances)

    with tab4:
        st.header("XGBoost Prediction")
        prediction = models['xgb'].predict(features)[0]
        st.metric(label="Predicted Titer", value=f"{prediction:.4f}")
        with st.expander("Feature Importances"):
            importances = pd.Series(models['xgb'].feature_importances_, index=features.columns).sort_values(ascending=False)
            st.dataframe(importances)

    with tab5:
        st.header("Model Prediction Comparison")
        all_predictions = {'Model': ['Lasso Regression', 'Random Forest', 'XGBoost', 'Gradient Boosting'],
                           'Predicted Titer': [models['lasso'].predict(features)[0], models['rf'].predict(features)[0],
                                               models['xgb'].predict(features)[0], models['gb'].predict(features)[0]]}
        comparison_df = pd.DataFrame(all_predictions)
        st.dataframe(comparison_df.style.format({'Predicted Titer': '{:.4f}'}), use_container_width=True)
