# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# Import the new data generation function
from synthetic_data import generate_synthetic_data

# --- Page Configuration ---
st.set_page_config(page_title="Bioprocess CQA Predictor", page_icon="🔬", layout="wide")

# --- Model Loading ---
@st.cache_resource
def load_all_models():
    """Loads all models from the /models directory."""
    models = {}
    model_files = {
        'gb': 'gb_model.pkl',
        'rf': 'rf_model.pkl',
        'xgb': 'xgb_model.pkl',
        'lasso': 'lasso_model.pkl'
    }
    base_dir = Path(__file__).resolve().parent
    models_dir = base_dir / 'models'
    for model_name, file_name in model_files.items():
        path = models_dir / file_name
        if path.exists():
            models[model_name] = joblib.load(path)
    return models

# --- Load Data ---
@st.cache_data
def load_demo_data():
    """Generates and caches the synthetic demo data."""
    return generate_synthetic_data()

# --- Main App ---
models = load_all_models()
demo_df = load_demo_data()

st.title("🔬 Bioprocess Critical Quality Attribute (CQA) Predictor")
st.write("This application uses internally-generated synthetic data to predict the final protein titer of a cell culture batch.")

# --- Main Logic ---
if not models:
    st.error("Models not found. Please run the `train_models.py` script first to train and save the models.")
else:
    # --- UI ---
    st.sidebar.title("Select a Batch")
    batch_ids = demo_df['batch_id'].unique()
    selected_batch_id = st.sidebar.selectbox("Select a Demo Batch to Analyze", sorted(batch_ids))

    # Filter data for the selected batch
    batch_df = demo_df[demo_df['batch_id'] == selected_batch_id].copy()

    # Create tabs
    tab_summary, tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Data Summary",
        "📊 SPC Results",
        "⛓️ Lasso Regression",
        "🌳 Random Forest",
        "🚀 XGBoost",
        "⚖️ Model Comparison",
    ])

    with tab_summary:
        st.header(f"Data Summary for: {selected_batch_id}")
        st.dataframe(batch_df.describe())
        st.subheader("Preview")
        st.dataframe(batch_df.head())

        st.subheader("Correlation Heatmap")
        corr = batch_df[['temperature', 'ph', 'dissolved_oxygen']].corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="Blues", ax=ax)
        ax.set_title("Correlation of Process Parameters")
        st.pyplot(fig)

    with tab1:
        st.header("Statistical Process Control (SPC) Chart")
        st.write("This chart monitors the stability of the temperature process parameter for the selected batch.")

        data = batch_df['temperature']
        cl = data.mean()
        std_dev = data.std()
        ucl = cl + 3 * std_dev
        lcl = cl - 3 * std_dev

        fig, ax = plt.subplots()
        sns.lineplot(x=data.index, y=data.values, marker='o', ax=ax, color='dodgerblue', label='Temperature')
        ax.axhline(cl, color='green', linestyle='--', label='Center Line (CL)')
        ax.axhline(ucl, color='red', linestyle='--', label='Control Limit (UCL)')
        ax.axhline(lcl, color='red', linestyle='--', label='Control Limit (LCL)')
        ax.set_title("SPC Chart for Temperature", fontsize=16)
        ax.set_xlabel("Data Point Index", fontsize=12)
        ax.set_ylabel("Temperature", fontsize=12)
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        st.pyplot(fig)

    features = create_tabular_features_single(batch_df)

    with tab2:
        st.header("Lasso Regression Prediction")
        prediction = models['lasso'].predict(features)[0]
        st.metric(label="Predicted Titer", value=f"{prediction:.4f}")
        st.info("Lasso Regression is a highly transparent linear model that automatically selects the most important process features.")
        
        with st.expander("View Model Coefficients"):
            coeffs = pd.DataFrame(
                models['lasso'].coef_,
                features.columns,
                columns=['Coefficient']
            ).sort_values(by='Coefficient', ascending=False)
            st.dataframe(coeffs)
            fig, ax = plt.subplots()
            sns.barplot(x=coeffs.index, y='Coefficient', data=coeffs, ax=ax, palette='crest')
            ax.set_title('Lasso Coefficients')
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)
            st.caption("Features with a coefficient of 0 were excluded by the model.")

    with tab3:
        st.header("Random Forest Prediction")
        prediction = models['rf'].predict(features)[0]
        st.metric(label="Predicted Titer", value=f"{prediction:.4f}")
        st.info("Random Forest is a robust ensemble model that combines many decision trees to prevent overfitting.")
        with st.expander("Feature Importances"):
            importances = pd.Series(models['rf'].feature_importances_, index=features.columns).sort_values(ascending=False)
            fig, ax = plt.subplots()
            sns.barplot(x=importances.index, y=importances.values, ax=ax, palette='flare')
            ax.set_title('Random Forest Feature Importance')
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)

    with tab4:
        st.header("XGBoost Prediction")
        prediction = models['xgb'].predict(features)[0]
        st.metric(label="Predicted Titer", value=f"{prediction:.4f}")
        st.info("XGBoost is a highly optimized version of Gradient Boosting, often leading to high performance.")
        with st.expander("Feature Importances"):
            importances = pd.Series(models['xgb'].feature_importances_, index=features.columns).sort_values(ascending=False)
            fig, ax = plt.subplots()
            sns.barplot(x=importances.index, y=importances.values, ax=ax, palette='mako')
            ax.set_title('XGBoost Feature Importance')
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)

    with tab5:
        st.header("Model Prediction Comparison")
        all_predictions = {
            'Model': ['Lasso Regression', 'Random Forest', 'XGBoost', 'Gradient Boosting'],
            'Predicted Titer': [
                models['lasso'].predict(features)[0],
                models['rf'].predict(features)[0],
                models['xgb'].predict(features)[0],
                models['gb'].predict(features)[0]
            ]
        }
        comparison_df = pd.DataFrame(all_predictions)
        st.dataframe(
            comparison_df.style.format({'Predicted Titer': '{:.4f}'}),
            use_container_width=True,
        )
        fig, ax = plt.subplots()
        sns.barplot(x='Model', y='Predicted Titer', data=comparison_df, ax=ax, palette='pastel')
        ax.set_title('Predicted Titer by Model')
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)
