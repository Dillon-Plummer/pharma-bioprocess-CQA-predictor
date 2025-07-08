# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
from pathlib import Path

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

models = load_all_models()

# --- Feature Engineering Function ---
def create_tabular_features_single(df):
    """Engineers features for a single uploaded batch."""
    return pd.DataFrame({
        'mean_temp': [df['temperature'].mean()], 'std_temp': [df['temperature'].std()],
        'mean_ph': [df['ph'].mean()], 'std_ph': [df['ph'].std()],
        'mean_do': [df['dissolved_oxygen'].mean()], 'std_do': [df['dissolved_oxygen'].std()]
    })

# --- UI ---
st.sidebar.title("Upload Batch Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file with sensor data.", type=["csv"])

st.title("🔬 Bioprocess Critical Quality Attribute (CQA) Predictor")
st.write("This application predicts the final protein titer of a cell culture batch based on its early-stage sensor data.")

# --- Main Logic ---
if not models:
    st.error("Models not found. Please run the `train_models.py` script first to train and save the models.")
elif uploaded_file is None:
    st.info("Please upload a file through the sidebar to begin analysis.")
else:
    try:
        batch_df = pd.read_csv(uploaded_file)
        # Basic check for required columns
        required_cols = ['temperature', 'ph', 'dissolved_oxygen']
        if not all(col in batch_df.columns for col in required_cols):
            st.error(f"The uploaded CSV must contain the columns: {', '.join(required_cols)}")
        else:
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
                st.header("Data Summary")
                st.dataframe(batch_df.describe())
                st.subheader("Preview")
                st.dataframe(batch_df.head())

            with tab1:
                st.header("Statistical Process Control (SPC) Chart")
                st.write("This chart monitors the stability of the temperature process parameter.")

                # Manually calculate SPC limits using pandas
                data = batch_df['temperature']
                cl = data.mean()
                std_dev = data.std()
                ucl = cl + 3 * std_dev
                lcl = cl - 3 * std_dev

                # Plotting the chart
                fig, ax = plt.subplots()
                ax.plot(data.index, data, marker='o', linestyle='-', color='dodgerblue', label='Temperature')
                ax.axhline(cl, color='green', linestyle='--', label='Center Line (CL)')
                ax.axhline(ucl, color='red', linestyle='--', label='Control Limit (UCL)')
                ax.axhline(lcl, color='red', linestyle='--', label='Control Limit (LCL)')
                ax.set_title("SPC Chart for Temperature", fontsize=16)
                ax.set_xlabel("Data Point Index", fontsize=12)
                ax.set_ylabel("Average Temperature", fontsize=12)
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
                    st.caption("Features with a coefficient of 0 were excluded by the model.")

            with tab3:
                st.header("Random Forest Prediction")
                prediction = models['rf'].predict(features)[0]
                st.metric(label="Predicted Titer", value=f"{prediction:.4f}")
                st.info("Random Forest is a robust ensemble model that combines many decision trees to prevent overfitting.")

            with tab4:
                st.header("XGBoost Prediction")
                prediction = models['xgb'].predict(features)[0]
                st.metric(label="Predicted Titer", value=f"{prediction:.4f}")
                st.info("XGBoost is a highly optimized version of Gradient Boosting, often leading to high performance.")

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
                csv = comparison_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv",
                )

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")