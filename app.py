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

st.title("🔬 The Early Warning Bioprocess Predictor")
st.write("""
**What if you could know the result of a 14-day manufacturing process after only 4 days?** This app predicts the final quality of a biopharmaceutical batch using early data, saving time and resources.
""")

if not models:
    st.error("Models not found. Please run the `train_models.py` script first.")
else:
    st.sidebar.title("Select a Batch")
    batch_ids = demo_df['batch_id'].unique()
    selected_batch_id = st.sidebar.selectbox("Select a Demo Batch to Analyze", sorted(batch_ids))

    batch_df = demo_df[demo_df['batch_id'] == selected_batch_id].copy()

    tab_summary, tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Batch Overview", "📊 Process Stability", "⛓️ Lasso Regression",
        "🌳 Random Forest", "🚀 XGBoost", "⚖️ Final Verdict"
    ])

    with tab_summary:
        st.header(f"What does the data for {selected_batch_id} look like?")
        st.write("Below are the raw sensor readings for the first four days of the selected batch. We can see how key parameters like pH and temperature change over time.")
        
        # Melt the DataFrame for easier plotting with Seaborn
        plot_df = batch_df.melt(id_vars=['time'], value_vars=['temperature', 'ph', 'dissolved_oxygen'],
                                var_name='Parameter', value_name='Value')

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=plot_df, x='time', y='Value', hue='Parameter', ax=ax)
        ax.set_title("Sensor Readings Over Time")
        ax.set_xlabel("Time (Days)")
        ax.set_ylabel("Sensor Value")
        ax.grid(True, linestyle='--')
        st.pyplot(fig)
        
        with st.expander("Show Raw Data Summary"):
            st.dataframe(batch_df.describe())

    with tab1:
        st.header("Is this batch running within normal limits?")
        st.write("""
        This **Statistical Process Control (SPC) chart** acts like a car's tachometer. As long as the blue line (temperature) stays between the red 'guard rails' (control limits), the process is stable. 
        Points outside the rails signal a potential problem.
        """)
        data = batch_df['temperature']
        cl, std_dev = data.mean(), data.std()
        ucl, lcl = cl + 3 * std_dev, cl - 3 * std_dev
        fig, ax = plt.subplots()
        sns.lineplot(x=data.index, y=data.values, marker='o', ax=ax, color='dodgerblue', label='Temperature')
        ax.axhline(cl, color='green', linestyle='--', label='Center Line (Average)')
        ax.axhline(ucl, color='red', linestyle='--', label='Upper Control Limit')
        ax.axhline(lcl, color='red', linestyle='--', label='Lower Control Limit')
        ax.set_title("Process Stability Chart for Temperature", fontsize=16)
        ax.set_xlabel("Data Point Index")
        ax.set_ylabel("Temperature")
        ax.legend()
        st.pyplot(fig)

    features = create_tabular_features(batch_df, is_training=False)

    with tab2:
        st.header("Which factors are driving the prediction?")
        st.write("""
        **Lasso Regression** is a transparent model that tells us exactly which factors it used. It creates a simple formula where important factors get a large coefficient (bar), and unimportant ones are set to zero.
        """)
        prediction = models['lasso'].predict(features)[0]
        st.metric(label="Lasso's Predicted Titer", value=f"{prediction:.4f}", help="A score from 0 to 1, where higher is better.")

        with st.expander("Show the model's reasoning"):
            coeffs = pd.DataFrame(models['lasso'].coef_, features.columns, columns=['Coefficient']).sort_values(by='Coefficient', ascending=False)
            fig, ax = plt.subplots()
            sns.barplot(x=coeffs['Coefficient'], y=coeffs.index, ax=ax, palette='viridis')
            ax.set_title("Factor Importance (Lasso Coefficients)")
            ax.set_xlabel("Impact on Prediction (Coefficient Value)")
            st.pyplot(fig)

    with tab3:
        st.header("What does a different model think?")
        st.write("""
        **Random Forest** is like asking hundreds of experts (decision trees) for their opinion and averaging the result. It's very robust and provides its own ranking of which factors were most important.
        """)
        prediction = models['rf'].predict(features)[0]
        st.metric(label="Random Forest's Predicted Titer", value=f"{prediction:.4f}", help="A score from 0 to 1, where higher is better.")
        with st.expander("Show this model's reasoning"):
            importances = pd.Series(models['rf'].feature_importances_, index=features.columns).sort_values(ascending=False)
            fig, ax = plt.subplots()
            sns.barplot(x=importances.values, y=importances.index, ax=ax, palette='flare')
            ax.set_title('Factor Importance (Random Forest)')
            ax.set_xlabel("Importance Score")
            st.pyplot(fig)

    with tab4:
        st.header("What does our most powerful model think?")
        st.write("""
        **XGBoost** is a high-performance model, often used in data science competitions. It builds trees sequentially, with each new tree correcting the errors of the previous one.
        """)
        prediction = models['xgb'].predict(features)[0]
        st.metric(label="XGBoost's Predicted Titer", value=f"{prediction:.4f}", help="A score from 0 to 1, where higher is better.")
        with st.expander("Show this model's reasoning"):
            importances = pd.Series(models['xgb'].feature_importances_, index=features.columns).sort_values(ascending=False)
            fig, ax = plt.subplots()
            sns.barplot(x=importances.values, y=importances.index, ax=ax, palette='mako')
            ax.set_title('Factor Importance (XGBoost)')
            ax.set_xlabel("Importance Score")
            st.pyplot(fig)

    with tab5:
        st.header("What's the final verdict on this batch?")
        st.write("""
        By comparing the predictions from all our models, we can arrive at a more confident conclusion. If all models agree, we can be more certain of the batch's future outcome.
        """)
        all_predictions = {'Model': ['Lasso Regression', 'Random Forest', 'XGBoost', 'Gradient Boosting'],
                           'Predicted Titer': [models['lasso'].predict(features)[0], models['rf'].predict(features)[0],
                                               models['xgb'].predict(features)[0], models['gb'].predict(features)[0]]}
        comparison_df = pd.DataFrame(all_predictions)
        
        fig, ax = plt.subplots()
        sns.barplot(x='Predicted Titer', y='Model', data=comparison_df.sort_values('Predicted Titer', ascending=False), ax=ax, palette='coolwarm')
        ax.set_title('Final Titer Prediction by Model')
        ax.set_xlabel('Predicted Titer (Higher is Better)')
        ax.set_ylabel('')
        ax.set_xlim(0, 1) # Set x-axis from 0 to 1 for consistency
        st.pyplot(fig)

        with st.expander("Show Prediction Data Table"):
            st.dataframe(comparison_df.style.format({'Predicted Titer': '{:.4f}'}), use_container_width=True)
