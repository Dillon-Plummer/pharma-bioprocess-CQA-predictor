# train_models.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import os

# Import the new data generation and feature engineering functions
from synthetic_data import generate_synthetic_data
from feature_engineering import create_gb_features

# --- 1. Generate Data & Engineer Features ---
print("Generating synthetic data and engineering features...")
df = generate_synthetic_data()

features_tabular = create_gb_features(df)
labels = df.groupby('batch_id')['titer'].last()
data_tabular = features_tabular.join(labels)

X = data_tabular.drop('titer', axis=1)
y = data_tabular['titer']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. Train and Save Models ---
MODELS_DIR = 'models'
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

print("Training Gradient Boosting...")
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
joblib.dump(gb_model, os.path.join(MODELS_DIR, 'gb_model.pkl'))
print(f"  GB MSE: {mean_squared_error(y_test, gb_model.predict(X_test)):.4f}")

print("Training Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, os.path.join(MODELS_DIR, 'rf_model.pkl'))
print(f"  RF MSE: {mean_squared_error(y_test, rf_model.predict(X_test)):.4f}")

print("Training XGBoost...")
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
joblib.dump(xgb_model, os.path.join(MODELS_DIR, 'xgb_model.pkl'))
print(f"  XGBoost MSE: {mean_squared_error(y_test, xgb_model.predict(X_test)):.4f}")

print("Training Lasso Regression...")
lasso_model = Lasso(alpha=0.01, random_state=42)
lasso_model.fit(X_train, y_train)
joblib.dump(lasso_model, os.path.join(MODELS_DIR, 'lasso_model.pkl'))
print(f"  Lasso MSE: {mean_squared_error(y_test, lasso_model.predict(X_test)):.4f}")

print(f"\nAll models trained and saved successfully to the '{MODELS_DIR}' directory! ✅")
