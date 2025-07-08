# train_models.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import os

from synthetic_data import generate_synthetic_data
from feature_engineering import create_tabular_features # <-- Use the updated function

print("Generating synthetic data...")
df = generate_synthetic_data()

print("Engineering features for training...")
# Process the full training dataset
features_tabular = create_tabular_features(df, is_training=True)
labels = df.groupby('batch_id')['titer'].last()
data_tabular = features_tabular.join(labels)

X = data_tabular.drop('titer', axis=1)
y = data_tabular['titer']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

MODELS_DIR = 'models'
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

print("Training all models...")
# ... (rest of the training and saving code is the same)
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
joblib.dump(gb_model, os.path.join(MODELS_DIR, 'gb_model.pkl'))

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, os.path.join(MODELS_DIR, 'rf_model.pkl'))

xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
joblib.dump(xgb_model, os.path.join(MODELS_DIR, 'xgb_model.pkl'))

lasso_model = Lasso(alpha=0.01, random_state=42)
lasso_model.fit(X_train, y_train)
joblib.dump(lasso_model, os.path.join(MODELS_DIR, 'lasso_model.pkl'))

print(f"\nAll models trained and saved successfully! ✅")
