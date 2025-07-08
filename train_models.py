# train_models.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

# Import our custom feature engineering function
from feature_engineering import create_gb_features

# --- 1. Load Data & Engineer Features ---
print("Loading data and engineering features...")
try:
    df = pd.read_excel('data/synthetic_cell_culture_data.xlsx')
except FileNotFoundError:
    print("Error: 'data/synthetic_cell_culture_data.xlsx' not found.")
    print("Please make sure you have the dataset in the 'data' directory.")
    exit()

features_tabular = create_gb_features(df)
labels = df.groupby('batch_id')['titer'].last()
data_tabular = features_tabular.join(labels)

X = data_tabular.drop('titer', axis=1)
y = data_tabular['titer']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. Train and Save Models ---
print("Training Gradient Boosting...")
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
joblib.dump(gb_model, 'models/gb_model.pkl')
print(f"  GB MSE: {mean_squared_error(y_test, gb_model.predict(X_test)):.4f}")
gb_cv = cross_val_score(gb_model, X, y, cv=5, scoring='neg_mean_squared_error')
print(f"  GB CV MSE: {-gb_cv.mean():.4f}")

print("Training Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, 'models/rf_model.pkl')
print(f"  RF MSE: {mean_squared_error(y_test, rf_model.predict(X_test)):.4f}")
rf_cv = cross_val_score(rf_model, X, y, cv=5, scoring='neg_mean_squared_error')
print(f"  RF CV MSE: {-rf_cv.mean():.4f}")

print("Training XGBoost...")
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
joblib.dump(xgb_model, 'models/xgb_model.pkl')
xgb_cv = cross_val_score(xgb_model, X, y, cv=5, scoring='neg_mean_squared_error')
print(f"  XGBoost MSE: {mean_squared_error(y_test, xgb_model.predict(X_test)):.4f}")
print(f"  XGBoost CV MSE: {-xgb_cv.mean():.4f}")

print("Training Lasso Regression...")
lasso_model = Lasso(alpha=0.01, random_state=42)
lasso_model.fit(X_train, y_train)
joblib.dump(lasso_model, 'models/lasso_model.pkl')
lasso_cv = cross_val_score(lasso_model, X, y, cv=5, scoring='neg_mean_squared_error')
print(f"  Lasso MSE: {mean_squared_error(y_test, lasso_model.predict(X_test)):.4f}")
print(f"  Lasso CV MSE: {-lasso_cv.mean():.4f}")

print("\nAll models trained and saved successfully! ✅")