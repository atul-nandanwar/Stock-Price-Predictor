import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "dataset" / "AAPL_stock_data.csv"
MODEL_DIR = PROJECT_ROOT / "model"
SCREENSHOT_DIR = PROJECT_ROOT / "screenshots" / "model"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(
    DATASET_PATH,
    header=[0, 1]
)

print("Dataset loaded successfully.")
print("Original shape:", df.shape)


# ============================================================
# 3. CLEAN MULTI-LEVEL COLUMNS
# ============================================================

# Remove the ticker level from columns
df.columns = df.columns.get_level_values(0)

print("\nColumns after cleaning:")
print(df.columns.tolist())


# ============================================================
# 4. REMOVE UNNECESSARY COLUMNS
# ============================================================

# Date is stored in the first column
if "Price" in df.columns:
    df = df.drop(columns=["Price"])

# Remove rows containing missing values
df = df.dropna().reset_index(drop=True)

print("\nShape after cleaning:", df.shape)
print("Missing values:")
print(df.isnull().sum())


# ============================================================
# 5. FEATURE ENGINEERING
# ============================================================

# Convert numeric columns to numeric type
numeric_columns = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")


# 7-day moving average
df["MA_7"] = df["Close"].rolling(window=7).mean()

# 21-day moving average
df["MA_21"] = df["Close"].rolling(window=21).mean()

# Previous day's closing price
df["Previous_Close"] = df["Close"].shift(1)

# Target = next day's closing price
df["Target"] = df["Close"].shift(-1)


# Remove rows created by rolling/shift operations
df = df.dropna().reset_index(drop=True)


print("\nFeature engineering completed.")
print("Final dataset shape:", df.shape)


# ============================================================
# 6. SELECT FEATURES AND TARGET
# ============================================================

features = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "MA_7",
    "MA_21",
    "Previous_Close"
]

X = df[features]
y = df["Target"]


# ============================================================
# 7. TIME-BASED TRAIN/TEST SPLIT
# ============================================================

# IMPORTANT:
# Stock data must not be randomly shuffled.
# Earlier data = training
# Later data = testing

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 8. DEFINE MODELS
# ============================================================

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42,
        max_depth=10
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        max_depth=15
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}


# ============================================================
# 9. TRAIN AND EVALUATE MODELS
# ============================================================

results = []

trained_models = {}

print("\n" + "=" * 70)
print("MODEL TRAINING")
print("=" * 70)


for name, model in models.items():

    print(f"\nTraining {name}...")

    # Train model
    model.fit(X_train, y_train)

    # Prediction
    predictions = model.predict(X_test)

    # Evaluation metrics
    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

    trained_models[name] = model

    print(f"MAE      : {mae:.4f}")
    print(f"RMSE     : {rmse:.4f}")
    print(f"R2 Score : {r2:.4f}")


# ============================================================
# 10. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

# Sort by RMSE
results_df = results_df.sort_values(
    by="RMSE",
    ascending=True
).reset_index(drop=True)


print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(results_df.to_string(index=False))


# ============================================================
# 11. SELECT BEST MODEL
# ============================================================

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[best_model_name]

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print("Best Model:", best_model_name)


# ============================================================
# 12. SAVE BEST MODEL
# ============================================================

model_path = MODEL_DIR / "stock_price_model.pkl"

joblib.dump(
    best_model,
    model_path
)

# Save feature names as well
feature_path = MODEL_DIR / "feature_columns.pkl"

joblib.dump(
    features,
    feature_path
)


print("\nBest model saved successfully.")
print("Model path:", model_path)
print("Feature file:", feature_path)


# ============================================================
# 13. SAVE MODEL COMPARISON
# ============================================================

results_path = MODEL_DIR / "model_comparison.csv"

results_df.to_csv(
    results_path,
    index=False
)

print("Model comparison saved at:", results_path)


# ============================================================
# 14. PLOT ACTUAL VS PREDICTED
# ============================================================

best_predictions = best_model.predict(X_test)

plt.figure(figsize=(12, 6))

plt.plot(
    y_test.values,
    label="Actual Price"
)

plt.plot(
    best_predictions,
    label="Predicted Price"
)

plt.title(
    f"AAPL Stock Price Prediction - {best_model_name}"
)

plt.xlabel("Test Samples")
plt.ylabel("Stock Price")

plt.legend()

plt.tight_layout()

plot_path = SCREENSHOT_DIR / "actual_vs_predicted.png"

plt.savefig(plot_path, dpi=300)

plt.show()

print("Prediction graph saved at:", plot_path)


# ============================================================
# 15. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("MODEL TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)

print(f"Best Model : {best_model_name}")
print(f"RMSE       : {results_df.iloc[0]['RMSE']:.4f}")
print(f"MAE        : {results_df.iloc[0]['MAE']:.4f}")
print(f"R2 Score   : {results_df.iloc[0]['R2 Score']:.4f}")