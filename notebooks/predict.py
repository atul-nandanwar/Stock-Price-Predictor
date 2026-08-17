import pandas as pd
import joblib
import numpy as np

from pathlib import Path


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "dataset" / "AAPL_stock_data.csv"
MODEL_PATH = PROJECT_ROOT / "model" / "stock_price_model.pkl"
FEATURE_PATH = PROJECT_ROOT / "model" / "feature_columns.pkl"


# ============================================================
# 2. LOAD MODEL AND FEATURES
# ============================================================

print("\nLoading trained model...")

model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURE_PATH)

print("Model loaded successfully.")


# ============================================================
# 3. LOAD DATASET
# ============================================================

df = pd.read_csv(
    DATASET_PATH,
    header=[0, 1]
)

# Remove ticker level
df.columns = df.columns.get_level_values(0)

# Remove unnecessary first column
if "Price" in df.columns:
    df = df.drop(columns=["Price"])

# Remove missing rows
df = df.dropna().reset_index(drop=True)


# ============================================================
# 4. CONVERT NUMERIC COLUMNS
# ============================================================

numeric_columns = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# 5. CREATE FEATURES
# ============================================================

df["MA_7"] = df["Close"].rolling(
    window=7
).mean()

df["MA_21"] = df["Close"].rolling(
    window=21
).mean()

df["Previous_Close"] = df["Close"].shift(1)

df = df.dropna().reset_index(drop=True)


# ============================================================
# 6. GET MOST RECENT DATA
# ============================================================

latest_data = df.iloc[-1]

input_data = pd.DataFrame(
    [[
        latest_data["Open"],
        latest_data["High"],
        latest_data["Low"],
        latest_data["Close"],
        latest_data["Volume"],
        latest_data["MA_7"],
        latest_data["MA_21"],
        latest_data["Previous_Close"]
    ]],
    columns=features
)


# ============================================================
# 7. MAKE PREDICTION
# ============================================================

prediction = model.predict(input_data)

predicted_price = float(prediction[0])

current_price = float(latest_data["Close"])

change = predicted_price - current_price

percentage_change = (
    change / current_price
) * 100


# ============================================================
# 8. DISPLAY RESULT
# ============================================================

print("\n" + "=" * 60)
print("AAPL STOCK PRICE PREDICTION")
print("=" * 60)

print(f"Current Closing Price : ${current_price:.2f}")

print(
    f"Predicted Next-Day Price : ${predicted_price:.2f}"
)

print(
    f"Expected Change : ${change:.2f}"
)

print(
    f"Expected Change % : {percentage_change:.2f}%"
)

print("=" * 60)

print("\nPrediction completed successfully.")