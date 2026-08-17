import yfinance as yf
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset directory
DATASET_DIR = PROJECT_ROOT / "dataset"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# Stock symbol
ticker = "AAPL"

# Download historical data
data = yf.download(
    ticker,
    start="2015-01-01",
    end="2026-08-17",
    auto_adjust=False
)

# Dataset file path
file_path = DATASET_DIR / "AAPL_stock_data.csv"

# Save dataset
data.to_csv(file_path)

print("DATA COLLECTION COMPLETED SUCCESSFULLY")
print(f"Stock: {ticker}")
print(f"Rows: {len(data)}")
print(f"Columns: {list(data.columns)}")
print(f"Dataset saved at: {file_path}")