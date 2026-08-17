import streamlit as st
import pandas as pd
import joblib
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import timedelta

# ============================================================
# PAGE CONFIGURATION & ULTRA-FUTURISTIC CSS
# ============================================================
st.set_page_config(
    page_title="NEXUS AI | Quantitative Terminal",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800&family=Rajdhani:wght@500;600;700&display=swap');

    .stApp {
        background-color: #080B10;
        color: #E0E6ED;
        font-family: 'Rajdhani', sans-serif;
    }
    
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(16, 23, 38, 0.9) 0%, rgba(10, 15, 26, 0.95) 100%);
        border: 1px solid rgba(0, 240, 255, 0.25);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 0 20px rgba(0, 240, 255, 0.08);
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        border-color: rgba(0, 240, 255, 0.6);
        transform: translateY(-2px);
    }
    
    .live-ticker {
        background: rgba(13, 20, 36, 0.8);
        border-bottom: 1px solid rgba(0, 240, 255, 0.2);
        padding: 10px 15px;
        border-radius: 8px;
        display: flex;
        gap: 30px;
        overflow: hidden;
        font-size: 0.95rem;
        margin-bottom: 25px;
    }
    .neon-text-blue { color: #00F0FF; font-weight: 700; text-shadow: 0 0 10px rgba(0,240,255,0.5); }
    .neon-text-green { color: #00FFA3; font-weight: 700; text-shadow: 0 0 10px rgba(0,255,163,0.5); }
    .neon-text-red { color: #FF0055; font-weight: 700; text-shadow: 0 0 10px rgba(255,0,85,0.5); }
</style>
""", unsafe_allow_html=True)

# ============================================================
# PROJECT PATHS & MODEL LOADING
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "model" / "stock_price_model.pkl"
FEATURE_PATH = PROJECT_ROOT / "model" / "feature_columns.pkl"

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURE_PATH)
    return model, features

@st.cache_data(ttl=3600)
def get_stock_data(ticker, period="2y"):
    data = yf.download(ticker, period=period, auto_adjust=False, progress=False)
    ticker_obj = yf.Ticker(ticker)
    info = ticker_obj.info if hasattr(ticker_obj, 'info') else {}
    return data, info

def prepare_features(data):
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    req = ["Open", "High", "Low", "Close", "Volume"]
    df = data[req].copy()
    for col in req:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        
    df["MA_7"] = df["Close"].rolling(window=7).mean()
    df["MA_21"] = df["Close"].rolling(window=21).mean()
    df["Previous_Close"] = df["Close"].shift(1)
    
    # RSI Indicator
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df.dropna()

# ============================================================
# SIDEBAR CONTROLS
# ============================================================
with st.sidebar:
    st.markdown("<h2 style='color:#00F0FF; font-family:Orbitron;'>🔮 NEXUS AI</h2>", unsafe_allow_html=True)
    st.caption("Quantum ML Forecasting Engine")
    st.markdown("---")
    
    st.markdown("### 🌐 Market Selection")
    market_type = st.radio("Exchange", ["India (NSE)", "US Tech / Global", "Crypto", "Custom Ticker"], horizontal=True)
    
    if market_type == "India (NSE)":
        ticker = st.selectbox(
            "Select Indian Stock",
            ["RELIANCE.NS", "TCS.NS", "INFY.NS", "TATAMOTORS.NS", "HDFCBANK.NS", "SBIN.NS", "ITC.NS", "ICICIBANK.NS", "BHARTIARTL.NS", "WIPRO.NS"]
        )
        currency_sym = "₹"
    elif market_type == "US Tech / Global":
        ticker = st.selectbox(
            "Select US Stock",
            ["NVDA", "AAPL", "MSFT", "TSLA", "AMZN", "GOOGL", "META", "AMD", "NFLX", "INTC"]
        )
        currency_sym = "$"
    elif market_type == "Crypto":
        ticker = st.selectbox(
            "Select Crypto Asset",
            ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "BNB-USD", "ADA-USD"]
        )
        currency_sym = "$"
    else:
        ticker = st.text_input("Enter Any Global Symbol", value="TATAPOWER.NS").upper().strip()
        currency_sym = "$"

    col1, col2 = st.columns(2)
    with col1:
        timeframe = st.selectbox("Lookback", ["1 Mo", "3 Mo", "6 Mo", "1 Yr", "2 Yr"], index=3)
    with col2:
        chart_mode = st.selectbox("Style", ["Cyber Candlestick", "Laser Line"])
        
    st.markdown("### Technical Overlays")
    show_ma = st.toggle("Moving Averages (7 & 21)", value=True)
    show_forecast_line = st.toggle("Plot ML Forecast Target", value=True)
    show_rsi = st.toggle("Show RSI Subplot", value=True)
    show_volume = st.toggle("Show Volume Subplot", value=True)
    
    st.markdown("---")
    predict_button = st.button("⚡ EXECUTE NEURAL INFERENCE", type="primary", use_container_width=True)

# ============================================================
# LOAD MODEL & EXECUTE
# ============================================================
try:
    model, features = load_model()
except Exception as error:
    st.error(f"Failed to load model weights: {error}")
    st.stop()

# Live Market Ticker Tape
st.markdown("""
<div class="live-ticker">
    <span><b>NIFTY 50:</b> <span class="neon-text-green">+0.75%</span></span>
    <span><b>NASDAQ:</b> <span class="neon-text-green">+1.42%</span></span>
    <span><b>BTC/USD:</b> <span class="neon-text-blue">$94,210.00</span></span>
    <span><b>AI ENGINE:</b> <span class="neon-text-green">ONLINE (OPTIMAL)</span></span>
</div>
""", unsafe_allow_html=True)

with st.spinner(f"Connecting to live global feed for {ticker}..."):
    try:
        raw_data, info = get_stock_data(ticker)
        if raw_data.empty:
            st.error(f"Asset '{ticker}' not found. Note: For Indian stocks use '.NS' suffix (e.g. RELIANCE.NS).")
            st.stop()
            
        prepared_data = prepare_features(raw_data)
        if prepared_data.empty:
            st.error("Insufficient historical trading data to build indicators.")
            st.stop()
            
        latest = prepared_data.iloc[-1]
        
        # Model Prediction
        input_data = pd.DataFrame([[
            latest["Open"], latest["High"], latest["Low"], latest["Close"],
            latest["Volume"], latest["MA_7"], latest["MA_21"], latest["Previous_Close"]
        ]], columns=features)
        
        predicted_price = float(model.predict(input_data)[0])
        current_price = float(latest["Close"])
        price_diff = predicted_price - current_price
        pct_diff = (price_diff / current_price) * 100
        
        # Next Target Date
        next_date = prepared_data.index[-1] + timedelta(days=1)
        if next_date.weekday() == 5: next_date += timedelta(days=2)
        elif next_date.weekday() == 6: next_date += timedelta(days=1)
        
        # Header Info
        company_name = info.get('longName', ticker)
        st.markdown(f"<h1 style='color:#FFFFFF; font-family:Orbitron;'>{company_name} <span style='color:#00F0FF;'>[{ticker}]</span></h1>", unsafe_allow_html=True)
        
        # Futuristic Metric Cards
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Live Market Price", f"{currency_sym}{current_price:,.2f}")
        m2.metric("AI Target Prediction", f"{currency_sym}{predicted_price:,.2f}", delta=f"{pct_diff:+.2f}%")
        m3.metric("Expected Alpha", f"{currency_sym}{price_diff:+,.2f}", delta_color="normal")
        
        signal = "STRONG BUY 🚀" if pct_diff > 1.0 else ("BUY 📈" if pct_diff > 0 else ("STRONG SELL 🔻" if pct_diff < -1.0 else "SELL 📉"))
        m4.metric("Neural Signal", signal)
        
        st.markdown("---")
        
        # Subplots configuration
        rows = 1 + int(show_volume) + int(show_rsi)
        row_heights = [0.65] + ([0.18] if show_volume else []) + ([0.17] if show_rsi else [])
        if rows == 1: row_heights = [1.0]
        
        fig = make_subplots(
            rows=rows, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=row_heights
        )
        
        window_dict = {"1 Mo": 22, "3 Mo": 66, "6 Mo": 132, "1 Yr": 252, "2 Yr": 504}
        plot_df = prepared_data.tail(window_dict.get(timeframe, 252))
        
        # Main Candlestick / Line
        if "Candlestick" in chart_mode:
            fig.add_trace(go.Candlestick(
                x=plot_df.index, open=plot_df['Open'], high=plot_df['High'],
                low=plot_df['Low'], close=plot_df['Close'], name='OHLC',
                increasing_line_color='#00FFA3', decreasing_line_color='#FF0055'
            ), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(
                x=plot_df.index, y=plot_df['Close'], mode='lines', name='Price',
                line=dict(color='#00F0FF', width=2.5)
            ), row=1, col=1)
            
        if show_ma:
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA_7'], name='MA (7)', line=dict(color='#FFB800', width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA_21'], name='MA (21)', line=dict(color='#BD00FF', width=1.5)), row=1, col=1)
            
        # Plot Prediction Target
        if show_forecast_line:
            fig.add_trace(go.Scatter(
                x=[plot_df.index[-1], next_date],
                y=[current_price, predicted_price],
                mode='lines+markers',
                name='AI Forecast Path',
                line=dict(color='#00F0FF', width=3, dash='dot'),
                marker=dict(size=8, color='#00F0FF', symbol='diamond')
            ), row=1, col=1)
            
        current_row = 2
        if show_volume:
            vol_colors = ['#00FFA3' if c >= o else '#FF0055' for c, o in zip(plot_df['Close'], plot_df['Open'])]
            fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['Volume'], name='Volume', marker_color=vol_colors, opacity=0.8), row=current_row, col=1)
            current_row += 1
            
        if show_rsi:
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['RSI'], name='RSI (14)', line=dict(color='#00F0FF', width=1.5)), row=current_row, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="#FF0055", row=current_row, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="#00FFA3", row=current_row, col=1)

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(8, 11, 16, 0.95)',
            plot_bgcolor='rgba(13, 20, 36, 0.6)',
            height=680,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabs for Raw Analysis
        t1, t2 = st.tabs(["⚡ Neural Vector Feed", "📈 Quantitative Signals"])
        with t1:
            st.dataframe(plot_df.tail(10).sort_index(ascending=False), use_container_width=True)
        with t2:
            st.json({
                "Asset Symbol": ticker,
                "Latest RSI": round(latest['RSI'], 2),
                "MA 7 Spread": round(current_price - latest['MA_7'], 2),
                "MA 21 Spread": round(current_price - latest['MA_21'], 2),
                "Predicted Target Price": round(predicted_price, 2),
                "Target Timestamp": str(next_date.date())
            })

    except Exception as err:
        st.error(f"Engine Exception: {err}")