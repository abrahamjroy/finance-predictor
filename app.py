import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.data_loader import DataLoader
from src.forecasting import ForecastEngine
from src.sentiment import SentimentEngine
from src.llm_engine import LLMEngine
from src.utils import setup_dirs

# Page Config
st.set_page_config(page_title="Finance Predictor Pro", layout="wide", page_icon="📈")

# --- Ticker Data Fetching (Moved to Top) ---
major_indices = ["^GSPC", "^DJI", "^IXIC", "BTC-USD", "ETH-USD", "EURUSD=X", "GBPUSD=X"]
ticker_data = []
# Quick fetch or fallback to ensure UI renders fast
ticker_data = ["S&P 500: $5,300.00 ▲", "BTC: $68,000.00 ▲", "ETH: $3,800.00 ▲", "EUR/USD: 1.08 ▼"]

# Custom CSS for Ticker Tapes & Font
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=VT323&display=swap');
    
    .ticker-wrap {
        position: fixed;
        width: 100%;
        overflow: hidden;
        height: 30px;
        background-color: #000;
        color: #0f0; /* Green terminal text */
        font-family: 'VT323', monospace;
        font-size: 20px;
        line-height: 30px;
        z-index: 999999; /* Force on top of Streamlit header */
        white-space: nowrap;
        box-sizing: border-box;
        pointer-events: none; /* Allow clicking through */
    }
    
    .ticker-top { top: 0; left: 0; border-bottom: 1px solid #333; }
    .ticker-bottom { bottom: 0; left: 0; border-top: 1px solid #333; }
    
    .ticker {
        display: inline-block;
        padding-left: 100%;
        animation: ticker 40s linear infinite; /* Slowed down */
    }
    
    .ticker-item {
        display: inline-block;
        padding: 0 2rem;
    }
    
    @keyframes ticker {
        0%   { transform: translate3d(0, 0, 0); }
        100% { transform: translate3d(-100%, 0, 0); }
    }
    
    /* Adjust main content to not be hidden by tickers */
    .block-container {
        padding-top: 60px; /* Increased padding for top ticker */
        padding-bottom: 50px;
    }
    </style>
    """, unsafe_allow_html=True)

top_ticker_html = f"""
<div class="ticker-wrap ticker-top">
    <div class="ticker">
        {'   |   '.join(ticker_data)}   |   (Loading Real-time Data...)
    </div>
</div>
"""
st.markdown(top_ticker_html, unsafe_allow_html=True)

# Initialize Components
@st.cache_resource
def init_engines():
    setup_dirs()
    return ForecastEngine(), SentimentEngine(), LLMEngine()

forecaster, sentiment_analyzer, llm_engine = init_engines()


# Sidebar
st.sidebar.title("Configuration")
ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL").upper()
period = st.sidebar.selectbox("History Period", ["1y", "2y", "5y", "max"], index=1)
forecast_days = st.sidebar.slider("Forecast Days", 7, 90, 30)

st.sidebar.subheader("Prediction Models")
model_options = list(forecaster.models.keys())
selected_models = []
for model in model_options:
    if st.sidebar.checkbox(model, value=True):
        selected_models.append(model)

# Main Content
st.title(f"📈 Financial Analysis: {ticker}")

# Data Loading
with st.spinner("Fetching Data..."):
    df = DataLoader.fetch_history(ticker, period=period)
    news = DataLoader.fetch_news(ticker)

if df.empty:
    st.error("No data found. Please check the ticker symbol.")
else:
    # 1. Historical Chart
    # 1. Historical Chart & Predictions
    st.subheader("Price History & Forecast")
    
    # Initialize session state for predictions
    if 'predictions' not in st.session_state:
        st.session_state['predictions'] = None
    if 'forecast_days' not in st.session_state:
        st.session_state['forecast_days'] = 30

    # Run Predictions Button
    if st.button("Run Predictions"):
        with st.spinner("Running Forecasting Algorithms..."):
            preds = {}
            # Use the sidebar value
            days = forecast_days 
            st.session_state['forecast_days'] = days
            
            for model_name in selected_models:
                try:
                    pred_series = forecaster.models[model_name](df, days)
                    preds[model_name] = pred_series.values
                except Exception as e:
                    st.warning(f"Model {model_name} failed: {e}")
            
            st.session_state['predictions'] = preds

    # Plotting
    fig = go.Figure()
    
    # Historical Data (Candlestick)
    fig.add_trace(go.Candlestick(x=df.index,
                open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'], name='OHLC'))

    # Add Predictions if available
    if st.session_state['predictions']:
        preds = st.session_state['predictions']
        days = st.session_state['forecast_days']
        future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=days)
        pred_df = pd.DataFrame(preds, index=future_dates)
        
        # Color Palette
        colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD', 
            '#D4A5A5', '#9B59B6', '#3498DB', '#E67E22', '#2ECC71'
        ]
        
        # Individual Lines
        for i, col in enumerate(pred_df.columns):
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(
                x=pred_df.index, 
                y=pred_df[col], 
                name=col, 
                mode='lines', 
                line=dict(width=2, color=color, dash='dot')
            ))
        
        # Average Line
        avg_pred = pred_df.mean(axis=1)
        fig.add_trace(go.Scatter(
            x=pred_df.index, 
            y=avg_pred, 
            name='AVERAGE', 
            line=dict(color='#FFD700', width=5) # Gold, thick
        ))
        
        # Update title/metrics
        st.info(f"Forecast ({days} days) - Average Prediction: ${avg_pred.iloc[-1]:.2f}")

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=600,
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

    # 3. Sentiment & AI Analysis
    st.subheader("Market Sentiment & AI Analysis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Sentiment Score")
        score = sentiment_analyzer.analyze_news(news)
        
        # Gauge-like display
        if score > 0.05:
            st.success(f"Bullish ({score:.2f})")
        elif score < -0.05:
            st.error(f"Bearish ({score:.2f})")
        else:
            st.warning(f"Neutral ({score:.2f})")
            
        st.markdown("#### Recent Headlines")
        for item in news[:5]:
            st.caption(f"- [{item['title']}]({item['link']})")

    with col2:
        st.markdown("### AI Market Analyst")
        if st.button("Generate AI Analysis"):
            with st.spinner("AI is thinking... (This uses your RTX 4070)"):
                # Construct Prompt
                last_price = df['Close'].iloc[-1]
                price_change = df['Close'].iloc[-1] - df['Close'].iloc[0]
                pct_change = (price_change / df['Close'].iloc[0]) * 100
                
                prompt = f"""
                You are a financial analyst. Analyze the following data for {ticker}:
                - Current Price: ${last_price:.2f}
                - Period Change: {pct_change:.2f}%
                - News Sentiment Score: {score:.2f} (-1 to 1)
                
                Recent Headlines:
                {[n['title'] for n in news[:3]]}
                
                Provide a concise market outlook and trading recommendation based on this data.
                """
                
                analysis = llm_engine.analyze(prompt)
                st.write(analysis)

# --- Bottom Ticker (News) ---
news_headlines = [f"{n['publisher']}: {n['title']}" for n in news]
if not news_headlines:
    news_headlines = ["No recent news available."]

bottom_ticker_html = f"""
<div class="ticker-wrap ticker-bottom">
    <div class="ticker">
        {'   +++   '.join(news_headlines)}
    </div>
</div>
"""
st.markdown(bottom_ticker_html, unsafe_allow_html=True)
