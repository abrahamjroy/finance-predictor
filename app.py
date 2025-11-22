import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.data_loader import DataLoader
from src.forecasting import ForecastEngine
from src.sentiment import SentimentEngine
from src.llm_engine import LLMEngine
from src.utils import setup_dirs

# Page Config
st.set_page_config(page_title="Finance Predictor Pro", layout="wide", page_icon="📈")

# --- Ticker Data Fetching with Real-Time Prices ---
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_ticker_data():
    """Fetch real-time data for 50+ global assets."""
    import yfinance as yf
    
    # Expanded asset list (50+ Global Stocks & Indices)
    assets = {
        # Indices
        "^GSPC": "S&P 500", "^DJI": "DOW", "^IXIC": "NASDAQ", "^FTSE": "FTSE 100", 
        "^N225": "NIKKEI 225", "^GDAXI": "DAX", "^HSI": "HANG SENG",
        
        # US Tech Giants
        "AAPL": "APPLE", "MSFT": "MICROSOFT", "GOOGL": "GOOGLE", "AMZN": "AMAZON", 
        "NVDA": "NVIDIA", "TSLA": "TESLA", "META": "META", "NFLX": "NETFLIX",
        "AMD": "AMD", "INTC": "INTEL", "ORCL": "ORACLE", "CRM": "SALESFORCE",
        
        # Global Blue Chips
        "JPM": "JPMORGAN", "V": "VISA", "JNJ": "J&J", "WMT": "WALMART", 
        "PG": "P&G", "KO": "COCA-COLA", "PEP": "PEPSI", "XOM": "EXXON",
        "CVX": "CHEVRON", "BAC": "BOA", "MA": "MASTERCARD", "DIS": "DISNEY",
        
        # European Giants
        "NESN.SW": "NESTLE", "ROG.SW": "ROCHE", "NOVN.SW": "NOVARTIS", 
        "ASML": "ASML", "SAP": "SAP", "LVMUY": "LVMH", "SIEGY": "SIEMENS",
        "TTE": "TOTALENERGIES", "SNY": "SANOFI", "AZN": "ASTRAZENECA",
        
        # Asian Giants
        "TSM": "TSMC", "BABA": "ALIBABA", "TCEHY": "TENCENT", "SONY": "SONY",
        "TM": "TOYOTA", "HDB": "HDFC BANK", "IBN": "ICICI BANK", "INFY": "INFOSYS",
        
        # Commodities & Crypto
        "GC=F": "GOLD", "SI=F": "SILVER", "CL=F": "OIL", "BTC-USD": "BITCOIN", "ETH-USD": "ETHEREUM"
    }
    
    ticker_items = []
    
    # Batch fetch for speed? yfinance handles threading internally usually, 
    # but individual Ticker calls are safer for mixed assets.
    for symbol, display_name in assets.items():
        try:
            ticker = yf.Ticker(symbol)
            # Fast fetch
            hist = ticker.history(period="2d")
            
            if not hist.empty and len(hist) >= 2:
                current = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2]
                change = current - prev
                pct = (change / prev) * 100
                
                color = "#00ff00" if change >= 0 else "#ff3333"
                arrow = "▲" if change >= 0 else "▼"
                
                ticker_items.append(
                    f'<span style="color:{color}">{display_name}: {current:.2f} {arrow} {abs(pct):.2f}%</span>'
                )
        except:
            continue
            
    return ticker_items

# Initialize Ticker in Session State to prevent refresh
if 'ticker_html' not in st.session_state:
    with st.spinner("Loading Global Market Data..."):
        ticker_data = fetch_ticker_data()
        st.session_state['ticker_html'] = f"""
        <div class="ticker-wrap ticker-top">
            <div class="ticker">
                {'   |   '.join(ticker_data)}
            </div>
        </div>
        """

# Sidebar Configuration
st.sidebar.title("Configuration")

# Theme Toggle
theme = st.sidebar.radio("Theme Mode", ["Dark", "Light"], index=0)

# Dynamic CSS Variables based on Theme
if theme == "Dark":
    bg_color = "#0e1117"
    text_color = "#fafafa"
    h1_color = "#ffffff"
    h2_color = "#e0e0e0"
    h3_color = "#d0d0d0"
    card_bg = "#262730"
    chat_user_bg = "#1e3a8a" # Dark blue
    chat_ai_bg = "#14532d"   # Dark green
else:
    bg_color = "#ffffff"
    text_color = "#1a1a1a"
    h1_color = "#1a1a2e"
    h2_color = "#16213e"
    h3_color = "#0f3460"
    card_bg = "#f8f9fa"
    chat_user_bg = "#e3f2fd"
    chat_ai_bg = "#f1f8e9"

# Custom CSS for Professional Typography & Dot-Matrix Ticker
st.markdown(f"""
    <style>
    /* Import Professional Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600&family=Lora:wght@400;600&family=Merriweather:wght@300;400&family=Libre+Baskerville&family=Roboto+Mono:wght@400;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');
    
    /* Global Typography */
    .stApp {{
        background-color: {bg_color};
        font-family: 'Merriweather', serif;
        font-size: 16px;
        font-weight: 300;
        line-height: 1.7;
        color: {text_color};
    }}
    
    /* Headings */
    h1, h2, h3 {{
        font-family: 'Playfair Display', serif;
        font-weight: 600;
        line-height: 1.2;
    }}
    
    h1 {{ font-size: 3rem; letter-spacing: -0.02em; color: {h1_color}; }}
    h2 {{ font-size: 2rem; letter-spacing: -0.01em; color: {h2_color}; }}
    h3 {{ font-size: 1.5rem; color: {h3_color}; }}
    
    /* Body Text */
    p, .stMarkdown, .stText, li {{
        font-family: 'Lora', serif;
        color: {text_color};
    }}
    
    /* Buttons */
    .stButton > button {{
        font-family: 'Libre Baskerville', serif;
        font-weight: 600;
    }}
    
    /* Numeric Values */
    .stMetric, .metric-value {{
        font-family: 'Roboto Mono', monospace;
    }}
    
    /* Dot-Matrix Ticker */
    .ticker-wrap {{
        position: fixed;
        width: 100%;
        overflow: hidden;
        height: 36px;
        background: linear-gradient(180deg, #000000 0%, #0a0a0a 100%);
        border: 1px solid #1a1a1a;
        font-family: 'Press Start 2P', monospace;
        font-size: 10px;
        line-height: 36px;
        z-index: 999999;
        white-space: nowrap;
        box-sizing: border-box;
        pointer-events: none;
        text-shadow: 0 0 8px currentColor, 0 0 15px currentColor;
        letter-spacing: 1px;
        color: #00ff00; /* Default ticker color */
    }}
    
    .ticker-top {{ top: 0; left: 0; border-bottom: 2px solid #00ff00; box-shadow: 0 2px 10px rgba(0,255,0,0.3); }}
    .ticker-bottom {{ bottom: 0; left: 0; border-top: 2px solid #ff9900; box-shadow: 0 -2px 10px rgba(255,153,0,0.3); }}
    
    .ticker {{ display: inline-block; padding-left: 100%; animation: ticker 90s linear infinite; }}
    .ticker-news {{ animation: ticker-news 60s linear infinite; }}
    .ticker-item {{ display: inline-block; padding: 0 3rem; }}
    
    @keyframes ticker {{ 0% {{ transform: translate3d(0, 0, 0); }} 100% {{ transform: translate3d(-100%, 0, 0); }} }}
    @keyframes ticker-news {{ 0% {{ transform: translate3d(0, 0, 0); }} 100% {{ transform: translate3d(-100%, 0, 0); }} }}
    
    /* Main Content Padding */
    .block-container {{
        padding-top: 70px;
        padding-bottom: 60px;
    }}
    
    /* Chatbot Styling */
    .chat-container {{
        background-color: {card_bg};
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        border: 1px solid #dee2e6;
    }}
    
    .chat-message {{
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 8px;
        line-height: 1.5;
        color: {text_color};
    }}
    
    .chat-user {{
        background-color: {chat_user_bg};
        border-left: 4px solid #2196f3;
    }}
    
    .chat-ai {{
        background-color: {chat_ai_bg};
        border-left: 4px solid #4caf50;
    }}
    </style>
    """, unsafe_allow_html=True)

st.markdown(st.session_state['ticker_html'], unsafe_allow_html=True)


# Initialize Components
@st.cache_resource
def init_engines():
    setup_dirs()
    return ForecastEngine(), SentimentEngine()

forecaster, sentiment_analyzer = init_engines()

# Initialize LLM separately (not cached to avoid loading issues)
if 'llm_engine' not in st.session_state:
    with st.spinner("Loading AI Model..."):
        st.session_state['llm_engine'] = LLMEngine()

llm_engine = st.session_state['llm_engine']


# Sidebar
# st.sidebar.title("Configuration") # Removed as it's handled above
ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL").upper()
period = st.sidebar.selectbox("History Period", ["1y", "2y", "5y", "max"], index=1)
forecast_days = st.sidebar.slider("Forecast Days", 7, 90, 30)

# AI Model Status
st.sidebar.markdown("---")
st.sidebar.subheader("System Status")
if llm_engine and llm_engine.model:
    st.sidebar.success("🟢 AI Model Ready")
else:
    st.sidebar.error("🔴 AI Model Offline")
    if st.sidebar.button("Force Load Model"):
        with st.spinner("Reloading Model..."):
            llm_engine._load_model()
            st.rerun()

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
                    # Ensure values are properly flattened to 1D array
                    values = pred_series.values
                    # Handle nested arrays by raveling and ensuring proper shape
                    if values.ndim > 1:
                        values = values.ravel()
                    # Extract from nested single-element arrays
                    if len(values) > 0 and hasattr(values[0], '__iter__') and not isinstance(values[0], str):
                        values = np.array([float(v[0]) if hasattr(v, '__iter__') else float(v) for v in values])
                    preds[model_name] = values
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
        
        # Initialize session state for chat
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []
        if 'analysis_context' not in st.session_state:
            st.session_state['analysis_context'] = ""
        if 'current_ticker' not in st.session_state:
            st.session_state['current_ticker'] = ticker
        
        # Clear chat if ticker changed
        if st.session_state['current_ticker'] != ticker:
            st.session_state['chat_history'] = []
            st.session_state['analysis_context'] = ""
            st.session_state['current_ticker'] = ticker
        
        if st.button("Generate AI Analysis"):
            with st.spinner("AI is analyzing with GPU acceleration..."):
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
                
                # Robust LLM Check
                if not llm_engine.model:
                    st.warning("LLM not loaded. Attempting to load now...")
                    llm_engine._load_model()
                
                if llm_engine.model:
                    analysis = llm_engine.analyze(prompt)
                    st.write(analysis)
                else:
                    st.error("Failed to load AI model. Please check logs or try 'Force Load' in sidebar.")
                    analysis = "AI Analysis Unavailable."
                
                # Store as context for chatbot
                st.session_state['analysis_context'] = analysis
                # Clear previous chat when new analysis is generated
                st.session_state['chat_history'] = []
        
        # Display existing analysis if available
        if st.session_state['analysis_context']:
            st.markdown("---")
            st.markdown("#### 💬 Follow-up Questions")
            st.caption("Ask me anything about this analysis!")
            
            # Chat input
            user_question = st.text_input("Your question:", key="chat_input", placeholder="e.g., What about competitors? Any risks?")
            
            col_send, col_clear = st.columns([1, 1])
            with col_send:
                send_button = st.button("Send", use_container_width=True)
            with col_clear:
                if st.button("Clear Chat", use_container_width=True):
                    st.session_state['chat_history'] = []
                    st.rerun()
            
            if send_button and user_question:
                with st.spinner("Thinking..."):
                    # Build conversation context
                    messages = [
                        {"role": "system", "content": f"You are a financial analyst. Here's the context: {st.session_state['analysis_context']}"},
                    ]
                    
                    # Add last 5 exchanges
                    for msg in st.session_state['chat_history'][-10:]:
                        messages.append(msg)
                    
                    # Add current question
                    messages.append({"role": "user", "content": user_question})
                    
                    # Get response
                    response = llm_engine.chat(messages)
                    
                    # Update history
                    st.session_state['chat_history'].append({"role": "user", "content": user_question})
                    st.session_state['chat_history'].append({"role": "assistant", "content": response})
                    
                    # Force refresh
                    st.rerun()
            
            # Display chat history
            if st.session_state['chat_history']:
                st.markdown("**Chat History:**")
                for msg in st.session_state['chat_history']:
                    if msg['role'] == 'user':
                        st.markdown(f'<div class="chat-message chat-user"><strong>You:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="chat-message chat-ai"><strong>AI:</strong> {msg["content"]}</div>', unsafe_allow_html=True)


# --- Bottom Ticker (News) ---
news_headlines = [f"{n['publisher']}: {n['title']}" for n in news]
if not news_headlines:
    news_headlines = ["No recent news available."]

bottom_ticker_html = f"""
<div class="ticker-wrap ticker-bottom">
    <div class="ticker ticker-news">
        {'   +++   '.join(news_headlines)}
    </div>
</div>
"""
st.markdown(bottom_ticker_html, unsafe_allow_html=True)
