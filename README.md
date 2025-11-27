# Finance Predictor Pro (Native UI Edition)

<img width="1920" height="1053" alt="image" src="https://github.com/user-attachments/assets/dbe68db6-29cc-4c1a-94ad-66b29c2a60c1" />


A high-performance financial prediction desktop application built with **PyQt6** and **PyQtGraph**, featuring 10+ forecasting algorithms, sentiment analysis, and local LLM integration for AI-powered market insights.

## Features

- **Native Performance**: Built with PyQt6 for a responsive, hardware-accelerated desktop experience.
- **Modern Tabbed UI**: Sleek interface organized into **Dashboard**, **Analysis**, and **Charting** tabs for optimal workflow.
- **Multi-Algorithm Forecasting**: 10+ prediction models including Statistical (SMA, EMA, ARIMA, Holt-Winters), Machine Learning (Linear Regression, Random Forest, XGBoost, SVR), and Technical Indicators.
- **Sentiment-Adjusted Predictions**: Uses a **Bayesian-inspired drift adjustment** theorem to weight technical forecasts based on real-time news sentiment.
- **Advanced Quantitative Analysis**:
  - **Risk Metrics**: Value at Risk (VaR 95%), CVaR (Expected Shortfall), and Kelly Criterion.
  - **Market Regime**: Automatically detects if the market is **Trending** or **Ranging** using ADX.
- **Professional Charting**:
  - **Technical Indicators**: Ichimoku Cloud, Bollinger Bands, SMA, Stochastic, ATR.
  - **Pattern Recognition**: Auto-detects **Doji**, **Hammer**, and **Engulfing** candlestick patterns.
  - **Correlation Matrix**: Live heatmap showing asset correlations against benchmarks (SPY, BTC, GLD).
- **Real-time Data**: Fetches live stock, crypto, and forex data via `yfinance`.
- **AI Market Analyst**: Local LLM (Phi-4 Mini Reasoning via llama-cpp-python) with automatic GPU acceleration.
- **Optimized ML Engines**: Grid-search tuned Random Forest and XGBoost models for high-volatility assets.

## Installation

### Requirements
- Python 3.10+
- CUDA-capable GPU (optional, but recommended for AI features)

### Setup
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/finance-predictor.git
cd finance-predictor

# Switch to the native-ui branch
git checkout native-ui

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('vader_lexicon')"
```

## Usage

### Run Application
```bash
# Windows
run_app.bat

# Or manually
python app.py
```

### LLM Model Setup
Place your Phi-4 model file in the `models/` directory:
- `Phi-4-mini-reasoning-Q4_K_M.gguf`

## Project Structure

```
finance_predictor/
├── app.py                 # Main PyQt6 application
├── src/
│   ├── data_loader.py    # Data fetching logic
│   ├── forecasting.py    # Prediction algorithms (Optimized)
│   ├── quant_analysis.py # Risk metrics, patterns, indicators
│   ├── sentiment.py      # Sentiment analysis
│   ├── llm_engine.py     # LLM integration
│   ├── inference_script.py # Isolated inference process
│   └── utils.py          # Helper functions
├── requirements.txt
└── run_app.bat           # Launcher script
```

## Technologies

- **UI**: PyQt6 (Qt for Python)
- **Visualization**: PyQtGraph
- **Data**: yfinance, pandas, numpy
- **Technical Analysis**: `ta` library
- **ML**: scikit-learn, XGBoost, statsmodels, Prophet, arch, pykalman
- **NLP**: NLTK
- **AI/LLM**: llama-cpp-python (Phi-4)

## License

MIT

## Author

Built with ❤️ for financial analysis testing and app building by Abraham Jeevan Roy
