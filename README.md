<div align="center">
  <img src="assets/logo.png" alt="Finance Predictor Logo" width="120" />
  <h1>Finance Predictor Pro (Native UI Edition)</h1>
</div>

<div align="center">
  <img src="https://github.com/user-attachments/assets/b6ca30c2-c83c-4a1d-ae02-553fde7f4e68" alt="Dashboard" width="100%" />
  
  <p>
    <b>Next-Gen Financial Forecasting & Analysis</b><br>
    <i>Powered by PyQt6, XGBoost, and Local LLMs</i>
  </p>
</div>

A high-performance financial prediction desktop application built with **PyQt6** and **PyQtGraph**, featuring 10+ forecasting algorithms, sentiment analysis, and local LLM integration for AI-powered market insights.

## Features

- **Native Performance**: Built with PyQt6 for a responsive, hardware-accelerated desktop experience.
- **Modern Tabbed UI**: Sleek interface organized into **Dashboard**, **Analysis**, and **Charting** tabs for optimal workflow.
- **Multi-Algorithm Forecasting**: 10+ prediction models including Statistical (SMA, EMA, ARIMA, Holt-Winters), Machine Learning (Linear Regression, Random Forest, XGBoost, SVR), and **Computer Vision (CNN-GAF)**.
- **Sentiment-Adjusted Predictions**: Uses a **Bayesian-inspired drift adjustment** theorem to weight technical forecasts based on real-time news sentiment.
- **Advanced Quantitative Analysis**:
  - **Risk Metrics**: Value at Risk (VaR 95%), CVaR (Expected Shortfall), and Kelly Criterion.
  - **Market Regime**: Automatically detects if the market is **Trending** or **Ranging** using ADX.
- **Professional Charting**:
  - **Technical Indicators**: Ichimoku Cloud, Bollinger Bands, SMA, Stochastic, ATR.
  - **Pattern Recognition**: Auto-detects **Doji**, **Hammer**, and **Engulfing** candlestick patterns.
  - **Correlation Matrix**: Live heatmap showing asset correlations against benchmarks (SPY, BTC, GLD).
- **Real-time Data**: Fetches live stock, crypto, and forex data via `yfinance`.
- **Agentic AI Copilot**: Autonomous AI agent capable of controlling the application (loading tickers, running predictions, adjusting settings) via natural language commands.
- **Advanced Reasoning Model**: Powered by **Granite 4.0 Reasoning** (IBM) for deep financial analysis and planning.
- **Optimized ML Engines**: Grid-search tuned Random Forest and XGBoost models for high-volatility assets.

## Interface Gallery

<div align="center">
  <table>
    <tr>
      <td align="center"><b>Advanced Analysis</b></td>
      <td align="center"><b>Professional Charting</b></td>
    </tr>
    <tr>
      <td><img src="https://github.com/user-attachments/assets/8f9b9ea9-44a9-448f-b2e3-278d9c4c31cc" alt="Analysis Tab" width="100%" /></td>
      <td><img src="https://github.com/user-attachments/assets/5a4cd3b4-6391-49ea-8000-7f82a2f2389c" alt="Charting Tab" width="100%" /></td>
    </tr>
  </table>
</div>

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

### Headless Mode (WIP)
Run in terminal without GUI:
```bash
python app.py --headless
```

### LLM Model Setup
Place your Granite 4.0 model file in the `models/` directory:
- `granite-4.0-h-tiny-adaptive-reasoning.i1-IQ4_XS.gguf`

## Project Structure

```
finance_predictor/
├── app.py                 # Main PyQt6 application
├── src/
│   ├── data_loader.py    # Data fetching logic
│   ├── forecasting.py    # Prediction algorithms (Optimized)
│   ├── cv_forecasting.py # Computer Vision forecasting (CNN-GAF)
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
- **Computer Vision**: PyTorch, pyts (Gramian Angular Fields)
- **NLP**: NLTK
- **AI/LLM**: llama-cpp-python (Granite 4.0 Reasoning)

## License

MIT

## Author

Built with ❤️ for financial analysis testing and app building by Abraham Jeevan Roy
