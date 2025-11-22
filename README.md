# Finance Predictor Pro

<img width="1899" height="1008" alt="image" src="https://github.com/user-attachments/assets/b861818a-8d8b-4262-93b4-319b53870d61" />


A comprehensive financial prediction application built with Python and Streamlit, featuring 10+ forecasting algorithms, sentiment analysis, and local LLM integration for AI-powered market insights.

## Features

- **Multi-Algorithm Forecasting**: 10+ prediction models including Statistical (SMA, EMA, ARIMA, Holt-Winters), Machine Learning (Linear Regression, Random Forest, XGBoost, SVR), and Technical Indicators (RSI, Bollinger Bands)
- **Real-time Data**: Fetches live stock, crypto, and forex data via `yfinance`
- **Sentiment Analysis**: NLTK VADER-based headline sentiment scoring
- **AI Market Analyst**: Local LLM (Phi-4 Mini Reasoning via llama-cpp-python) with automatic GPU acceleration
- **Interactive Visualizations**: Plotly-powered charts with color-coded predictions
- **Ticker Tapes**: Scrolling dot-matrix style tickers for indices and news
- **GPU Optimized**: Automatic NVIDIA CUDA acceleration for XGBoost and LLM inference

## Installation

### Requirements
- Python 3.10+
- CUDA-capable GPU (optional, but recommended)

### Setup
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/finance-predictor.git
cd finance-predictor

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('vader_lexicon')"
```

## Usage

### Run Locally
```bash
# Windows
run_app.bat

# Linux/Mac
streamlit run app.py
```

### LLM Model Setup
Place your Phi-3 model file in the `models/` directory:
- `Phi-3-mini-4k-instruct-q4.gguf` (or allow GPT4All to download automatically)

## Project Structure

```
finance_predictor/
├── app.py                 # Main Streamlit application
├── src/
│   ├── data_loader.py    # Data fetching logic
│   ├── forecasting.py    # Prediction algorithms
│   ├── sentiment.py      # Sentiment analysis
│   ├── llm_engine.py     # LLM integration
│   └── utils.py          # Helper functions
├── requirements.txt
└── build_exe.py          # PyInstaller packaging script
```

## Building Standalone Executable

```bash
python build_exe.py
```

## Technologies

- **UI**: Streamlit
- **Data**: yfinance, pandas, numpy
- **ML**: scikit-learn, XGBoost, statsmodels, Prophet
- **NLP**: NLTK, GPT4All
- **Visualization**: Plotly

## License

MIT

## Author

Built with ❤️ for financial analysis testing and app building by Abraham Jeevan Roy
