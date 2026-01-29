<div align="center">
  <img src="assets/logo.png" alt="Finance Predictor Logo" width="120" />
  <h1>Finance Predictor Pro (Native UI Edition)</h1>
  
  <p>
    <b>Next-Gen Financial Forecasting & Analysis</b><br>
    <i>Powered by PyQt6, XGBoost, and Local LLMs</i>
  </p>
  
  <!-- Badges -->
  <p>
    <img src="https://img.shields.io/badge/MCP-Compliant-00A67E?style=for-the-badge&logo=anthropic&logoColor=white" alt="MCP Compliant" />
    <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+" />
    <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="MIT License" />
    <img src="https://img.shields.io/badge/AI-Powered-FF6F61?style=for-the-badge&logo=openai&logoColor=white" alt="AI Powered" />
  </p>
  
  <p>
    <img src="https://img.shields.io/badge/PyQt6-UI-41CD52?style=flat-square&logo=qt&logoColor=white" alt="PyQt6" />
    <img src="https://img.shields.io/badge/XGBoost-ML-blue?style=flat-square" alt="XGBoost" />
    <img src="https://img.shields.io/badge/yfinance-Data-red?style=flat-square" alt="yfinance" />
    <img src="https://img.shields.io/badge/FastMCP-Server-00A67E?style=flat-square" alt="FastMCP" />
  </p>
  
  <img src="https://github.com/user-attachments/assets/b6ca30c2-c83c-4a1d-ae02-553fde7f4e68" alt="Dashboard" width="100%" />
</div>

---

## 🌟 Highlights

- **🔌 MCP Integration**: Full Model Context Protocol support - use with Claude Desktop, ChatGPT, and other AI assistants
- **🖥️ Native Performance**: Hardware-accelerated PyQt6 desktop UI
- **🤖 AI-Powered**: Local Granite 4.0 LLM for advanced market analysis
- **📊 10+ Forecasting Models**: From statistical methods to deep learning (CNN-GAF)
- **⚡ Real-Time Data**: Live stock, crypto, and forex data via yfinance
- **📈 Advanced Analytics**: Risk metrics, sentiment analysis, pattern recognition

## 🚀 Features

### MCP Integration (NEW!)

- **🔌 Model Context Protocol Compliant**: Full MCP server implementation with 9 tools, 4 resources, and 4 prompt templates
- **🤝 AI Assistant Ready**: Works with Claude Desktop, ChatGPT, and any MCP-compatible client
- **🛠️ Programmatic Access**: AI assistants can fetch data, run predictions, analyze sentiment, calculate risk metrics, and more
- **📋 Workflow Templates**: Pre-built prompts for market analysis, risk assessment, investment recommendations, and portfolio optimization
- **🔄 Real-Time Resources**: Live data feeds accessible via standardized URIs

### Desktop Application

- **Native Performance**: Built with PyQt6 for responsive, hardware-accelerated experience
- **Modern Tabbed UI**: Sleek interface organized into **Dashboard**, **Analysis**, and **Charting** tabs
- **Custom Controls**: macOS-style window controls, Material Design 3 components
- **Live Ticker Tapes**: Scrolling tickers for 50+ major stocks and real-time news headlines

### Forecasting & Prediction

- **10+ Algorithms**: 
  - **Statistical**: SMA, EMA, ARIMA, Holt-Winters
  - **Machine Learning**: Linear Regression, Random Forest (optimized), XGBoost (optimized), SVR
  - **Advanced**: CNN-GAF (Computer Vision), Prophet, Kalman Filter, Monte Carlo, LSTM
- **Sentiment-Adjusted Forecasts**: Bayesian-inspired drift adjustment using news sentiment
- **Ensemble Methods**: Stacked meta-models combining top performers

### Technical Analysis

- **Professional Charting**: PyQtGraph-powered charts with technical overlays
- **Indicators**: Ichimoku Cloud, Bollinger Bands, SMA, Stochastic, ATR, RSI, MACD, ADX
- **Pattern Recognition**: Auto-detects Doji, Hammer, Shooting Star, Engulfing, Morning/Evening Star
- **Correlation Matrix**: Live heatmap showing asset correlations against benchmarks (SPY, BTC, GLD)

### Risk Management

- **Quantitative Metrics**:
  - Value at Risk (VaR 95%)
  - Conditional VaR (CVaR / Expected Shortfall)
  - Annualized Volatility
  - Sharpe Ratio
  - Kelly Criterion (optimal position sizing)
- **Market Regime Detection**: Automatically detects Trending vs Ranging markets using ADX

### AI & Intelligence

- **Agentic AI Copilot**: Autonomous AI agent that can control the application via natural language
- **Advanced Reasoning**: Powered by Granite 4.0 Reasoning (IBM) for deep financial analysis
- **Sentiment Analysis**: VADER-based analysis of 15+ reliable news sources (Reuters, Bloomberg, CNBC, WSJ)
- **MCP Tools Integration**: AI assistants can orchestrate complex multi-step analyses

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

---

## 📊 Capabilities at a Glance

| Category | Count | Highlights |
|----------|-------|------------|
| **MCP Tools** | 9 | Stock data, predictions, sentiment, risk, patterns, indicators, AI analysis, correlation, regime |
| **Forecasting Models** | 10+ | XGBoost, Random Forest, ARIMA, CNN-GAF, Prophet, Kalman, Monte Carlo, LSTM |
| **Technical Indicators** | 7+ | RSI, MACD, Bollinger Bands, ADX, ATR, Stochastic, Ichimoku |
| **Risk Metrics** | 5 | VaR (95%), CVaR, Volatility, Sharpe Ratio, Kelly Criterion |
| **Candlestick Patterns** | 5 | Doji, Hammer, Shooting Star, Engulfing, Morning/Evening Star |
| **MCP Resources** | 4 | Historical data, company info, news feed, metrics |
| **MCP Prompts** | 4 | Market analysis, risk assessment, investment recs, portfolio optimization |
| **News Sources** | 15+ | Reuters, Bloomberg, CNBC, WSJ, Financial Times, Forbes, MarketWatch |

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

## MCP Integration

Finance Predictor is **Model Context Protocol (MCP) compliant**, enabling AI assistants like Claude Desktop, ChatGPT, and other MCP clients to access all core features programmatically.

### Available Capabilities

**🔧 Tools** (9 actionable functions):
- `fetch_stock_data` - Fetch historical OHLCV data
- `run_predictions` - Execute 10+ forecasting models (XGBoost, Random Forest, ARIMA, CNN-GAF, etc.)
- `analyze_sentiment` - News sentiment analysis with VADER
- `calculate_risk_metrics` - VaR, CVaR, Sharpe Ratio, Kelly Criterion
- `detect_candlestick_patterns` - Doji, Hammer, Engulfing patterns
- `get_technical_indicators` - RSI, MACD, Bollinger Bands, ADX
- `run_ai_analysis` - Local Granite 4.0 LLM analysis
- `get_correlation_matrix` - Multi-asset correlation analysis
- `get_market_regime` - Trending vs Ranging detection

**📊 Resources** (5 live data feeds):
- `stock://historical/{ticker}` - Historical price data
- `stock://info/{ticker}` - Company metadata
- `stock://news/{ticker}` - Recent news with sentiment
- `stock://metrics/{ticker}` - Risk metrics and patterns

**💡 Prompts** (4 templates):
- `market_analysis` - Comprehensive market analysis workflow
- `risk_assessment` - Risk evaluation template
- `investment_recommendation` - Buy/sell/hold recommendations
- `portfolio_optimization` - Multi-asset portfolio optimization

### Quick Start

1. **Install MCP dependencies**:
   ```bash
   pip install fastmcp mcp pydantic
   ```

2. **Run the MCP server**:
   ```bash
   python -m src.mcp_server
   ```

3. **Configure Claude Desktop** (or other MCP client):
   
   Copy the contents of `mcp_config.json` to your Claude Desktop configuration:
   - **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
   - **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **Linux**: `~/.config/Claude/claude_desktop_config.json`

4. **Restart Claude Desktop** - Finance Predictor tools will now be available!

### Example Usage in Claude

Try these prompts in Claude Desktop after setup:

> "Use the finance-predictor tools to analyze AAPL stock. Fetch the data, run predictions using XGBoost and Random Forest, analyze recent news sentiment, and provide a comprehensive investment recommendation."

> "Calculate risk metrics for TSLA including VaR, CVaR, and Kelly Criterion. Then compare its correlation with SPY benchmark."

> "Run a complete market analysis for NVDA using the market_analysis prompt template."

For detailed usage examples and troubleshooting, see **[MCP_GUIDE.md](MCP_GUIDE.md)**.


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

## 🛠️ Technologies

### UI & Visualization
- **PyQt6** - Qt for Python (native desktop UI)
- **PyQtGraph** - High-performance plotting library

### Data & APIs
- **yfinance** (≥0.2.66) - Real-time financial data
- **pandas** - Data manipulation
- **numpy** (<2.0) - Numerical computing

### Machine Learning & Forecasting
- **scikit-learn** - Classical ML algorithms
- **XGBoost** - Gradient boosting (optimized)
- **statsmodels** - Statistical models (ARIMA)
- **Prophet** - Facebook's time series forecaster
- **PyTorch** - Deep learning (CNN-GAF)
- **pyts** - Time series transformations (Gramian Angular Fields)
- **arch** - GARCH volatility modeling
- **pykalman** - Kalman filtering

### Technical Analysis
- **ta** - Technical indicators library

### Natural Language Processing
- **NLTK** - Sentiment analysis (VADER)

### AI & LLM
- **llama-cpp-python** - Local LLM inference (Granite 4.0)

### MCP (Model Context Protocol)
- **fastmcp** (≥1.0.0) - Streamlined MCP server framework
- **mcp** (≥0.9.0) - Official MCP SDK
- **pydantic** (≥2.0.0) - Type validation for MCP tools

---

## 🎯 What Makes This Unique?

1. **🔌 Full MCP Compliance**: First finance app with complete Model Context Protocol support - AI assistants can orchestrate complex multi-step analyses
2. **🧠 Hybrid Intelligence**: Combines 10+ traditional forecasting models with cutting-edge AI (Granite 4.0) and computer vision (CNN-GAF)
3. **📊 Institutional-Grade Risk**: VaR, CVaR, Kelly Criterion, and market regime detection - metrics used by hedge funds and quant traders
4. **⚡ Native Performance**: PyQt6 desktop app with hardware acceleration - no slow web frameworks
5. **🎨 Beautiful UI**: Material Design 3 components, macOS-style controls, live ticker tapes
6. **🔒 Privacy-First**: Local LLM (Granite 4.0) - your financial data never leaves your machine

## License

MIT

## Author

Built with ❤️ by **Abraham Jeevan Roy**  
For financial analysis testing and application development

---

<div align="center">
  <p><b>⭐ Star this repo if you find it useful! ⭐</b></p>
  
  <p>
    <a href="https://github.com/YOUR_USERNAME/finance-predictor/issues">Report Bug</a> •
    <a href="https://github.com/YOUR_USERNAME/finance-predictor/issues">Request Feature</a> •
    <a href="MCP_GUIDE.md">MCP Documentation</a>
  </p>
</div>
