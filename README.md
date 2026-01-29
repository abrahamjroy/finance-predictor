<div align="center">
  <img src="assets/logo.png" alt="Finance Predictor Logo" width="120" />
  <h1>Finance Predictor Pro</h1>
  
  <p>
    <b>Institutional-Grade Financial Forecasting & Analysis Platform</b><br>
    <i>Native Desktop Application | Model Context Protocol Compliant</i>
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

## Overview

A production-ready financial forecasting platform combining classical statistical methods, machine learning, and deep learning techniques with institutional-grade risk analytics. Built with PyQt6 for native desktop performance and featuring complete Model Context Protocol (MCP) integration for programmatic access via AI assistants.

**Key Differentiators:**
- **MCP Compliance**: Industry-first full implementation of the Model Context Protocol, enabling seamless integration with Claude Desktop, ChatGPT, and other MCP-compatible clients
- **Hybrid Architecture**: Combines 10+ forecasting algorithms spanning statistical (ARIMA, Holt-Winters), ML (XGBoost, Random Forest), and DL (CNN-GAF, LSTM) approaches
- **Quantitative Rigor**: Institutional-grade risk metrics including VaR, CVaR, Sharpe Ratio, and Kelly Criterion
- **Native Performance**: Hardware-accelerated PyQt6 UI with real-time charting via PyQtGraph
- **Privacy-First AI**: Local LLM inference (DeepSeek-R1-Distill-Qwen-1.5B) - all analysis performed on-device

## Architecture & Features

### Model Context Protocol (MCP) Server

Complete implementation of the Model Context Protocol specification, exposing all application capabilities via a standardized JSON-RPC interface.

**Implementation Details:**
- **Server Framework**: Built on FastMCP SDK with automatic schema generation
- **Transport Layer**: stdio-based communication for seamless integration with MCP clients
- **Tool Registry**: 9 registered tools with Pydantic schema validation
- **Resource Handlers**: 4 dynamic resources with 5-minute TTL caching
- **Prompt Templates**: 4 pre-configured workflow templates for structured analysis
- **Type Safety**: Full type validation using Pydantic v2 models

**Available Tools:**
- `fetch_stock_data` - Historical OHLCV retrieval with configurable periods/intervals
- `run_predictions` - Multi-model forecasting pipeline execution
- `analyze_sentiment` - VADER-based sentiment analysis on curated news sources
- `calculate_risk_metrics` - VaR, CVaR, volatility, and Sharpe ratio computation
- `detect_candlestick_patterns` - Technical pattern recognition engine
- `get_technical_indicators` - Real-time indicator calculation (RSI, MACD, Bollinger, etc.)
- `run_ai_analysis` - Local LLM inference for qualitative analysis
- `get_correlation_matrix` - Multi-asset correlation matrix generation
- `get_market_regime` - ADX-based regime classification

### Desktop Application

**UI Framework**: PyQt6 with hardware-accelerated rendering  
**Charting Engine**: PyQtGraph for real-time, high-performance plotting  
**Design System**: Material Design 3 components with custom theme engine

**Architecture:**
- Multi-threaded design with separate worker threads for data fetching and model inference
- Event-driven architecture using Qt's signal/slot mechanism
- Modular tab-based interface (Dashboard, Analysis, Charting)
- Custom window controls with platform-specific styling

### Forecasting Engine

Multi-strategy prediction pipeline with 10+ algorithm implementations:

**Statistical Methods:**
- Simple/Exponential Moving Averages (SMA/EMA)
- AutoRegressive Integrated Moving Average (ARIMA)
- Holt-Winters Exponential Smoothing
- Kalman Filtering

**Machine Learning:**
- Linear Regression (baseline)
- Random Forest Regressor (grid-search optimized)
- XGBoost (hyperparameter-tuned for high volatility)
- Support Vector Regression (RBF kernel)

**Deep Learning:**
- CNN-GAF: Convolutional Neural Network with Gramian Angular Field transformations
- FB Prophet: Time series forecaster with trend/seasonality decomposition
- LSTM: Long Short-Term Memory networks
- Monte Carlo simulation for probabilistic forecasting

**Novel Techniques:**
- Sentiment-adjusted drift correction using Bayesian inference
- Ensemble stacking with meta-learner
- Cross-validation with time series splits

### Technical Analysis Suite

**Indicators** (implemented via `ta` library + custom implementations):
- Momentum: RSI, MACD, Stochastic Oscillator
- Volatility: Bollinger Bands, ATR, Keltner Channels
- Trend: Ichimoku Cloud, ADX, Parabolic SAR
- Volume: OBV, CMF, VWAP

**Pattern Recognition:**
- Single-candle patterns: Doji, Hammer, Shooting Star
- Multi-candle patterns: Engulfing, Morning/Evening Star
- Real-time detection with configurable sensitivity

**Correlation Analysis:**
- Pearson correlation matrix for multi-asset portfolios
- Benchmark comparison (SPY, BTC, GLD)
- Live heatmap visualization

### Risk Analytics

**Metrics Implementation:**
- **Value at Risk (VaR)**: Historical simulation method at 95% confidence
- **Conditional VaR (CVaR)**: Expected shortfall beyond VaR threshold
- **Volatility**: Annualized standard deviation with exponential weighting
- **Sharpe Ratio**: Risk-adjusted return calculation (assuming risk-free rate)
- **Kelly Criterion**: Optimal position sizing based on win probability and payoff ratio

**Market Regime Detection:**
- ADX-based classification (Trending: ADX > 25, Ranging: ADX < 25)
- Regime-adaptive strategy recommendations

### AI Integration

**Local LLM**: DeepSeek-R1-Distill-Qwen-1.5B  
**Inference**: llama-cpp-python with GGUF quantization (Q6_K/Q4_K_M)  
**Architecture**: Isolated subprocess to prevent DLL conflicts

**Capabilities:**
- Qualitative market analysis and narrative generation
- Strategy formulation based on quantitative inputs
- Natural language interface for application control (agentic mode)

**Sentiment Analysis:**
- VADER (Valence Aware Dictionary and sEntiment Reasoner)
- Curated news sources: Reuters, Bloomberg, CNBC, WSJ, Financial Times, Forbes, MarketWatch
- Aggregate sentiment scoring with source reliability weighting

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

## Technical Specifications

| Component | Implementation | Details |
|-----------|----------------|---------|
| **MCP Tools** | 9 | Stock data API, multi-model forecaster, sentiment analyzer, risk calculator, pattern detector, indicator engine, LLM interface, correlation analyzer, regime classifier |
| **Forecasting Models** | 10+ | XGBoost, Random Forest, ARIMA, CNN-GAF, Prophet, Kalman, Monte Carlo, LSTM, Holt-Winters, SVR |
| **Technical Indicators** | 7+ | RSI, MACD, Bollinger Bands, ADX, ATR, Stochastic Oscillator, Ichimoku Cloud |
| **Risk Metrics** | 5 | VaR (95%), CVaR, Annualized Volatility, Sharpe Ratio, Kelly Criterion |
| **Pattern Recognition** | 5 | Doji, Hammer, Shooting Star, Engulfing (bullish/bearish), Morning/Evening Star |
| **MCP Resources** | 4 | Historical OHLCV, company metadata, news aggregation, risk metrics |
| **MCP Prompts** | 4 | Market analysis, risk assessment, investment recommendations, portfolio optimization |
| **Data Sources** | 15+ | Reuters, Bloomberg, CNBC, WSJ, Financial Times, Forbes, MarketWatch, etc. |

## Installation

### Prerequisites
- **Python**: 3.10 or higher
- **CUDA**: Optional, but recommended for deep learning models (CNN-GAF, LSTM)
- **Memory**: Minimum 8GB RAM; 16GB recommended for concurrent model execution

### Installation Steps

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/finance-predictor.git
cd finance-predictor

# Switch to native-ui branch
git checkout native-ui

# Install Python dependencies
pip install -r requirements.txt

# Download NLTK VADER lexicon
python -c "import nltk; nltk.download('vader_lexicon')"
```

## Usage

### Desktop Application

```bash
# Windows
run_app.bat

# Cross-platform
python app.py
```

### Headless Mode

For CLI-based execution without GUI:

```bash
python app.py --headless
```

### MCP Server

To expose application capabilities via Model Context Protocol:

```bash
# Start MCP server
python -m src.mcp_server
```

## Model Context Protocol Integration

Complete MCP specification implementation enabling programmatic access by AI assistants and automation tools.

### Server Configuration

**1. Install MCP dependencies:**
```bash
pip install fastmcp mcp pydantic
```

**2. Launch MCP server:**
```bash
python -m src.mcp_server
```

**3. Configure MCP client:**

For Claude Desktop, edit configuration file:
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

Add server configuration from `mcp_config.json`.

**4. Restart client** - Finance Predictor tools will be registered and available.

### API Endpoints

**Tools (9):**
- `fetch_stock_data` - OHLCV retrieval with period/interval configuration
- `run_predictions` - Multi-model forecasting pipeline
- `analyze_sentiment` - VADER-based news sentiment analysis
- `calculate_risk_metrics` - VaR, CVaR, Sharpe, Kelly computation
- `detect_candlestick_patterns` - Technical pattern recognition
- `get_technical_indicators` - RSI, MACD, Bollinger, ADX calculation
- `run_ai_analysis` - Local LLM inference (DeepSeek-R1-Distill-Qwen-1.5B)
- `get_correlation_matrix` - Multi-asset correlation analysis
- `get_market_regime` - ADX-based regime classification

**Resources (4):**
- `stock://historical/{ticker}` - Historical price data (1-year daily)
- `stock://info/{ticker}` - Company metadata and fundamentals
- `stock://news/{ticker}` - Recent news with sentiment scores
- `stock://metrics/{ticker}` - Risk metrics and detected patterns

**Prompts (4):**
- `market_analysis` - Structured 6-step market analysis workflow
- `risk_assessment` - Comprehensive risk evaluation template
- `investment_recommendation` - Buy/sell/hold decision framework
- `portfolio_optimization` - Multi-asset allocation strategy

### Example Queries

```
# Via Claude Desktop or compatible MCP client

"Analyze AAPL: fetch 2-year data, run XGBoost and Random Forest predictions for 30 days, analyze news sentiment, and provide investment recommendation."

"Compare risk profiles of TSLA vs SPY: calculate VaR, CVaR, Sharpe ratio, and correlation coefficient."

"Execute market_analysis prompt template for NVDA with full technical and fundamental analysis."
```

Detailed documentation: [MCP_GUIDE.md](MCP_GUIDE.md)

### LLM Configuration

Place DeepSeek-R1-Distill-Qwen-1.5B model file in `models/` directory:
```
models/DeepSeek-R1-Distill-Qwen-1.5B-Q6_K.gguf
```

**Download:** [Hugging Face - bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF](https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF)

Recommended quantizations:
- **Q6_K** (1.4GB) - Best quality for financial analysis
- **Q4_K_M** (990MB) - Good balance of speed and quality
- **Q4_0** (880MB) - Fastest, lowest quality

## Project Structure

```
finance-predictor/
├── app.py                      # Main PyQt6 application entry point
├── src/
│   ├── __init__.py
│   ├── __main__.py             # MCP server CLI entry point
│   ├── data_loader.py          # yfinance API wrapper
│   ├── forecasting.py          # ML/statistical prediction engines
│   ├── cv_forecasting.py       # CNN-GAF computer vision forecaster
│   ├── quant_analysis.py       # Risk metrics and pattern detection
│   ├── sentiment.py            # VADER sentiment analyzer
│   ├── llm_engine.py           # DeepSeek-R1 LLM interface
│   ├── inference_script.py     # Isolated LLM inference subprocess
│   ├── mcp_server.py           # FastMCP server implementation
│   ├── mcp_tools.py            # MCP tool definitions
│   ├── mcp_resources.py        # MCP resource handlers
│   └── utils.py                # Logging and directory setup
├── requirements.txt            # Python dependencies
├── mcp_config.json             # MCP client configuration template
├── MCP_GUIDE.md                # MCP integration documentation
└── run_app.bat                 # Windows launcher script
```

## Technology Stack

### UI & Visualization
- **PyQt6** - Qt 6 for Python (native cross-platform UI framework)
- **PyQtGraph** - High-performance scientific plotting library

### Data & APIs
- **yfinance** (≥0.2.66) - Yahoo Finance API wrapper for market data
- **pandas** - Data structures and analysis toolkit
- **numpy** (<2.0) - Numerical computing (constrained for Prophet compatibility)

### Machine Learning & Forecasting
- **scikit-learn** - Classical ML algorithms (Random Forest, SVR, Linear Regression)
- **XGBoost** - Gradient boosting framework (optimized with grid search)
- **statsmodels** - Statistical models (ARIMA, GARCH)
- **Prophet** - Facebook's time series forecasting library
- **PyTorch** - Deep learning framework (CNN-GAF implementation)
- **pyts** - Time series transformation library (Gramian Angular Fields)
- **arch** - Autoregressive conditional heteroskedasticity (GARCH) models
- **pykalman** - Kalman filtering and smoothing

### Technical Analysis
- **ta** - Technical analysis indicator library

### Natural Language Processing
- **NLTK** - Natural Language Toolkit (VADER sentiment analysis)

### AI & LLM
- **llama-cpp-python** - Python bindings for llama.cpp (GGUF model inference)

### MCP (Model Context Protocol)
- **fastmcp** (≥1.0.0) - High-level MCP server framework
- **mcp** (≥0.9.0) - Official Anthropic MCP SDK
- **pydantic** (≥2.0.0) - Data validation using Python type annotations

---

## Design Decisions

**Why MCP Integration?**  
Model Context Protocol standardizes how AI assistants interact with external tools. By implementing MCP, Finance Predictor becomes a first-class data source for Claude, ChatGPT, and other MCP-compatible clients, enabling complex multi-step financial analyses orchestrated by AI.

**Why PyQt6 Over Web Frameworks?**  
Native desktop applications provide:
- Lower latency UI updates (critical for real-time charting)
- Direct GPU access for hardware-accelerated rendering
- No network overhead for local operations
- Better resource management for computationally intensive models

**Why Local LLM?**  
Privacy and data sovereignty. Financial data analysis often involves sensitive information. Local inference ensures all data remains on-device while still providing advanced AI capabilities.

**Why 10+ Forecasting Models?**  
No single model dominates across all market conditions. Ensemble approach:
- Diversifies prediction risk
- Captures different signal types (trend, mean-reversion, volatility)
- Enables model performance comparison
- Supports regime-adaptive strategy selection

## License

MIT License - see [LICENSE](LICENSE) for details

## Contributing

Contributions are welcome. Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

## Author

**Abraham Jeevan Roy**  
Financial Analysis & Application Development

---

<div align="center">
  <p>
    <a href="https://github.com/YOUR_USERNAME/finance-predictor/issues">Report Bug</a> •
    <a href="https://github.com/YOUR_USERNAME/finance-predictor/issues">Request Feature</a> •
    <a href="MCP_GUIDE.md">MCP Documentation</a>
  </p>
</div>
