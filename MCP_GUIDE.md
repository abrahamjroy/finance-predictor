# Finance Predictor - MCP Integration Guide

This guide provides detailed information on using the Finance Predictor MCP (Model Context Protocol) server with AI assistants like Claude Desktop, ChatGPT, and other MCP-compatible clients.

## Table of Contents

- [Overview](#overview)
- [Setup Instructions](#setup-instructions)
- [Available Tools](#available-tools)
- [Available Resources](#available-resources)
- [Available Prompts](#available-prompts)
- [Usage Examples](#usage-examples)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)

## Overview

The Finance Predictor MCP server exposes all of the application's financial analysis capabilities through a standardized protocol that AI assistants can interact with. This enables:

- **Automated Analysis**: AI can fetch data, run predictions, and analyze sentiment without manual intervention
- **Multi-Tool Workflows**: Chain multiple analysis tools together for comprehensive market insights
- **Real-Time Data**: Access live stock prices, news, and market data through MCP resources
- **Intelligent Prompting**: Use pre-built templates for common financial analysis tasks

## Setup Instructions

### 1. Install Dependencies

First, ensure all MCP dependencies are installed:

```bash
cd c:\Users\royka\Documents\Projects\finance-predictor
pip install -r requirements.txt
```

This will install `fastmcp`, `mcp`, and `pydantic` along with all other dependencies.

### 2. Test the Server

Run the MCP server directly to verify it works:

```bash
python -m src.mcp_server
```

The server should start and display connection information. Press `Ctrl+C` to stop.

### 3. Configure Claude Desktop

To use Finance Predictor with Claude Desktop:

1. Open your Claude Desktop config file:
   - **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
   - **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **Linux**: `~/.config/Claude/claude_desktop_config.json`

2. Add the Finance Predictor server configuration:

```json
{
  "mcpServers": {
    "finance-predictor": {
      "command": "python",
      "args": ["-m", "src.mcp_server"],
      "cwd": "c:/Users/royka/Documents/Projects/finance-predictor"
    }
  }
}
```

**Note**: Update the `cwd` path if your installation is in a different location.

3. Restart Claude Desktop

4. Verify the integration by looking for the "Finance Predictor Pro" server in Claude's settings or by asking Claude: *"What MCP servers are available?"*

### 4. (Optional) Test with MCP Inspector

For debugging and interactive testing, use the MCP Inspector:

```bash
npx @modelcontextprotocol/inspector python -m src.mcp_server
```

This opens a web UI where you can test all tools and resources.

## Available Tools

### 📈 `fetch_stock_data_tool`

Fetch historical OHLCV (Open, High, Low, Close, Volume) data for any ticker.

**Parameters:**
- `ticker` (str, required): Stock ticker symbol (e.g., "AAPL", "TSLA", "BTC-USD")
- `period` (str, default: "2y"): Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
- `interval` (str, default: "1d"): Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)

**Returns:** Historical data with latest price and metadata

**Example Usage in Claude:**
> "Fetch 1 year of daily stock data for AAPL"

---

### 🔮 `run_predictions_tool`

Run multiple forecasting models on a stock.

**Parameters:**
- `ticker` (str, required): Stock ticker symbol
- `days` (int, default: 30): Number of days to forecast (1-365)
- `models` (list[str], optional): Specific models to use. If None, runs all models.

**Available Models:**
- Statistical: SMA (20), EMA (12), ARIMA, Holt-Winters
- Machine Learning: Linear Regression, Random Forest, XGBoost, SVR
- Advanced: CNN-GAF, Prophet, Ensemble

**Returns:** Predictions from each model with current price and metadata

**Example Usage in Claude:**
> "Run 30-day predictions on TSLA using XGBoost and Random Forest models"

---

### 💬 `analyze_sentiment_tool`

Analyze sentiment from recent news articles.

**Parameters:**
- `ticker` (str, required): Stock ticker symbol

**Returns:** Sentiment scores, article summaries, and overall sentiment classification

**Example Usage in Claude:**
> "Analyze news sentiment for NVDA"

---

### ⚠️ `calculate_risk_metrics_tool`

Calculate comprehensive risk metrics for a stock.

**Parameters:**
- `ticker` (str, required): Stock ticker symbol
- `period` (str, default: "1y"): Historical period for calculation

**Returns:** VaR (95%), CVaR, volatility, Sharpe ratio, Kelly Criterion

**Example Usage in Claude:**
> "Calculate risk metrics for MSFT over the past 6 months"

---

### 📊 `detect_candlestick_patterns_tool`

Detect technical chart patterns.

**Parameters:**
- `ticker` (str, required): Stock ticker symbol
- `period` (str, default: "3mo"): Historical period for pattern detection

**Returns:** Detected patterns (Doji, Hammer, Shooting Star, Engulfing, Morning/Evening Star)

**Example Usage in Claude:**
> "Detect candlestick patterns in GME over the last month"

---

### 📉 `get_technical_indicators_tool`

Calculate technical analysis indicators.

**Parameters:**
- `ticker` (str, required): Stock ticker symbol
- `indicators` (list[str], default: ["RSI", "MACD", "Bollinger Bands"]): Indicators to calculate
- `period` (str, default: "6mo"): Historical period

**Available Indicators:** RSI, MACD, Bollinger Bands, ATR, Stochastic, Ichimoku, ADX

**Returns:** Current values for each requested indicator

**Example Usage in Claude:**
> "Get RSI, MACD, and ADX indicators for SPY"

---

### 🤖 `run_ai_analysis_tool`

Run local DeepSeek-R1-Distill-Qwen-1.5B LLM analysis on a stock.

**Parameters:**
- `ticker` (str, required): Stock ticker symbol
- `prompt` (str, required): Analysis question or request
- `include_data` (bool, default: True): Whether to include recent stock data in context

**Returns:** AI-generated analysis with thinking process and strategy

**Example Usage in Claude:**
> "Use the AI analysis tool to evaluate whether AAPL is a good buy right now based on technical and fundamental factors"

---

### 🔗 `get_correlation_matrix_tool`

Calculate correlation between multiple assets.

**Parameters:**
- `tickers` (list[str], required): List of ticker symbols
- `benchmark` (str, default: "SPY"): Benchmark for comparison
- `period` (str, default: "1y"): Historical period

**Returns:** Correlation matrix with all pairwise correlations

**Example Usage in Claude:**
> "Calculate correlation matrix for AAPL, MSFT, GOOGL, and AMZN against SPY"

---

### 📍 `get_market_regime_tool`

Detect if market is trending or ranging.

**Parameters:**
- `ticker` (str, required): Stock ticker symbol
- `period` (str, default: "3mo"): Historical period

**Returns:** ADX value, regime classification (Trending/Ranging), and interpretation

**Example Usage in Claude:**
> "What's the current market regime for Bitcoin (BTC-USD)?"

---

## Available Resources

Resources provide live data feeds that can be accessed by URI.

### `stock://historical/{ticker}`

Historical OHLCV data for the past year in markdown table format. Includes 52-week high/low and latest price.

**Example:** `stock://historical/AAPL`

---

### `stock://info/{ticker}`

Company metadata including sector, industry, market cap, PE ratio, dividend yield, and business summary.

**Example:** `stock://info/MSFT`

---

### `stock://news/{ticker}`

Recent news articles with sentiment analysis. Shows headline, publisher, sentiment score, and reliability indicator.

**Example:** `stock://news/TSLA`

---

### `stock://metrics/{ticker}`

Quantitative analysis including risk metrics (VaR, CVaR, volatility, Sharpe), market regime (ADX), and detected candlestick patterns.

**Example:** `stock://metrics/NVDA`

---

## Available Prompts

Prompts provide templates for structured analysis workflows.

### `market_analysis`

Comprehensive 6-step market analysis workflow:
1. Current market conditions
2. Technical analysis
3. Fundamental analysis
4. Risk assessment
5. Multi-model forecast
6. Investment recommendation

**Parameters:** `ticker` (str)

**Example Usage in Claude:**
> "Run the market_analysis prompt for GOOGL"

---

### `risk_assessment`

Thorough 6-step risk evaluation:
1. Volatility analysis
2. Downside risk metrics
3. Risk-adjusted performance
4. Sentiment & news risk
5. Position sizing
6. Risk mitigation strategies

**Parameters:** `ticker` (str)

---

### `investment_recommendation`

Structured buy/sell/hold decision framework:
1. Quick summary with conviction level
2. Bullish factors
3. Bearish factors
4. Price targets (conservative, base, optimistic)
5. Action plan (entry, exit, stop loss)
6. Monitoring plan

**Parameters:** `ticker` (str), `time_horizon` (str, default: "medium")

---

### `portfolio_optimization`

Multi-asset portfolio optimization workflow:
1. Correlation analysis
2. Individual asset analysis
3. Risk-return profile
4. Optimal allocation
5. Portfolio-level metrics
6. Rebalancing strategy

**Parameters:** `tickers` (list[str])

**Example Usage in Claude:**
> "Use the portfolio_optimization prompt for AAPL, MSFT, GOOGL, AMZN, and NVDA"

---

## Usage Examples

### Example 1: Quick Stock Analysis

**Prompt:**
> "Analyze AAPL: fetch its latest data, run predictions using XGBoost, check sentiment, and tell me if I should buy."

**What Claude will do:**
1. Call `fetch_stock_data_tool("AAPL")`
2. Call `run_predictions_tool("AAPL", models=["XGBoost"])`
3. Call `analyze_sentiment_tool("AAPL")`
4. Synthesize findings into buy/hold/sell recommendation

---

### Example 2: Portfolio Risk Comparison

**Prompt:**
> "Compare risk metrics between TSLA and SPY. Which one has better risk-adjusted returns?"

**What Claude will do:**
1. Call `calculate_risk_metrics_tool("TSLA")`
2. Call `calculate_risk_metrics_tool("SPY")`
3. Call `get_correlation_matrix_tool(["TSLA", "SPY"])`
4. Compare Sharpe ratios and provide analysis

---

### Example 3: Market Regime Strategy

**Prompt:**
> "Check if QQQ is trending or ranging, then suggest appropriate trading strategies based on the regime."

**What Claude will do:**
1. Call `get_market_regime_tool("QQQ")`
2. Based on ADX value, recommend trend-following or mean-reversion strategies

---

### Example 4: Multi-Model Consensus Forecast

**Prompt:**
> "Run all available prediction models on NVDA for 60 days and tell me which models agree on the direction."

**What Claude will do:**
1. Call `run_predictions_tool("NVDA", days=60)` (no models specified = run all)
2. Analyze consensus across 10+ models
3. Identify outliers and report confidence level

---

### Example 5: Comprehensive Analysis with Template

**Prompt:**
> "Use the market_analysis template to thoroughly analyze Microsoft (MSFT)."

**What Claude will do:**
1. Use the `market_analysis` prompt template
2. Automatically orchestrate multiple tool calls following the 6-step workflow
3. Provide a complete, structured analysis report

---

## Troubleshooting

### Server won't start

**Issue:** `ModuleNotFoundError` when running `python -m src.mcp_server`

**Solution:**
1. Ensure you're in the correct directory: `cd c:\Users\royka\Documents\Projects\finance-predictor`
2. Verify dependencies are installed: `pip install fastmcp mcp pydantic`
3. Check Python version (requires 3.10+): `python --version`

---

### Claude doesn't see the server

**Issue:** Finance Predictor doesn't appear in Claude Desktop

**Solution:**
1. Verify the config file path is correct
2. Check that the `cwd` in `mcp_config.json` matches your installation directory
3. Restart Claude Desktop completely (quit and reopen)
4. Check Claude Desktop logs for errors

---

### Tools return "No data found"

**Issue:** Tools return errors about missing data for valid tickers

**Solution:**
1. Check your internet connection (yfinance requires network access)
2. Verify the ticker symbol is correct (try it on Yahoo Finance first)
3. Some tickers may have limited historical data - try a shorter period

---

### Slow performance

**Issue:** Tools take a long time to respond

**Solution:**
1. Predictions tools (especially CNN-GAF) can be slow - this is normal
2. Use specific models instead of running all: `models=["XGBoost", "Random Forest"]`
3. Reduce forecast days: `days=30` instead of `days=365`
4. Resources are cached for 5 minutes to improve performance

---

## Advanced Usage

### Chaining Multiple Analyses

You can ask Claude to perform complex, multi-step analyses:

> "For each of these tickers: AAPL, MSFT, GOOGL  
> 1. Fetch 1 year of data  
> 2. Calculate risk metrics  
> 3. Run XGBoost predictions for 30 days  
> 4. Analyze sentiment  
> 5. Recommend optimal portfolio weights based on Sharpe ratios and correlations"

Claude will orchestrate ~15 tool calls and synthesize the results.

---

### Using Resources Directly

Ask Claude to access resources without explicit tool calls:

> "Show me the latest news for Tesla from the stock://news/TSLA resource"

Claude will access the resource and display the markdown-formatted news feed with sentiment.

---

### Custom Analysis Scripts

You can guide Claude to create custom analysis workflows:

> "Create a momentum trading strategy for SPY:  
> 1. Check if it's in a trending regime (ADX)  
> 2. If trending, get RSI and MACD  
> 3. Identify entry signals  
> 4. Calculate position size using Kelly Criterion  
> 5. Set stop loss at 2x ATR below entry"

---

### Integration with Other MCP Servers

If you have multiple MCP servers configured (e.g., web search, database access), Claude can combine them:

> "Search the web for recent NVDA earnings reports, then use finance-predictor to analyze NVDA's price reaction and predict next week's movement"

---

## Best Practices

1. **Be Specific**: Instead of "analyze this stock", say "analyze AAPL using XGBoost predictions, RSI indicator, and news sentiment"

2. **Use Templates**: The prompt templates (market_analysis, risk_assessment, etc.) provide structured workflows

3. **Specify Models**: For predictions, specify which models to use to save time: `models=["XGBoost", "Random Forest"]`

4. **Batch Requests**: Analyze multiple tickers in one prompt instead of separate conversations

5. **Resource Awareness**: Remember resources are cached for 5 minutes - repeated queries are fast

6. **Error Handling**: If a tool fails, Claude can retry with different parameters or suggest alternatives

---

## Support

For issues or questions:
1. Check the [main README](README.md) for general setup
2. Review [HARDWARE_CONFIG.md](HARDWARE_CONFIG.md) for system requirements
3. Open an issue on GitHub with MCP server logs

---

**Happy analyzing! 📊🚀**
