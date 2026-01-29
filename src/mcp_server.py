"""
Finance Predictor MCP Server

This is the main MCP (Model Context Protocol) server that exposes all finance-predictor
capabilities to MCP clients like Claude Desktop, ChatGPT, and other AI assistants.

The server provides:
- Resources: Live financial data feeds (stock prices, news, metrics)
- Tools: Actionable functions (predictions, sentiment analysis, risk calculations)
- Prompts: Pre-built templates for common financial analysis tasks

Usage:
    python -m src.mcp_server
    
Or configure in Claude Desktop's config file.
"""

import asyncio
from typing import Any
from fastmcp import FastMCP

# Import tool implementations
from .mcp_tools import (
    fetch_stock_data,
    run_predictions,
    analyze_sentiment,
    calculate_risk_metrics,
    detect_candlestick_patterns,
    get_technical_indicators,
    run_ai_analysis,
    get_correlation_matrix,
    get_market_regime,
    StockDataParams,
    PredictionParams,
    SentimentParams,
    RiskMetricsParams,
    PatternDetectionParams,
    TechnicalIndicatorsParams,
    AIAnalysisParams,
    CorrelationParams,
    MarketRegimeParams,
)

# Import resource handlers
from .mcp_resources import handle_resource_request

from .utils import get_logger, setup_dirs

logger = get_logger(__name__)

# Initialize FastMCP server
mcp = FastMCP("Finance Predictor Pro")

# ============================================================================
# Register Tools
# ============================================================================

@mcp.tool()
async def fetch_stock_data_tool(
    ticker: str,
    period: str = "2y",
    interval: str = "1d"
) -> dict[str, Any]:
    """
    Fetch historical stock data for analysis.
    
    Args:
        ticker: Stock ticker symbol (e.g., AAPL, TSLA, BTC-USD, ^GSPC)
        period: Data period - Options: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        interval: Data interval - Options: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
    
    Returns:
        Dictionary containing historical OHLCV data, latest price, and metadata
    """
    params = StockDataParams(ticker=ticker, period=period, interval=interval)
    return await fetch_stock_data(params)


@mcp.tool()
async def run_predictions_tool(
    ticker: str,
    days: int = 30,
    models: list[str] | None = None
) -> dict[str, Any]:
    """
    Run forecasting models on a stock ticker.
    
    Executes prediction algorithms and returns forecasted prices. Available models include:
    - Statistical: SMA (20), EMA (12), ARIMA, Holt-Winters
    - Machine Learning: Linear Regression, Random Forest, XGBoost, SVR
    - Advanced: CNN-GAF (Computer Vision), Prophet, Ensemble
    
    Args:
        ticker: Stock ticker symbol
        days: Number of days to forecast (1-365)
        models: List of specific models to use. If None, runs all available models.
    
    Returns:
        Dictionary containing predictions from each model and metadata
    """
    params = PredictionParams(ticker=ticker, days=days, models=models)
    return await run_predictions(params)


@mcp.tool()
async def analyze_sentiment_tool(ticker: str) -> dict[str, Any]:
    """
    Analyze sentiment from recent news articles about a stock.
    
    Fetches recent news and performs VADER sentiment analysis. Prioritizes reliable 
    financial news sources (Reuters, Bloomberg, CNBC, WSJ, etc.).
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Dictionary containing sentiment scores, article summaries, and overall sentiment
    """
    params = SentimentParams(ticker=ticker)
    return await analyze_sentiment(params)


@mcp.tool()
async def calculate_risk_metrics_tool(
    ticker: str,
    period: str = "1y"
) -> dict[str, Any]:
    """
    Calculate comprehensive risk metrics for a stock.
    
    Computes industry-standard risk measures:
    - VaR (Value at Risk) at 95% confidence
    - CVaR (Conditional Value at Risk / Expected Shortfall)
    - Annualized Volatility
    - Sharpe Ratio (risk-adjusted returns)
    - Kelly Criterion (optimal position sizing)
    
    Args:
        ticker: Stock ticker symbol
        period: Historical period for calculation (1mo, 3mo, 6mo, 1y, 2y, 5y)
    
    Returns:
        Dictionary containing all risk metrics
    """
    params = RiskMetricsParams(ticker=ticker, period=period)
    return await calculate_risk_metrics(params)


@mcp.tool()
async def detect_candlestick_patterns_tool(
    ticker: str,
    period: str = "3mo"
) -> dict[str, Any]:
    """
    Detect candlestick patterns in recent price action.
    
    Identifies common technical patterns:
    - Doji (indecision)
    - Hammer (potential reversal)
    - Shooting Star (bearish reversal)
    - Engulfing (strong reversal signal)
    - Morning/Evening Star (trend reversal)
    
    Args:
        ticker: Stock ticker symbol
        period: Historical period for pattern detection
    
    Returns:
        Dictionary containing detected patterns with timestamps
    """
    params = PatternDetectionParams(ticker=ticker, period=period)
    return await detect_candlestick_patterns(params)


@mcp.tool()
async def get_technical_indicators_tool(
    ticker: str,
    indicators: list[str] = ["RSI", "MACD", "Bollinger Bands"],
    period: str = "6mo"
) -> dict[str, Any]:
    """
    Calculate technical indicators for a stock.
    
    Available indicators:
    - RSI: Relative Strength Index (overbought/oversold)
    - MACD: Moving Average Convergence Divergence (momentum)
    - Bollinger Bands: Volatility bands
    - ATR: Average True Range (volatility measure)
    - Stochastic: Stochastic Oscillator
    - Ichimoku: Ichimoku Cloud
    - ADX: Average Directional Index (trend strength)
    
    Args:
        ticker: Stock ticker symbol
        indicators: List of indicators to calculate
        period: Historical period for calculation
    
    Returns:
        Dictionary containing current values for each requested indicator
    """
    params = TechnicalIndicatorsParams(ticker=ticker, indicators=indicators, period=period)
    return await get_technical_indicators(params)


@mcp.tool()
async def run_ai_analysis_tool(
    ticker: str,
    prompt: str,
    include_data: bool = True
) -> dict[str, Any]:
    """
    Run AI-powered deep analysis using local Granite 4.0 LLM.
    
    Provides advanced reasoning and strategic insights on market conditions, 
    investment opportunities, and risk assessment. The AI has access to the 
    ticker's recent price data and can provide context-aware analysis.
    
    Args:
        ticker: Stock ticker symbol
        prompt: Analysis question or request
        include_data: Whether to include recent stock data in the AI's context
    
    Returns:
        Dictionary containing AI-generated analysis, thinking process, and strategy
    """
    params = AIAnalysisParams(ticker=ticker, prompt=prompt, include_data=include_data)
    return await run_ai_analysis(params)


@mcp.tool()
async def get_correlation_matrix_tool(
    tickers: list[str],
    benchmark: str = "SPY",
    period: str = "1y"
) -> dict[str, Any]:
    """
    Calculate correlation matrix between multiple assets.
    
    Analyzes how different assets move in relation to each other and a benchmark.
    Essential for portfolio diversification and risk management.
    
    Args:
        tickers: List of ticker symbols to analyze
        benchmark: Benchmark ticker for comparison (default: SPY for S&P 500)
        period: Historical period for correlation analysis
    
    Returns:
        Dictionary containing correlation matrix with all pairwise correlations
    """
    params = CorrelationParams(tickers=tickers, benchmark=benchmark, period=period)
    return await get_correlation_matrix(params)


@mcp.tool()
async def get_market_regime_tool(
    ticker: str,
    period: str = "3mo"
) -> dict[str, Any]:
    """
    Detect current market regime (Trending vs Ranging).
    
    Uses Average Directional Index (ADX) to classify market state:
    - Trending (ADX > 25): Strong directional movement, trend-following strategies work
    - Ranging (ADX < 25): Sideways movement, mean-reversion strategies work
    
    Args:
        ticker: Stock ticker symbol
        period: Historical period for regime detection
    
    Returns:
        Dictionary containing ADX value, regime classification, and interpretation
    """
    params = MarketRegimeParams(ticker=ticker, period=period)
    return await get_market_regime(params)


# ============================================================================
# Register Resources
# ============================================================================

@mcp.resource("stock://historical/{ticker}")
async def historical_data_resource(ticker: str) -> str:
    """Historical OHLCV data for a stock (last 1 year, daily)."""
    return await handle_resource_request(f"stock://historical/{ticker}")


@mcp.resource("stock://info/{ticker}")
async def ticker_info_resource(ticker: str) -> str:
    """Ticker metadata, sector, industry, market cap, PE ratio, etc."""
    return await handle_resource_request(f"stock://info/{ticker}")


@mcp.resource("stock://news/{ticker}")
async def news_resource(ticker: str) -> str:
    """Recent news articles with sentiment analysis."""
    return await handle_resource_request(f"stock://news/{ticker}")


@mcp.resource("stock://metrics/{ticker}")
async def metrics_resource(ticker: str) -> str:
    """Quantitative analysis, risk metrics, and detected patterns."""
    return await handle_resource_request(f"stock://metrics/{ticker}")


# ============================================================================
# Register Prompts
# ============================================================================

@mcp.prompt()
async def market_analysis(ticker: str) -> list[dict[str, str]]:
    """Comprehensive market analysis template for a stock."""
    return [
        {
            "role": "user",
            "content": f"""Perform a comprehensive market analysis for {ticker}. Include:

1. **Current Market Conditions**
   - Fetch current price and recent performance
   - Analyze market regime (trending vs ranging)
   - Identify key technical levels

2. **Technical Analysis**
   - Calculate and interpret RSI, MACD, Bollinger Bands
   - Detect any candlestick patterns
   - Assess trend strength using ADX

3. **Fundamental Analysis**
   - Review company information and sector
   - Analyze news sentiment
   - Consider market cap and valuation metrics

4. **Risk Assessment**
   - Calculate VaR, CVaR, and volatility
   - Determine appropriate position sizing (Kelly Criterion)
   - Identify key risk factors

5. **Forecast**
   - Run multiple prediction models (XGBoost, Random Forest, ARIMA)
   - Compare model consensus
   - Provide price targets for 30, 60, and 90 days

6. **Investment Recommendation**
   - Synthesize all findings
   - Provide clear buy/sell/hold recommendation
   - Suggest entry/exit points and stop losses

Use the available tools to gather all necessary data and provide a data-driven analysis."""
        }
    ]


@mcp.prompt()
async def risk_assessment(ticker: str) -> list[dict[str, str]]:
    """Risk evaluation template for a stock or portfolio."""
    return [
        {
            "role": "user",
            "content": f"""Conduct a thorough risk assessment for {ticker}:

1. **Volatility Analysis**
   - Calculate historical volatility
   - Compare to sector and market benchmarks
   - Identify volatility regimes

2. **Downside Risk Metrics**
   - Value at Risk (VaR 95%)
   - Conditional Value at Risk (CVaR)
   - Maximum drawdown

3. **Risk-Adjusted Performance**
   - Sharpe Ratio calculation
   - Risk-return trade-off analysis
   - Beta vs market

4. **Sentiment & News Risk**
   - Analyze recent news sentiment
   - Identify potential catalysts or red flags
   - Assess narrative risk

5. **Position Sizing**
   - Calculate optimal position size (Kelly Criterion)
   - Recommend allocation percentage
   - Define risk limits

6. **Risk Mitigation Strategies**
   - Suggest hedging approaches
   - Define stop-loss levels
   - Recommend portfolio diversification

Provide actionable risk management recommendations."""
        }
    ]


@mcp.prompt()
async def investment_recommendation(ticker: str, time_horizon: str = "medium") -> list[dict[str, str]]:
    """Investment recommendation template with buy/sell/hold decision."""
    return [
        {
            "role": "user",
            "content": f"""Generate an investment recommendation for {ticker} with a {time_horizon}-term horizon:

1. **Quick Summary**
   - Current price and trend
   - Overall recommendation (Buy/Sell/Hold)
   - Conviction level (High/Medium/Low)

2. **Bullish Factors**
   - Positive technical signals
   - Favorable sentiment
   - Strong fundamentals
   - Upside catalysts

3. **Bearish Factors**
   - Negative technical signals
   - Adverse sentiment
   - Fundamental concerns
   - Downside risks

4. **Price Targets**
   - Conservative target (30 days)
   - Base case target (60 days)
   - Optimistic target (90 days)

5. **Action Plan**
   - Entry strategy and price levels
   - Position sizing recommendation
   - Exit strategy (take profit targets)
   - Stop loss levels

6. **Monitoring Plan**
   - Key metrics to watch
   - Warning signals that would change thesis
   - Recommended review frequency

Be specific and data-driven in your recommendation."""
        }
    ]


@mcp.prompt()
async def portfolio_optimization(tickers: list[str]) -> list[dict[str, str]]:
    """Portfolio optimization template for multiple assets."""
    tickers_str = ", ".join(tickers)
    return [
        {
            "role": "user",
            "content": f"""Optimize a portfolio containing: {tickers_str}

1. **Correlation Analysis**
   - Calculate correlation matrix
   - Identify diversification opportunities
   - Flag highly correlated pairs

2. **Individual Asset Analysis**
   For each ticker:
   - Risk metrics (VaR, volatility, Sharpe)
   - Expected returns (based on predictions)
   - Market regime classification

3. **Risk-Return Profile**
   - Plot risk vs return for each asset
   - Identify efficient frontier candidates
   - Highlight outliers

4. **Optimal Allocation**
   - Recommend portfolio weights
   - Balance risk and return
   - Consider correlation benefits

5. **Portfolio-Level Metrics**
   - Expected portfolio return
   - Portfolio volatility
   - Portfolio Sharpe Ratio
   - Portfolio VaR

6. **Rebalancing Strategy**
   - Recommend rebalancing frequency
   - Define trigger points
   - Tax considerations

Provide a complete allocation strategy with rationale."""
        }
    ]


# ============================================================================
# Server Entry Point
# ============================================================================

def main():
    """Run the MCP server."""
    logger.info("🚀 Starting Finance Predictor MCP Server...")
    
    # Ensure required directories exist
    setup_dirs()
    
    # Run the server
    mcp.run()


if __name__ == "__main__":
    main()
