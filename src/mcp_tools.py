"""
MCP Tools - Tool implementations for the Finance Predictor MCP Server.

This module provides all the tool functions that can be called by MCP clients.
Each tool bridges MCP calls to existing finance-predictor functionality.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime

from .data_loader import DataLoader
from .forecasting import ForecastEngine
from .sentiment import SentimentAnalyzer
from .quant_analysis import QuantAnalyzer
from .llm_engine import LLMEngine
from .utils import get_logger

logger = get_logger(__name__)

# ============================================================================
# Parameter Models (Pydantic schemas for type validation)
# ============================================================================

class StockDataParams(BaseModel):
    """Parameters for fetching stock data."""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., AAPL, TSLA, BTC-USD)")
    period: str = Field("2y", description="Data period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max")
    interval: str = Field("1d", description="Data interval: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo")


class PredictionParams(BaseModel):
    """Parameters for running predictions."""
    ticker: str = Field(..., description="Stock ticker symbol")
    days: int = Field(30, description="Number of days to forecast", ge=1, le=365)
    models: Optional[List[str]] = Field(
        None, 
        description="List of models to use. If None, uses all available models. Options: SMA (20), EMA (12), Linear Regression, Random Forest, XGBoost, SVR, Holt-Winters, ARIMA, RSI Trend, Bollinger Trend, CNN-GAF"
    )


class SentimentParams(BaseModel):
    """Parameters for sentiment analysis."""
    ticker: str = Field(..., description="Stock ticker symbol")


class RiskMetricsParams(BaseModel):
    """Parameters for risk metric calculation."""
    ticker: str = Field(..., description="Stock ticker symbol")
    period: str = Field("1y", description="Historical period for risk calculation")


class PatternDetectionParams(BaseModel):
    """Parameters for candlestick pattern detection."""
    ticker: str = Field(..., description="Stock ticker symbol")
    period: str = Field("3mo", description="Historical period for pattern detection")


class TechnicalIndicatorsParams(BaseModel):
    """Parameters for technical indicators."""
    ticker: str = Field(..., description="Stock ticker symbol")
    indicators: List[str] = Field(
        ["RSI", "MACD", "Bollinger Bands"],
        description="List of indicators to calculate. Options: RSI, MACD, Bollinger Bands, ATR, Stochastic, Ichimoku, ADX"
    )
    period: str = Field("6mo", description="Historical period for indicator calculation")


class AIAnalysisParams(BaseModel):
    """Parameters for AI analysis."""
    ticker: str = Field(..., description="Stock ticker symbol")
    prompt: str = Field(..., description="Analysis prompt for the AI model")
    include_data: bool = Field(True, description="Whether to include recent stock data in the context")


class CorrelationParams(BaseModel):
    """Parameters for correlation matrix."""
    tickers: List[str] = Field(..., description="List of ticker symbols to analyze")
    benchmark: str = Field("SPY", description="Benchmark ticker for comparison")
    period: str = Field("1y", description="Historical period for correlation analysis")


class MarketRegimeParams(BaseModel):
    """Parameters for market regime detection."""
    ticker: str = Field(..., description="Stock ticker symbol")
    period: str = Field("3mo", description="Historical period for regime detection")


# ============================================================================
# Tool Implementations
# ============================================================================

async def fetch_stock_data(params: StockDataParams) -> Dict[str, Any]:
    """
    Fetch historical stock data for a given ticker.
    
    Returns OHLCV (Open, High, Low, Close, Volume) data for the specified period and interval.
    """
    try:
        logger.info(f"MCP: Fetching stock data for {params.ticker}")
        df = DataLoader.fetch_history(params.ticker, params.period, params.interval)
        
        if df.empty:
            return {
                "success": False,
                "error": f"No data found for {params.ticker}. Check if the ticker symbol is valid."
            }
        
        # Convert DataFrame to JSON-serializable format
        data_records = df.reset_index().to_dict(orient="records")
        
        # Convert Timestamp objects to strings
        for record in data_records:
            if 'Date' in record and hasattr(record['Date'], 'isoformat'):
                record['Date'] = record['Date'].isoformat()
        
        return {
            "success": True,
            "ticker": params.ticker,
            "period": params.period,
            "interval": params.interval,
            "data_points": len(data_records),
            "columns": df.columns.tolist(),
            "latest_price": float(df['Close'].iloc[-1]) if 'Close' in df.columns else None,
            "data": data_records[-100:]  # Return last 100 data points to avoid overwhelming output
        }
    except Exception as e:
        logger.error(f"Error fetching stock data: {e}")
        return {"success": False, "error": str(e)}


async def run_predictions(params: PredictionParams) -> Dict[str, Any]:
    """
    Run forecasting models on a stock ticker.
    
    Executes one or more prediction models and returns forecasted prices for the specified number of days.
    Available models include statistical (SMA, EMA, ARIMA), machine learning (Random Forest, XGBoost), 
    and advanced methods (CNN-GAF, Prophet).
    """
    try:
        logger.info(f"MCP: Running predictions for {params.ticker} ({params.days} days)")
        
        # Fetch historical data
        df = DataLoader.fetch_history(params.ticker, period="2y")
        if df.empty:
            return {
                "success": False,
                "error": f"No historical data available for {params.ticker}"
            }
        
        # Initialize forecast engine
        engine = ForecastEngine()
        
        # Get predictions
        if params.models:
            results = {}
            for model_name in params.models:
                if model_name in engine.models:
                    try:
                        pred = engine.models[model_name](df, params.days)
                        results[model_name] = pred.tolist() if isinstance(pred, np.ndarray) else pred
                    except Exception as e:
                        results[model_name] = {"error": str(e)}
                else:
                    results[model_name] = {"error": f"Model '{model_name}' not found"}
        else:
            # Run all models
            results_df = engine.get_all_predictions(df, params.days)
            results = {col: results_df[col].tolist() for col in results_df.columns}
        
        current_price = float(df['Close'].iloc[-1])
        
        return {
            "success": True,
            "ticker": params.ticker,
            "forecast_days": params.days,
            "current_price": current_price,
            "predictions": results,
            "models_used": list(results.keys())
        }
    except Exception as e:
        logger.error(f"Error running predictions: {e}")
        return {"success": False, "error": str(e)}


async def analyze_sentiment(params: SentimentParams) -> Dict[str, Any]:
    """
    Analyze sentiment from recent news articles about a stock.
    
    Fetches news articles and performs sentiment analysis using VADER (Valence Aware Dictionary 
    and sEntiment Reasoner). Returns sentiment scores and article summaries.
    """
    try:
        logger.info(f"MCP: Analyzing sentiment for {params.ticker}")
        
        # Fetch news
        news = DataLoader.fetch_news(params.ticker)
        if not news:
            return {
                "success": False,
                "error": f"No news articles found for {params.ticker}"
            }
        
        # Analyze sentiment
        analyzer = SentimentAnalyzer()
        sentiments = []
        total_score = 0
        
        for article in news:
            score = analyzer.analyze(article['title'])
            sentiments.append({
                "title": article['title'],
                "publisher": article['publisher'],
                "sentiment_score": score,
                "sentiment_label": "Positive" if score > 0.05 else "Negative" if score < -0.05 else "Neutral",
                "is_reliable": article.get('is_reliable', False)
            })
            total_score += score
        
        avg_sentiment = total_score / len(sentiments) if sentiments else 0
        
        return {
            "success": True,
            "ticker": params.ticker,
            "articles_analyzed": len(sentiments),
            "average_sentiment": avg_sentiment,
            "sentiment_label": "Positive" if avg_sentiment > 0.05 else "Negative" if avg_sentiment < -0.05 else "Neutral",
            "articles": sentiments
        }
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {e}")
        return {"success": False, "error": str(e)}


async def calculate_risk_metrics(params: RiskMetricsParams) -> Dict[str, Any]:
    """
    Calculate risk metrics for a stock.
    
    Computes Value at Risk (VaR), Conditional Value at Risk (CVaR), volatility, 
    Sharpe ratio, and Kelly Criterion for optimal position sizing.
    """
    try:
        logger.info(f"MCP: Calculating risk metrics for {params.ticker}")
        
        df = DataLoader.fetch_history(params.ticker, period=params.period)
        if df.empty:
            return {
                "success": False,
                "error": f"No data available for {params.ticker}"
            }
        
        analyzer = QuantAnalyzer()
        metrics = analyzer.calculate_risk_metrics(df)
        
        return {
            "success": True,
            "ticker": params.ticker,
            "period": params.period,
            "metrics": {
                "var_95": float(metrics.get('var_95', 0)),
                "cvar_95": float(metrics.get('cvar_95', 0)),
                "volatility": float(metrics.get('volatility', 0)),
                "sharpe_ratio": float(metrics.get('sharpe_ratio', 0)),
                "kelly_criterion": float(metrics.get('kelly_criterion', 0))
            }
        }
    except Exception as e:
        logger.error(f"Error calculating risk metrics: {e}")
        return {"success": False, "error": str(e)}


async def detect_candlestick_patterns(params: PatternDetectionParams) -> Dict[str, Any]:
    """
    Detect candlestick patterns in recent price action.
    
    Identifies common patterns like Doji, Hammer, Shooting Star, Engulfing, and Morning/Evening Star.
    """
    try:
        logger.info(f"MCP: Detecting patterns for {params.ticker}")
        
        df = DataLoader.fetch_history(params.ticker, period=params.period)
        if df.empty:
            return {
                "success": False,
                "error": f"No data available for {params.ticker}"
            }
        
        analyzer = QuantAnalyzer()
        patterns = analyzer.detect_patterns(df)
        
        return {
            "success": True,
            "ticker": params.ticker,
            "period": params.period,
            "patterns_detected": patterns
        }
    except Exception as e:
        logger.error(f"Error detecting patterns: {e}")
        return {"success": False, "error": str(e)}


async def get_technical_indicators(params: TechnicalIndicatorsParams) -> Dict[str, Any]:
    """
    Calculate technical indicators for a stock.
    
    Computes various technical analysis indicators including RSI, MACD, Bollinger Bands, 
    ATR, Stochastic Oscillator, Ichimoku Cloud, and ADX.
    """
    try:
        logger.info(f"MCP: Calculating indicators for {params.ticker}")
        
        df = DataLoader.fetch_history(params.ticker, period=params.period)
        if df.empty:
            return {
                "success": False,
                "error": f"No data available for {params.ticker}"
            }
        
        analyzer = QuantAnalyzer()
        indicators = {}
        
        for indicator in params.indicators:
            try:
                if indicator == "RSI":
                    indicators["RSI"] = float(analyzer.calculate_rsi(df).iloc[-1])
                elif indicator == "MACD":
                    macd_data = analyzer.calculate_macd(df)
                    indicators["MACD"] = {
                        "macd": float(macd_data['macd'].iloc[-1]),
                        "signal": float(macd_data['signal'].iloc[-1]),
                        "histogram": float(macd_data['histogram'].iloc[-1])
                    }
                elif indicator == "Bollinger Bands":
                    bb_data = analyzer.calculate_bollinger_bands(df)
                    indicators["Bollinger Bands"] = {
                        "upper": float(bb_data['upper'].iloc[-1]),
                        "middle": float(bb_data['middle'].iloc[-1]),
                        "lower": float(bb_data['lower'].iloc[-1])
                    }
                elif indicator == "ADX":
                    indicators["ADX"] = float(analyzer.calculate_adx(df).iloc[-1])
                # Add more indicators as needed
            except Exception as e:
                indicators[indicator] = {"error": str(e)}
        
        return {
            "success": True,
            "ticker": params.ticker,
            "period": params.period,
            "indicators": indicators
        }
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return {"success": False, "error": str(e)}


async def run_ai_analysis(params: AIAnalysisParams) -> Dict[str, Any]:
    """
    Run AI-powered analysis on a stock using the local Granite 4.0 LLM.
    
    Provides deep financial analysis, market insights, and strategic recommendations 
    using an advanced reasoning model.
    """
    try:
        logger.info(f"MCP: Running AI analysis for {params.ticker}")
        
        # Build context
        context = f"Analyze {params.ticker}.\n\n"
        
        if params.include_data:
            df = DataLoader.fetch_history(params.ticker, period="3mo")
            if not df.empty:
                current_price = df['Close'].iloc[-1]
                price_change = ((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100
                context += f"Current Price: ${current_price:.2f}\n"
                context += f"3-Month Change: {price_change:+.2f}%\n\n"
        
        context += f"User Question: {params.prompt}"
        
        # Run LLM analysis
        engine = LLMEngine()
        result = engine.analyze(context)
        
        if "error" in result:
            return {
                "success": False,
                "error": result["error"]
            }
        
        return {
            "success": True,
            "ticker": params.ticker,
            "analysis": result.get("response", ""),
            "thinking": result.get("thinking", ""),
            "strategy": result.get("strategy", "")
        }
    except Exception as e:
        logger.error(f"Error running AI analysis: {e}")
        return {"success": False, "error": str(e)}


async def get_correlation_matrix(params: CorrelationParams) -> Dict[str, Any]:
    """
    Calculate correlation matrix between multiple tickers.
    
    Analyzes how different assets move in relation to each other and a benchmark (default: SPY).
    Useful for portfolio diversification analysis.
    """
    try:
        logger.info(f"MCP: Calculating correlation matrix for {params.tickers}")
        
        # Fetch data for all tickers
        all_tickers = params.tickers + [params.benchmark]
        price_data = {}
        
        for ticker in all_tickers:
            df = DataLoader.fetch_history(ticker, period=params.period)
            if not df.empty:
                price_data[ticker] = df['Close']
        
        if len(price_data) < 2:
            return {
                "success": False,
                "error": "Need at least 2 valid tickers to calculate correlation"
            }
        
        # Create DataFrame and calculate correlation
        combined_df = pd.DataFrame(price_data)
        correlation_matrix = combined_df.corr()
        
        # Convert to dict
        corr_dict = correlation_matrix.to_dict()
        
        return {
            "success": True,
            "tickers": params.tickers,
            "benchmark": params.benchmark,
            "period": params.period,
            "correlation_matrix": corr_dict
        }
    except Exception as e:
        logger.error(f"Error calculating correlation: {e}")
        return {"success": False, "error": str(e)}


async def get_market_regime(params: MarketRegimeParams) -> Dict[str, Any]:
    """
    Detect current market regime (Trending vs Ranging) using ADX.
    
    Uses Average Directional Index (ADX) to determine if the market is in a trending 
    or ranging state. ADX > 25 typically indicates a trending market.
    """
    try:
        logger.info(f"MCP: Detecting market regime for {params.ticker}")
        
        df = DataLoader.fetch_history(params.ticker, period=params.period)
        if df.empty:
            return {
                "success": False,
                "error": f"No data available for {params.ticker}"
            }
        
        analyzer = QuantAnalyzer()
        adx_values = analyzer.calculate_adx(df)
        current_adx = float(adx_values.iloc[-1])
        
        regime = "Trending" if current_adx > 25 else "Ranging"
        strength = "Strong" if current_adx > 40 else "Moderate" if current_adx > 25 else "Weak"
        
        return {
            "success": True,
            "ticker": params.ticker,
            "period": params.period,
            "adx": current_adx,
            "regime": regime,
            "trend_strength": strength,
            "interpretation": f"The market is currently {regime.lower()} with {strength.lower()} directional movement (ADX: {current_adx:.2f})"
        }
    except Exception as e:
        logger.error(f"Error detecting market regime: {e}")
        return {"success": False, "error": str(e)}
