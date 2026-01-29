"""
MCP Resources - Resource handlers for the Finance Predictor MCP Server.

This module provides dynamic resource access for live financial data.
Resources are accessed via URIs and can be cached for performance.
"""

from typing import Dict, Optional
import pandas as pd
from datetime import datetime, timedelta

from .data_loader import DataLoader
from .quant_analysis import QuantAnalyzer
from .sentiment import SentimentAnalyzer
from .utils import get_logger

logger = get_logger(__name__)

# Simple in-memory cache (expires after 5 minutes)
_resource_cache: Dict[str, tuple[datetime, str]] = {}
CACHE_TTL = timedelta(minutes=5)


def _get_cached(key: str) -> Optional[str]:
    """Get cached resource if available and not expired."""
    if key in _resource_cache:
        timestamp, content = _resource_cache[key]
        if datetime.now() - timestamp < CACHE_TTL:
            return content
        else:
            del _resource_cache[key]
    return None


def _set_cached(key: str, content: str):
    """Cache resource content."""
    _resource_cache[key] = (datetime.now(), content)


async def get_historical_data_resource(ticker: str) -> str:
    """
    Resource: Historical stock data in markdown table format.
    URI: stock://historical/{ticker}
    """
    cache_key = f"historical:{ticker}"
    cached = _get_cached(cache_key)
    if cached:
        return cached
    
    try:
        df = DataLoader.fetch_history(ticker, period="1y", interval="1d")
        if df.empty:
            result = f"# Historical Data: {ticker}\n\n❌ No data available for {ticker}\n"
        else:
            result = f"# Historical Data: {ticker}\n\n"
            result += f"**Period**: Last 1 year\n"
            result += f"**Latest Price**: ${df['Close'].iloc[-1]:.2f}\n"
            result += f"**52-Week High**: ${df['High'].max():.2f}\n"
            result += f"**52-Week Low**: ${df['Low'].min():.2f}\n\n"
            
            # Show last 30 days
            recent_df = df.tail(30).reset_index()
            result += "## Recent Data (Last 30 Days)\n\n"
            result += recent_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].to_markdown(index=False)
        
        _set_cached(cache_key, result)
        return result
    except Exception as e:
        logger.error(f"Error fetching historical data resource: {e}")
        return f"# Error\n\nFailed to fetch data for {ticker}: {str(e)}"


async def get_ticker_info_resource(ticker: str) -> str:
    """
    Resource: Ticker information and metadata.
    URI: stock://info/{ticker}
    """
    cache_key = f"info:{ticker}"
    cached = _get_cached(cache_key)
    if cached:
        return cached
    
    try:
        info = DataLoader.get_ticker_info(ticker)
        if not info:
            result = f"# Ticker Info: {ticker}\n\n❌ No information available\n"
        else:
            result = f"# {info.get('longName', ticker)}\n\n"
            result += f"**Ticker**: {ticker}\n"
            result += f"**Sector**: {info.get('sector', 'N/A')}\n"
            result += f"**Industry**: {info.get('industry', 'N/A')}\n"
            result += f"**Market Cap**: ${info.get('marketCap', 0):,}\n"
            result += f"**PE Ratio**: {info.get('trailingPE', 'N/A')}\n"
            result += f"**Dividend Yield**: {info.get('dividendYield', 0) * 100:.2f}%\n"
            result += f"**52 Week Range**: ${info.get('fiftyTwoWeekLow', 0):.2f} - ${info.get('fiftyTwoWeekHigh', 0):.2f}\n\n"
            
            if 'longBusinessSummary' in info:
                result += f"## Business Summary\n\n{info['longBusinessSummary'][:500]}...\n"
        
        _set_cached(cache_key, result)
        return result
    except Exception as e:
        logger.error(f"Error fetching ticker info resource: {e}")
        return f"# Error\n\nFailed to fetch info for {ticker}: {str(e)}"


async def get_news_resource(ticker: str) -> str:
    """
    Resource: Recent news articles with sentiment.
    URI: stock://news/{ticker}
    """
    cache_key = f"news:{ticker}"
    cached = _get_cached(cache_key)
    if cached:
        return cached
    
    try:
        news = DataLoader.fetch_news(ticker)
        if not news:
            result = f"# News: {ticker}\n\n❌ No recent news articles found\n"
        else:
            result = f"# Recent News: {ticker}\n\n"
            result += f"**Articles**: {len(news)}\n\n"
            
            # Analyze sentiment
            analyzer = SentimentAnalyzer()
            total_sentiment = 0
            
            for i, article in enumerate(news[:10], 1):
                sentiment_score = analyzer.analyze(article['title'])
                total_sentiment += sentiment_score
                
                sentiment_emoji = "🟢" if sentiment_score > 0.05 else "🔴" if sentiment_score < -0.05 else "🟡"
                reliable_badge = "✓ Reliable" if article.get('is_reliable') else ""
                
                result += f"### {i}. {article['title']}\n"
                result += f"**Publisher**: {article['publisher']} {reliable_badge}\n"
                result += f"**Sentiment**: {sentiment_emoji} {sentiment_score:+.3f}\n"
                if article.get('link'):
                    result += f"**Link**: {article['link']}\n"
                result += "\n"
            
            avg_sentiment = total_sentiment / len(news[:10])
            overall_emoji = "🟢 Positive" if avg_sentiment > 0.05 else "🔴 Negative" if avg_sentiment < -0.05 else "🟡 Neutral"
            result += f"\n**Overall Sentiment**: {overall_emoji} ({avg_sentiment:+.3f})\n"
        
        _set_cached(cache_key, result)
        return result
    except Exception as e:
        logger.error(f"Error fetching news resource: {e}")
        return f"# Error\n\nFailed to fetch news for {ticker}: {str(e)}"


async def get_metrics_resource(ticker: str) -> str:
    """
    Resource: Quantitative analysis and risk metrics.
    URI: stock://metrics/{ticker}
    """
    cache_key = f"metrics:{ticker}"
    cached = _get_cached(cache_key)
    if cached:
        return cached
    
    try:
        df = DataLoader.fetch_history(ticker, period="1y")
        if df.empty:
            result = f"# Metrics: {ticker}\n\n❌ No data available\n"
        else:
            result = f"# Quantitative Metrics: {ticker}\n\n"
            
            analyzer = QuantAnalyzer()
            metrics = analyzer.calculate_risk_metrics(df)
            
            result += "## Risk Metrics\n\n"
            result += f"**Value at Risk (95%)**: {metrics.get('var_95', 0):.2%}\n"
            result += f"**CVaR/Expected Shortfall**: {metrics.get('cvar_95', 0):.2%}\n"
            result += f"**Annualized Volatility**: {metrics.get('volatility', 0):.2%}\n"
            result += f"**Sharpe Ratio**: {metrics.get('sharpe_ratio', 0):.2f}\n"
            result += f"**Kelly Criterion**: {metrics.get('kelly_criterion', 0):.2%}\n\n"
            
            # Market Regime
            adx_values = analyzer.calculate_adx(df)
            current_adx = adx_values.iloc[-1]
            regime = "📈 Trending" if current_adx > 25 else "📊 Ranging"
            
            result += "## Market Regime\n\n"
            result += f"**ADX**: {current_adx:.2f}\n"
            result += f"**Regime**: {regime}\n\n"
            
            # Patterns
            patterns = analyzer.detect_patterns(df)
            if patterns:
                result += "## Recent Patterns\n\n"
                for pattern, detected in patterns.items():
                    if detected:
                        result += f"- ✓ {pattern}\n"
        
        _set_cached(cache_key, result)
        return result
    except Exception as e:
        logger.error(f"Error fetching metrics resource: {e}")
        return f"# Error\n\nFailed to fetch metrics for {ticker}: {str(e)}"


async def get_predictions_resource(ticker: str, days: int = 30) -> str:
    """
    Resource: Cached prediction results (if available).
    URI: predictions://{ticker}/{days}
    
    Note: This is a placeholder. In production, you'd cache prediction results 
    from actual tool calls and serve them here.
    """
    return f"# Predictions: {ticker}\n\n⚠️ No cached predictions available. Use the `run_predictions` tool to generate forecasts.\n"


# Resource registry mapping URIs to handler functions
RESOURCE_HANDLERS = {
    "stock://historical/": get_historical_data_resource,
    "stock://info/": get_ticker_info_resource,
    "stock://news/": get_news_resource,
    "stock://metrics/": get_metrics_resource,
    "predictions://": get_predictions_resource,
}


async def handle_resource_request(uri: str) -> str:
    """
    Route resource requests to appropriate handlers.
    
    Args:
        uri: Resource URI (e.g., "stock://historical/AAPL")
    
    Returns:
        Resource content as markdown string
    """
    try:
        # Parse URI
        for prefix, handler in RESOURCE_HANDLERS.items():
            if uri.startswith(prefix):
                # Extract ticker/params from URI
                param = uri[len(prefix):]
                
                # Handle predictions:// with days parameter
                if prefix == "predictions://":
                    parts = param.split('/')
                    ticker = parts[0] if parts else ""
                    days = int(parts[1]) if len(parts) > 1 else 30
                    return await handler(ticker, days)
                else:
                    return await handler(param)
        
        return f"# Error\n\nUnknown resource URI: {uri}\n\nAvailable prefixes:\n" + "\n".join(f"- {p}" for p in RESOURCE_HANDLERS.keys())
    except Exception as e:
        logger.error(f"Error handling resource request: {e}")
        return f"# Error\n\nFailed to handle resource {uri}: {str(e)}"
