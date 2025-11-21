import yfinance as yf
import pandas as pd
from typing import Optional, Tuple, List, Dict
from .utils import get_logger

logger = get_logger(__name__)

class DataLoader:
    """
    Handles fetching of financial data and news using yfinance.
    """
    
    @staticmethod
    def fetch_history(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        """
        Fetches historical OHLCV data.
        """
        try:
            logger.info(f"Fetching data for {ticker} (Period: {period}, Interval: {interval})")
            # yfinance 0.2.66+ handles sessions/cookies automatically
            t = yf.Ticker(ticker)
            df = t.history(period=period, interval=interval)
            
            if df.empty:
                logger.warning(f"No data found for {ticker}")
                return pd.DataFrame()
                
            return df
        except Exception as e:
            logger.error(f"Error fetching history for {ticker}: {e}")
            return pd.DataFrame()

    @staticmethod
    def fetch_news(ticker: str) -> List[Dict]:
        """
        Fetches recent news for the ticker and normalizes the output.
        """
        try:
            logger.info(f"Fetching news for {ticker}")
            t = yf.Ticker(ticker)
            raw_news = t.news
            
            normalized_news = []
            for item in raw_news:
                # Handle different yfinance news structures
                content = item.get('content', {})
                
                # Extract Title
                title = item.get('title') or content.get('title', 'No Title')
                
                # Extract Link
                link = item.get('link')
                if not link:
                    canonical = content.get('canonicalUrl')
                    if isinstance(canonical, dict):
                        link = canonical.get('url')
                    else:
                        link = canonical
                
                normalized_news.append({
                    'title': title,
                    'link': link or '#',
                    'publisher': content.get('provider', {}).get('displayName', 'Unknown'),
                    'published': content.get('pubDate')
                })
                
            return normalized_news
        except Exception as e:
            logger.error(f"Error fetching news for {ticker}: {e}")
            return []

    @staticmethod
    def get_ticker_info(ticker: str) -> Dict:
        """
        Fetches basic info about the ticker.
        """
        try:
            t = yf.Ticker(ticker)
            return t.info
        except Exception as e:
            logger.error(f"Error fetching info for {ticker}: {e}")
            return {}
