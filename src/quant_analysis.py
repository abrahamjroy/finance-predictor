import pandas as pd
import numpy as np
from scipy import stats
import ta

class QuantAnalyzer:
    """
    Provides quantitative analysis metrics and technical indicators.
    """
    
    @staticmethod
    def calculate_metrics(df: pd.DataFrame, risk_free_rate: float = 0.02) -> dict:
        """
        Calculates key quantitative metrics:
        - Sharpe Ratio
        - Sortino Ratio
        - Max Drawdown
        - Annualized Volatility
        - Skewness & Kurtosis
        """
        if df.empty:
            return {}
            
        # Calculate daily returns
        returns = df['Close'].pct_change().dropna()
        
        if returns.empty:
            return {}
            
        # Annualization factor (crypto=365, stocks=252)
        # We'll assume 252 for standard finance
        ann_factor = 252
        
        # 1. Annualized Volatility
        volatility = returns.std() * np.sqrt(ann_factor)
        
        # 2. Sharpe Ratio
        # (Mean Return - Risk Free) / Std Dev
        excess_returns = returns - (risk_free_rate / ann_factor)
        sharpe = np.sqrt(ann_factor) * (excess_returns.mean() / returns.std())
        
        # 3. Sortino Ratio
        # (Mean Return - Risk Free) / Downside Deviation
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        sortino = np.sqrt(ann_factor) * (excess_returns.mean() / downside_std) if downside_std != 0 else 0
        
        # 4. Max Drawdown
        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min()
        
        # 5. Statistical Moments
        skew = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        return {
            "Annual Volatility": f"{volatility:.2%}",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Sortino Ratio": f"{sortino:.2f}",
            "Max Drawdown": f"{max_drawdown:.2%}",
            "Skewness": f"{skew:.2f}",
            "Kurtosis": f"{kurtosis:.2f}"
        }

    @staticmethod
    def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds technical indicators to the dataframe.
        """
        df = df.copy()
        
        # Trend
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)
        
        # Momentum
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        df['MACD'] = ta.trend.macd_diff(df['Close'])
        
        # Volatility
        bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
        df['BB_High'] = bb.bollinger_hband()
        df['BB_Low'] = bb.bollinger_lband()
        
        return df
