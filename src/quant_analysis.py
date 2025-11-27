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

    @staticmethod
    def calculate_risk_metrics(df: pd.DataFrame, confidence_level: float = 0.95) -> dict:
        """
        Calculates advanced risk metrics:
        - Value at Risk (VaR)
        - Conditional VaR (CVaR) / Expected Shortfall
        - Kelly Criterion
        """
        if df.empty:
            return {}
            
        returns = df['Close'].pct_change().dropna()
        if returns.empty:
            return {}
            
        # 1. Value at Risk (VaR) - Historical Method
        # The loss threshold at the given confidence level
        var = np.percentile(returns, 100 * (1 - confidence_level))
        
        # 2. Conditional VaR (CVaR)
        # Average loss of days exceeding VaR
        cvar = returns[returns <= var].mean()
        
        # 3. Kelly Criterion
        # f = p - q/b (Simplified: mean/variance for continuous approximation)
        # Fraction of bankroll to bet
        mean_return = returns.mean()
        var_return = returns.var()
        kelly = mean_return / var_return if var_return > 0 else 0
        
        # Cap Kelly at reasonable levels (e.g., 20% max leverage for safety in display)
        kelly_display = min(max(kelly, 0), 2.0) 
        
        return {
            "VaR (95%)": f"{var:.2%}",
            "CVaR (95%)": f"{cvar:.2%}",
            "Kelly Criterion": f"{kelly_display:.2f}x"
        }

    @staticmethod
    def detect_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Detects candlestick patterns: Doji, Hammer, Engulfing.
        Adds boolean columns to the dataframe.
        """
        df = df.copy()
        
        op = df['Open']
        hi = df['High']
        lo = df['Low']
        cl = df['Close']
        
        # Candle parts
        body = np.abs(cl - op)
        upper_wick = hi - np.maximum(cl, op)
        lower_wick = np.minimum(cl, op) - lo
        candle_range = hi - lo
        
        # 1. Doji: Very small body relative to range
        df['Pattern_Doji'] = body <= (candle_range * 0.1)
        
        # 2. Hammer: Small body, long lower wick, short upper wick
        # Occurs at bottom of trend (simplified here to just shape)
        df['Pattern_Hammer'] = (
            (lower_wick >= 2 * body) & 
            (upper_wick <= 0.1 * body) &
            (body > 0)
        )
        
        # 3. Bullish Engulfing
        # Previous candle red, current candle green and engulfs previous body
        prev_op = op.shift(1)
        prev_cl = cl.shift(1)
        prev_body = np.abs(prev_cl - prev_op)
        
        is_green = cl > op
        is_prev_red = prev_cl < prev_op
        
        df['Pattern_Engulfing'] = (
            is_green & is_prev_red &
            (op < prev_cl) & (cl > prev_op) # Strictly engulfing body
        )
        
        return df

    @staticmethod
    def calculate_correlation(main_ticker: str, period: str = "1y") -> pd.DataFrame:
        """
        Fetches a basket of assets and calculates correlation matrix.
        """
        import yfinance as yf
        
        basket = [main_ticker, "SPY", "QQQ", "IWM", "GLD", "BTC-USD", "TLT"]
        
        try:
            data = yf.download(basket, period=period, progress=False)['Close']
            # If multi-index columns (yfinance update), flatten
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
                
            # Calculate correlation
            corr_matrix = data.corr()
            return corr_matrix
        except Exception as e:
            print(f"Correlation error: {e}")
            return pd.DataFrame()

    @staticmethod
    def add_advanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds Ichimoku, Stochastic, and ATR.
        """
        df = df.copy()
        
        # 1. Ichimoku Cloud
        # Conversion Line (Tenkan-sen): (9-period high + 9-period low)/2
        nine_high = df['High'].rolling(window=9).max()
        nine_low = df['Low'].rolling(window=9).min()
        df['Ichimoku_Conversion'] = (nine_high + nine_low) / 2
        
        # Base Line (Kijun-sen): (26-period high + 26-period low)/2
        twenty_six_high = df['High'].rolling(window=26).max()
        twenty_six_low = df['Low'].rolling(window=26).min()
        df['Ichimoku_Base'] = (twenty_six_high + twenty_six_low) / 2
        
        # Leading Span A (Senkou Span A): (Conversion + Base)/2
        df['Ichimoku_SpanA'] = ((df['Ichimoku_Conversion'] + df['Ichimoku_Base']) / 2).shift(26)
        
        # Leading Span B (Senkou Span B): (52-period high + 52-period low)/2
        fifty_two_high = df['High'].rolling(window=52).max()
        fifty_two_low = df['Low'].rolling(window=52).min()
        df['Ichimoku_SpanB'] = ((fifty_two_high + fifty_two_low) / 2).shift(26)
        
        # 2. Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(
            high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3
        )
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        
        # 3. ATR (Average True Range)
        atr = ta.volatility.AverageTrueRange(
            high=df['High'], low=df['Low'], close=df['Close'], window=14
        )
        df['ATR'] = atr.average_true_range()
        
        return df

    @staticmethod
    def calculate_market_regime(df: pd.DataFrame) -> dict:
        """
        Determines the market regime using ADX (Average Directional Index).
        ADX > 25 indicates a strong trend.
        ADX < 20 indicates a weak trend (ranging).
        """
        if df.empty:
            return {"ADX": 0, "Regime": "Unknown"}
            
        try:
            adx_ind = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14)
            adx = adx_ind.adx().iloc[-1]
            
            if adx >= 25:
                regime = "Strong Trend 🚀"
            elif adx < 20:
                regime = "Sideways / Ranging 🦀"
            else:
                regime = "Weak Trend ➡️"
                
            return {"ADX": adx, "Regime": regime}
        except Exception as e:
            return {"ADX": 0, "Regime": "Error"}
