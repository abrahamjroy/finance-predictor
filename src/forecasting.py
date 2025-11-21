import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
import ta
from typing import Dict, List, Tuple
from .utils import get_logger

logger = get_logger(__name__)

class ForecastEngine:
    """
    Implements 10+ forecasting algorithms for financial time series.
    """
    
    def __init__(self):
        self.models = {
            "SMA (20)": self.predict_sma,
            "EMA (20)": self.predict_ema,
            "Linear Regression": self.predict_linear_regression,
            "Random Forest": self.predict_rf,
            "XGBoost": self.predict_xgboost,
            "SVR": self.predict_svr,
            "Holt-Winters": self.predict_holt_winters,
            "ARIMA": self.predict_arima,
            "RSI Trend": self.predict_rsi_trend,
            "Bollinger Trend": self.predict_bollinger_trend
        }

    def _prepare_ml_features(self, df: pd.DataFrame, lookback: int = 5) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
        """
        Creates lag features for ML models.
        """
        data = df['Close'].values.reshape(-1, 1)
        X, y = [], []
        for i in range(len(data) - lookback):
            X.append(data[i:i+lookback].flatten())
            y.append(data[i+lookback])
        
        X = np.array(X)
        y = np.array(y)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, y, scaler

    def predict_sma(self, df: pd.DataFrame, days: int = 30) -> pd.Series:
        """Simple Moving Average projection."""
        last_val = df['Close'].iloc[-1]
        sma = df['Close'].rolling(window=20).mean().iloc[-1]
        # Simple projection: trend towards SMA or continue slope? 
        # For simplicity in 'prediction', we project the last SMA slope
        slope = (sma - df['Close'].rolling(window=20).mean().iloc[-5]) / 5
        preds = [sma + slope * i for i in range(1, days + 1)]
        return pd.Series(preds, name="SMA")

    def predict_ema(self, df: pd.DataFrame, days: int = 30) -> pd.Series:
        """Exponential Moving Average projection."""
        ema = ta.trend.EMAIndicator(close=df['Close'], window=20).ema_indicator().iloc[-1]
        # Project flat for simplicity or use recent momentum
        return pd.Series([ema] * days, name="EMA")

    def predict_linear_regression(self, df: pd.DataFrame, days: int = 30) -> pd.Series:
        """Linear Regression on time index."""
        y = df['Close'].values
        X = np.arange(len(y)).reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, y)
        
        future_X = np.arange(len(y), len(y) + days).reshape(-1, 1)
        preds = model.predict(future_X)
        return pd.Series(preds, name="Linear Reg")

    def predict_rf(self, df: pd.DataFrame, days: int = 30) -> pd.Series:
        """Random Forest Regressor."""
        lookback = 5
        X, y, scaler = self._prepare_ml_features(df, lookback)
        model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
        model.fit(X, y.ravel())
        
        last_window = df['Close'].values[-lookback:].reshape(1, -1)
        preds = []
        curr_window = scaler.transform(last_window)
        
        for _ in range(days):
            pred = model.predict(curr_window)[0]
            preds.append(pred)
            # Update window: shift left, add new pred
            new_row = np.append(last_window[0][1:], pred).reshape(1, -1)
            last_window = new_row
            curr_window = scaler.transform(new_row)
            
        return pd.Series(preds, name="Random Forest")

    def predict_xgboost(self, df: pd.DataFrame, days: int = 30) -> pd.Series:
        """XGBoost Regressor."""
        lookback = 5
        X, y, scaler = self._prepare_ml_features(df, lookback)
        
        try:
            # Try GPU first (User has RTX 4070)
            model = xgb.XGBRegressor(n_estimators=100, device="cuda", tree_method="hist")
            model.fit(X, y)
        except Exception as e:
            logger.warning(f"XGBoost GPU failed, falling back to CPU: {e}")
            model = xgb.XGBRegressor(n_estimators=100, n_jobs=-1)
            model.fit(X, y)
        
        last_window = df['Close'].values[-lookback:].reshape(1, -1)
        preds = []
        curr_window = scaler.transform(last_window)
        
        for _ in range(days):
            pred = model.predict(curr_window)[0]
            preds.append(pred)
            new_row = np.append(last_window[0][1:], pred).reshape(1, -1)
            last_window = new_row
            curr_window = scaler.transform(new_row)
            
        return pd.Series(preds, name="XGBoost")

    def predict_svr(self, df: pd.DataFrame, days: int = 30) -> pd.Series:
        """Support Vector Regression."""
        lookback = 5
        X, y, scaler = self._prepare_ml_features(df, lookback)
        model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
        model.fit(X, y.ravel())
        
        last_window = df['Close'].values[-lookback:].reshape(1, -1)
        preds = []
        curr_window = scaler.transform(last_window)
        
        for _ in range(days):
            pred = model.predict(curr_window)[0]
            preds.append(pred)
            new_row = np.append(last_window[0][1:], pred).reshape(1, -1)
            last_window = new_row
            curr_window = scaler.transform(new_row)
            
        return pd.Series(preds, name="SVR")

    def predict_holt_winters(self, df: pd.DataFrame, days: int = 30) -> pd.Series:
        """Holt-Winters Exponential Smoothing."""
        try:
            model = ExponentialSmoothing(df['Close'], trend='add', seasonal=None).fit()
            return model.forecast(days)
        except:
            return pd.Series([df['Close'].iloc[-1]] * days, name="HW")

    def predict_arima(self, df: pd.DataFrame, days: int = 30) -> pd.Series:
        """ARIMA Model."""
        try:
            # Simple ARIMA(1,1,1) for speed
            model = ARIMA(df['Close'], order=(1, 1, 1)).fit()
            return model.forecast(days)
        except:
             return pd.Series([df['Close'].iloc[-1]] * days, name="ARIMA")

    def predict_rsi_trend(self, df: pd.DataFrame, days: int = 30) -> pd.Series:
        """Heuristic based on RSI overbought/oversold."""
        rsi = ta.momentum.RSIIndicator(df['Close']).rsi().iloc[-1]
        last_price = df['Close'].iloc[-1]
        
        # If oversold (<30), predict up. If overbought (>70), predict down.
        if rsi < 30:
            direction = 1.01 # 1% up daily
        elif rsi > 70:
            direction = 0.99 # 1% down daily
        else:
            direction = 1.0005 # Slight drift up
            
        preds = [last_price * (direction ** i) for i in range(1, days + 1)]
        return pd.Series(preds, name="RSI Trend")

    def predict_bollinger_trend(self, df: pd.DataFrame, days: int = 30) -> pd.Series:
        """Heuristic based on Bollinger Bands."""
        indicator = ta.volatility.BollingerBands(df['Close'])
        bb_hi = indicator.bollinger_hband().iloc[-1]
        bb_lo = indicator.bollinger_lband().iloc[-1]
        last_price = df['Close'].iloc[-1]
        
        # Mean reversion logic
        if last_price > bb_hi:
            target = bb_hi # Revert to band
            step = (target - last_price) / days
        elif last_price < bb_lo:
            target = bb_lo
            step = (target - last_price) / days
        else:
            step = 0
            
        preds = [last_price + (step * i) for i in range(1, days + 1)]
        return pd.Series(preds, name="Bollinger")

    def get_all_predictions(self, df: pd.DataFrame, days: int = 30) -> pd.DataFrame:
        """Runs all models and returns a DataFrame of predictions."""
        results = {}
        for name, func in self.models.items():
            try:
                results[name] = func(df, days).values
            except Exception as e:
                logger.error(f"Model {name} failed: {e}")
                
        return pd.DataFrame(results)
