import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pyts.image import GramianAngularField
from torch.utils.data import DataLoader, TensorDataset
from .utils import get_logger

logger = get_logger(__name__)

class TimeSeriesCNN(nn.Module):
    def __init__(self, input_size):
        super(TimeSeriesCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        # Calculate flattened size
        # Input: (N, N) -> Pool -> (N/2, N/2) -> Pool -> (N/4, N/4)
        # We only do one pool after conv1, and maybe one after conv2?
        # Let's keep it simple:
        # 20x20 -> Conv(3x3) -> 20x20 -> Pool(2x2) -> 10x10
        # 10x10 -> Conv(3x3) -> 10x10 -> Pool(2x2) -> 5x5
        
        final_dim = input_size // 4
        self.fc1 = nn.Linear(32 * final_dim * final_dim, 64)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CVForecaster:
    def __init__(self, lookback=20):
        self.lookback = lookback
        self.gasf = GramianAngularField(image_size=lookback, method='summation')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def _prepare_data(self, data):
        # Create sliding windows
        X, y = [], []
        for i in range(len(data) - self.lookback):
            window = data[i:i+self.lookback]
            target = data[i+self.lookback]
            X.append(window)
            y.append(target)
            
        return np.array(X), np.array(y)
        
    def predict(self, df: pd.DataFrame, days: int = 30) -> pd.Series:
        try:
            # Normalize data
            values = df['Close'].values
            min_val = np.min(values)
            max_val = np.max(values)
            scaled_values = (values - min_val) / (max_val - min_val + 1e-8)
            
            X, y = self._prepare_data(scaled_values)
            
            if len(X) < 50: # Not enough data
                logger.warning("Not enough data for CNN-GAF")
                return pd.Series([values[-1]] * days, name="CNN-GAF")
            
            # Transform to GAF images
            # X shape: (samples, lookback)
            X_gasf = self.gasf.fit_transform(X)
            # X_gasf shape: (samples, lookback, lookback)
            
            # Convert to PyTorch tensors
            X_tensor = torch.FloatTensor(X_gasf).unsqueeze(1).to(self.device) # (B, 1, H, W)
            y_tensor = torch.FloatTensor(y).unsqueeze(1).to(self.device)
            
            # Train model
            model = TimeSeriesCNN(self.lookback).to(self.device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            dataset = TensorDataset(X_tensor, y_tensor)
            loader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            model.train()
            epochs = 50 # Fast training
            for epoch in range(epochs):
                for batch_X, batch_y in loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
            # Predict future
            model.eval()
            preds = []
            current_window = scaled_values[-self.lookback:]
            
            with torch.no_grad():
                for _ in range(days):
                    # Transform current window
                    # Reshape to (1, lookback) for GAF
                    curr_gasf = self.gasf.transform(current_window.reshape(1, -1))
                    curr_tensor = torch.FloatTensor(curr_gasf).unsqueeze(1).to(self.device)
                    
                    pred_scaled = model(curr_tensor).item()
                    
                    # Inverse transform
                    pred_actual = pred_scaled * (max_val - min_val) + min_val
                    preds.append(pred_actual)
                    
                    # Update window
                    current_window = np.append(current_window[1:], pred_scaled)
            
            return pd.Series(preds, name="CNN-GAF")
            
        except Exception as e:
            logger.error(f"CNN-GAF failed: {e}")
            return pd.Series([df['Close'].iloc[-1]] * days, name="CNN-GAF")
