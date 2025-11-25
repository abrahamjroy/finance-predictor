import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import pandas as pd
import numpy as np
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QComboBox, 
                             QSlider, QPushButton, QFrame, QMessageBox, QScrollArea, QTextEdit)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QPainter, QColor, QFont, QPalette

import pyqtgraph as pg

from src.data_loader import DataLoader
from src.forecasting import ForecastEngine
from src.sentiment import SentimentEngine
from src.llm_engine import LLMEngine
from src.utils import setup_dirs

# Configure PyQtGraph
pg.setConfigOptions(antialias=True)
pg.setConfigOption('background', '#0e1117')
pg.setConfigOption('foreground', 'w')

class TickerTape(QWidget):
    def __init__(self, parent=None, speed=2, background_color="#000000", text_color="#00FF00"):
        super().__init__(parent)
        self.setFixedHeight(40)
        self.items = []
        self.speed = speed
        self.offset = 0
        self.background_color = QColor(background_color)
        self.text_color = QColor(text_color)
        self.font = QFont("Consolas", 12, QFont.Weight.Bold)
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_position)
        self.timer.start(20)
        
    def set_items(self, items):
        self.items = items
        self.offset = 0
        self.update()
        
    def update_position(self):
        if not self.items:
            return
        self.offset -= self.speed
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.background_color)
        painter.setFont(self.font)
        
        if not self.items:
            return
            
        x = self.offset
        
        for item in self.items:
            if isinstance(item, tuple):
                text, color_str = item
                painter.setPen(QColor(color_str))
            else:
                text = item
                painter.setPen(self.text_color)
                
            text_width = painter.fontMetrics().horizontalAdvance(text)
            
            if x + text_width > 0 and x < self.width():
                painter.drawText(x, 25, text)
            
            x += text_width + 40
            
        if x < 0:
            self.offset = self.width()

class DataLoaderThread(QThread):
    data_loaded = pyqtSignal(object, object)
    error_occurred = pyqtSignal(str)

    def __init__(self, ticker, period):
        super().__init__()
        self.ticker = ticker
        self.period = period

    def run(self):
        try:
            df = DataLoader.fetch_history(self.ticker, self.period)
            news = DataLoader.fetch_news(self.ticker)
            self.data_loaded.emit(df, news)
        except Exception as e:
            self.error_occurred.emit(str(e))

class AIThread(QThread):
    analysis_ready = pyqtSignal(str)
    
    def __init__(self, engine, prompt):
        super().__init__()
        self.engine = engine
        self.prompt = prompt
        
    def run(self):
        try:
            if self.engine:
                result = self.engine.analyze(self.prompt)
                # Handle dict response
                if isinstance(result, dict):
                    if "error" in result:
                        self.analysis_ready.emit(f"Error: {result['error']}")
                    else:
                        response = result.get("response", "No response")
                        # Optionally show thinking
                        thinking = result.get("thinking", "")
                        if thinking:
                            full_response = f"[Reasoning: {thinking[:100]}...]\n\n{response}"
                            self.analysis_ready.emit(full_response)
                        else:
                            self.analysis_ready.emit(response)
                else:
                    self.analysis_ready.emit(str(result))
        except Exception as e:
            self.analysis_ready.emit(f"Error: {e}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Finance Predictor Pro (PyQtGraph)")
        self.resize(1400, 900)
        
        # State
        self.ticker = "AAPL"
        self.df = pd.DataFrame()
        self.news = []
        self.llm_engine = None
        self.forecaster = ForecastEngine()
        self.sentiment_analyzer = SentimentEngine()
        self.predictions = {}
        
        # Setup UI
        self.setup_ui()
        self.setup_styling()
        
        # Initialize Engines
        setup_dirs()
        threading.Thread(target=self.load_llm, daemon=True).start()
        
        # Initial Data Load
        self.update_ticker_tape_data()
        
        # Preload data
        print("Preloading initial data for AAPL...")
        initial_df = DataLoader.fetch_history("AAPL", "2y")
        initial_news = DataLoader.fetch_news("AAPL")
        print("Data preloaded successfully!")
        
        self.df = initial_df
        self.news = initial_news
        self.ticker = "AAPL"
        
        self.update_chart()
        self.update_sentiment()
        self.update_news_ticker()

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Top Ticker
        self.top_ticker = TickerTape(background_color="#000000")
        layout.addWidget(self.top_ticker)
        
        # Main Content
        content_layout = QHBoxLayout()
        layout.addLayout(content_layout, stretch=1)
        
        # Chart Area (PyQtGraph)
        chart_widget = QWidget()
        chart_layout = QVBoxLayout(chart_widget)
        
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#0e1117')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel('left', 'Price ($)', color='white', size='12pt')
        self.plot_widget.setLabel('bottom', 'Date', color='white', size='12pt')
        self.plot_widget.addLegend()
        
        chart_layout.addWidget(self.plot_widget)
        content_layout.addWidget(chart_widget, stretch=2)
        
        # Sidebar
        sidebar = QFrame()
        sidebar.setFixedWidth(350)
        sidebar.setStyleSheet("background-color: #161b22; color: white;")
        sidebar_layout = QVBoxLayout(sidebar)
        
        sidebar_layout.addWidget(QLabel("<b>Configuration</b>"))
        
        self.ticker_input = QLineEdit("AAPL")
        self.ticker_input.setStyleSheet("background-color: #333; color: white; padding: 5px;")
        self.ticker_input.returnPressed.connect(self.load_data)
        sidebar_layout.addWidget(QLabel("Ticker:"))
        sidebar_layout.addWidget(self.ticker_input)
        
        self.period_combo = QComboBox()
        self.period_combo.addItems(["1y", "2y", "5y", "max"])
        self.period_combo.setCurrentText("2y")
        self.period_combo.setStyleSheet("background-color: #333; color: white;")
        sidebar_layout.addWidget(QLabel("Period:"))
        sidebar_layout.addWidget(self.period_combo)
        
        self.days_slider = QSlider(Qt.Orientation.Horizontal)
        self.days_slider.setRange(7, 90)
        self.days_slider.setValue(30)
        self.days_label = QLabel("Forecast Days: 30")
        self.days_slider.valueChanged.connect(lambda v: self.days_label.setText(f"Forecast Days: {v}"))
        sidebar_layout.addWidget(self.days_label)
        sidebar_layout.addWidget(self.days_slider)
        
        self.run_btn = QPushButton("Run Predictions")
        self.run_btn.setStyleSheet("background-color: #1f6feb; color: white; padding: 8px; font-weight: bold;")
        self.run_btn.clicked.connect(self.run_predictions)
        sidebar_layout.addWidget(self.run_btn)
        
        sidebar_layout.addSpacing(20)
        
        self.sentiment_label = QLabel("Sentiment: N/A")
        self.sentiment_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        sidebar_layout.addWidget(self.sentiment_label)
        
        sidebar_layout.addSpacing(20)
        sidebar_layout.addWidget(QLabel("<b>AI Chat Assistant</b>"))
        
        # Chat history display
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setStyleSheet("background-color: #0d1117; color: white; padding: 10px;")
        self.chat_history.setText("💬 AI Assistant ready. Ask me anything about the stock!\n")
        sidebar_layout.addWidget(self.chat_history)
        
        # Chat input
        chat_input_layout = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Ask a question...")
        self.chat_input.setStyleSheet("background-color: #333; color: white; padding: 8px;")
        self.chat_input.returnPressed.connect(self.send_chat_message)
        
        self.send_btn = QPushButton("Send")
        self.send_btn.setStyleSheet("background-color: #238636; color: white; padding: 8px;")
        self.send_btn.clicked.connect(self.send_chat_message)
        
        chat_input_layout.addWidget(self.chat_input)
        chat_input_layout.addWidget(self.send_btn)
        sidebar_layout.addLayout(chat_input_layout)
        
        # Quick action button
        self.quick_analysis_btn = QPushButton("📊 Quick Analysis")
        self.quick_analysis_btn.setStyleSheet("background-color: #1f6feb; color: white; padding: 6px;")
        self.quick_analysis_btn.clicked.connect(lambda: self.send_chat_message("Analyze this stock"))
        sidebar_layout.addWidget(self.quick_analysis_btn)
        
        content_layout.addWidget(sidebar)
        
        # Bottom Ticker
        self.bottom_ticker = TickerTape(background_color="#000000", text_color="#00FFFF")
        layout.addWidget(self.bottom_ticker)
        
    def setup_styling(self):
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor("#0e1117"))
        palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        self.setPalette(palette)

    def load_llm(self):
        self.llm_engine = LLMEngine()
        print("LLM Loaded")

    def update_ticker_tape_data(self):
        # Comprehensive list of 250+ global stocks
        tickers = [
            # US Tech
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "INTC", "ORCL",
            # US Finance
            "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "AXP", "V", "MA",
            # US Healthcare
            "JNJ", "UNH", "PFE", "ABBV", "TMO", "MRK", "ABT", "DHR", "BMY", "LLY",
            # US Consumer
            "WMT", "HD", "MCD", "NKE", "SBUX", "TGT", "COST", "LOW", "DIS", "NFLX",
            # Indian Stocks (NSE)
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "HINDUNILVR.NS",
            "BHARTIARTL.NS", "ITC.NS", "SBIN.NS", "BAJFINANCE.NS", "KOTAKBANK.NS", "LT.NS",
            "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "TITAN.NS", "WIPRO.NS", "TECHM.NS",
            "ULTRACEMCO.NS", "NESTLEIND.NS", "HCLTECH.NS", "POWERGRID.NS", "NTPC.NS",
            # Commodities & Energy
            "GC=F", "SI=F", "CL=F", "NG=F", "HG=F",  # Gold, Silver, Oil, Gas, Copper
            "XOM", "CVX", "COP", "SLB", "EOG",  # Energy companies
            # Agriculture & Food
            "ADM", "BG", "TSN", "CAG", "GIS", "K", "MDLZ", "KHC", "HSY", "CPB",
            # Dairy & Meat
            "DANOY", "FDP", "CALM", "SAFM",
            # Crypto
            "BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD", "SOL-USD", "XRP-USD",
            # European
            "SAP", "ASML", "NVO", "TM", "NSRGY", "RHHBY", "SNY", "BP", "SHEL",
            # Asian
            "BABA", "TSM", "SONY", "9988.HK", "0700.HK", "005930.KS",
            # Indices
            "^GSPC", "^DJI", "^IXIC", "^NSEI", "^BSESN", "^FTSE", "^N225",
            # Materials
            "LIN", "APD", "SHW", "ECL", "DD", "NEM", "FCX",
            # Industrial
            "BA", "HON", "UPS", "CAT", "DE", "MMM", "GE",
            # Real Estate
            "AMT", "PLD", "CCI", "EQIX", "PSA",
            # Utilities
            "NEE", "DUK", "SO", "D", "AEP",
            # More Indian Agriculture/FMCG
            "DABUR.NS", "BRITANNIA.NS", "GODREJCP.NS", "MARICO.NS", "COLPAL.NS",
            "TATACONSUM.NS", "VBL.NS", "MCDOWELL-N.NS", "PGHH.NS",
            # Fertilizers & Agri
            "COROMANDEL.NS", "UPL.NS", "PIIND.NS", "CHAMBLFERT.NS",
            # More Commodities
            "VEDL.NS", "HINDALCO.NS", "TATASTEEL.NS", "JSWSTEEL.NS", "SAIL.NS",
        ]
        
        # Create ticker items with mock prices (in real app, fetch live data)
        items = []
        for i, ticker in enumerate(tickers[:250]):  # Limit to 250
            price = 100 + (i * 10) % 500
            change = ((i * 7) % 20 - 10) / 10
            color = "#00FF00" if change > 0 else "#FF0000"
            symbol = "▲" if change > 0 else "▼"
            items.append((f"{ticker}: {price:.2f} {symbol} {abs(change):.1f}%", color))
        
        self.top_ticker.set_items(items * 2)
        
    def load_data(self):
        ticker = self.ticker_input.text()
        period = self.period_combo.currentText()
        
        # Show loading message
        self.sentiment_label.setText("Loading...")
        self.sentiment_label.setStyleSheet("font-size: 16px; font-weight: bold; color: yellow;")
        QApplication.processEvents()  # Force UI update
        
        try:
            # Load synchronously in main thread - NO THREADING
            self.df = DataLoader.fetch_history(ticker, period)
            self.news = DataLoader.fetch_news(ticker)
            self.ticker = ticker
            
            self.update_chart()
            self.update_sentiment()
            self.update_news_ticker()
            
            QMessageBox.information(self, "Success", f"Loaded {len(self.df)} days of data for {ticker}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}")
            self.sentiment_label.setText("Sentiment: N/A")
            self.sentiment_label.setStyleSheet("font-size: 16px; font-weight: bold; color: orange;")
        
    def update_sentiment(self):
        try:
            score = self.sentiment_analyzer.analyze_news(self.news)
            color = "green" if score > 0.05 else ("red" if score < -0.05 else "orange")
            text = "Bullish" if score > 0.05 else ("Bearish" if score < -0.05 else "Neutral")
            self.sentiment_label.setText(f"Sentiment: {text} ({score:.2f})")
            self.sentiment_label.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {color};")
        except Exception as e:
            print(f"Sentiment update error: {e}")
    
    def update_news_ticker(self):
        if self.news:
            news_items = [f"{n['publisher']}: {n['title']}" for n in self.news]
            self.bottom_ticker.set_items(news_items * 5)
        else:
            self.bottom_ticker.set_items(["No recent news found."])

    def update_chart(self):
        try:
            self.plot_widget.clear()
            
            if self.df.empty:
                return
            
            # Convert datetime index to timestamps
            timestamps = [d.timestamp() for d in self.df.index]
            
            # Plot historical data
            self.plot_widget.plot(timestamps, self.df['Close'].values, 
                                pen=pg.mkPen(color='w', width=2), name='Historical Price')
            
            # Plot predictions
            if self.predictions:
                future_dates = pd.date_range(start=self.df.index[-1] + pd.Timedelta(days=1), 
                                            periods=self.days_slider.value())
                future_timestamps = [d.timestamp() for d in future_dates]
                
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD']
                
                for i, (name, values) in enumerate(self.predictions.items()):
                    color = colors[i % len(colors)]
                    self.plot_widget.plot(future_timestamps, values,
                                        pen=pg.mkPen(color=color, width=2, style=Qt.PenStyle.DashLine),
                                        name=name)
                
                # Ensemble average
                avg_pred = np.mean(list(self.predictions.values()), axis=0)
                self.plot_widget.plot(future_timestamps, avg_pred,
                                    pen=pg.mkPen(color='#FFD700', width=3),
                                    name='ENSEMBLE AVERAGE')
            
            # Set title
            self.plot_widget.setTitle(f"{self.ticker} Price Analysis", color='w', size='14pt')
            
            # Format x-axis to show dates
            axis = self.plot_widget.getAxis('bottom')
            axis.setTicks([[(ts, datetime.fromtimestamp(ts).strftime('%Y-%m-%d')) 
                           for ts in timestamps[::len(timestamps)//10]]])
            
        except Exception as e:
            print(f"Chart update error: {e}")
            import traceback
            traceback.print_exc()

    def run_predictions(self):
        if self.df.empty:
            QMessageBox.warning(self, "No Data", "Please load ticker data first.")
            return
            
        days = self.days_slider.value()
        preds = {}
        
        # Show loading
        self.run_btn.setText("Running...")
        self.run_btn.setEnabled(False)
        QApplication.processEvents()
        
        try:
            # Parallel execution for speed
            def run_model(name, model_func):
                try:
                    pred_series = model_func(self.df, days)
                    values = pred_series.values if hasattr(pred_series, 'values') else pred_series
                    
                    def flatten(arr):
                        result = []
                        for item in np.atleast_1d(arr):
                            if isinstance(item, (list, np.ndarray)):
                                result.extend(flatten(item))
                            else:
                                result.append(float(item))
                        return result
                    
                    values = np.array(flatten(values))
                    if len(values) == days:
                        return (name, values)
                except Exception as e:
                    print(f"Model {name} failed: {e}")
                return None
            
            # Execute models in parallel
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(run_model, name, func): name 
                          for name, func in self.forecaster.models.items()}
                
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        name, values = result
                        preds[name] = values
                    
            if preds:
                self.predictions = preds
                self.update_chart()
                QMessageBox.information(self, "Success", 
                    f"Generated predictions using {len(preds)} models in parallel!")
            else:
                QMessageBox.warning(self, "No Predictions", "All prediction models failed.")
        except Exception as e:
            QMessageBox.critical(self, "Prediction Error", f"An error occurred: {str(e)}")
        finally:
            self.run_btn.setText("Run Predictions")
            self.run_btn.setEnabled(True)

    def send_chat_message(self, preset_message=None):
        if not self.llm_engine:
            self.append_chat("🤖 AI", "AI Engine is still loading, please wait...")
            return
        
        message = preset_message if preset_message else self.chat_input.text().strip()
        if not message:
            return
        
        # Clear input
        self.chat_input.clear()
        
        # Show user message
        self.append_chat("👤 You", message)
        
        # Show thinking indicator
        self.append_chat("🤖 AI", "Thinking...")
        
        # Build context-aware prompt
        context = f"Stock: {self.ticker}\n"
        if not self.df.empty:
            context += f"Current Price: ${self.df['Close'].iloc[-1]:.2f}\n"
            context += f"52-week High: ${self.df['Close'].max():.2f}\n"
            context += f"52-week Low: ${self.df['Close'].min():.2f}\n"
        
        prompt = f"{context}\nUser Question: {message}\n\nProvide a concise, helpful response."
        
        self.ai_thread = AIThread(self.llm_engine, prompt)
        self.ai_thread.analysis_ready.connect(self.display_ai_response)
        self.ai_thread.start()
    
    def append_chat(self, sender, message):
        current = self.chat_history.toPlainText()
        self.chat_history.setText(f"{current}\n{sender}: {message}\n")
        # Scroll to bottom
        self.chat_history.verticalScrollBar().setValue(
            self.chat_history.verticalScrollBar().maximum())
    
    def display_ai_response(self, response):
        # Remove "Thinking..." message
        text = self.chat_history.toPlainText()
        lines = text.split('\n')
        if lines and "Thinking..." in lines[-2]:
            lines = lines[:-2]
            self.chat_history.setText('\n'.join(lines))
        
        # Add actual response
        self.append_chat("🤖 AI", response)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
