import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import pandas as pd
import numpy as np
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QComboBox, 
                             QSlider, QPushButton, QFrame, QMessageBox, QTextEdit, QGraphicsBlurEffect)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QPropertyAnimation, QEasingCurve, QRect
from PyQt6.QtGui import QPainter, QColor, QFont, QPalette, QLinearGradient, QPen, QBrush

import pyqtgraph as pg

from src.data_loader import DataLoader
from src.forecasting import ForecastEngine
from src.sentiment import SentimentEngine
from src.llm_engine import LLMEngine
from src.utils import setup_dirs

# Configure PyQtGraph
pg.setConfigOptions(antialias=True)
pg.setConfigOption('background', 'transparent')
pg.setConfigOption('foreground', 'w')

class GlassFrame(QFrame):
    """Glassmorphism frame with frosted glass effect"""
    def __init__(self, parent=None, blur_radius=15):
        super().__init__(parent)
        self.blur_radius = blur_radius
        self.setAutoFillBackground(False)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Frosted glass background
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QColor(255, 255, 255, 25))  # Semi-transparent white
        gradient.setColorAt(1, QColor(255, 255, 255, 15))
        
        painter.setBrush(QBrush(gradient))
        painter.setPen(QPen(QColor(255, 255, 255, 40), 1))
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 20, 20)

class GlassButton(QPushButton):
    """Premium glass button with hover effects"""
    def __init__(self, text, parent=None, color="#007AFF"):
        super().__init__(text, parent)
        self.base_color = QColor(color)
        self.hovered = False
        self.setMinimumHeight(44)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
    def enterEvent(self, event):
        self.hovered = True
        self.update()
        
    def leaveEvent(self, event):
        self.hovered = False
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Glass effect
        gradient = QLinearGradient(0, 0, 0, self.height())
        if self.hovered:
            gradient.setColorAt(0, QColor(self.base_color.red(), self.base_color.green(), 
                                         self.base_color.blue(), 180))
            gradient.setColorAt(1, QColor(self.base_color.red(), self.base_color.green(), 
                                         self.base_color.blue(), 140))
        else:
            gradient.setColorAt(0, QColor(self.base_color.red(), self.base_color.green(), 
                                         self.base_color.blue(), 150))
            gradient.setColorAt(1, QColor(self.base_color.red(), self.base_color.green(), 
                                         self.base_color.blue(), 110))
        
        painter.setBrush(QBrush(gradient))
        painter.setPen(QPen(QColor(255, 255, 255, 60), 1))
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 12, 12)
        
        # Text
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("SF Pro Display", 13, QFont.Weight.Medium))
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.text())

class TickerTape(QWidget):
    def __init__(self, parent=None, speed=2):
        super().__init__(parent)
        self.setFixedHeight(50)
        self.items = []
        self.speed = speed
        self.offset = 0
        self.font = QFont("SF Mono", 11, QFont.Weight.Medium)
        
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
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Glass background
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QColor(0, 0, 0, 120))
        gradient.setColorAt(1, QColor(0, 0, 0, 80))
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(self.rect())
        
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
                painter.setPen(QColor("#00FF88"))
                
            text_width = painter.fontMetrics().horizontalAdvance(text)
            
            if x + text_width > 0 and x < self.width():
                painter.drawText(x, 32, text)
            
            x += text_width + 50
            
        if x < 0:
            self.offset = self.width()

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
                if isinstance(result, dict):
                    if "error" in result:
                        self.analysis_ready.emit(f"Error: {result['error']}")
                    else:
                        response = result.get("response", "No response")
                        self.analysis_ready.emit(response)
                else:
                    self.analysis_ready.emit(str(result))
        except Exception as e:
            self.analysis_ready.emit(f"Error: {e}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Finance Predictor Pro")
        self.resize(1600, 1000)
        
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
        
        # Initialize
        setup_dirs()
        threading.Thread(target=self.load_llm, daemon=True).start()
        
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
        self.update_ticker_tape_data()

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Top Ticker
        self.top_ticker = TickerTape()
        layout.addWidget(self.top_ticker)
        
        # Main Content with padding
        content_container = QWidget()
        content_container_layout = QVBoxLayout(content_container)
        content_container_layout.setContentsMargins(20, 20, 20, 20)
        
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)
        
        # Chart Area (Glass Panel)
        chart_panel = GlassFrame()
        chart_panel_layout = QVBoxLayout(chart_panel)
        chart_panel_layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title_label = QLabel(f"📈 {self.ticker} Analysis")
        title_label.setFont(QFont("SF Pro Display", 24, QFont.Weight.Bold))
        title_label.setStyleSheet("color: white;")
        chart_panel_layout.addWidget(title_label)
        
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('transparent')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
        self.plot_widget.setLabel('left', 'Price ($)', color='white', size='11pt')
        self.plot_widget.setLabel('bottom', 'Date', color='white', size='11pt')
        self.plot_widget.addLegend()
        chart_panel_layout.addWidget(self.plot_widget)
        
        content_layout.addWidget(chart_panel, stretch=2)
        
        # Sidebar (Glass Panel)
        sidebar = GlassFrame()
        sidebar.setFixedWidth(380)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(20, 20, 20, 20)
        sidebar_layout.setSpacing(15)
        
        # Controls Section
        controls_label = QLabel("⚙️ Controls")
        controls_label.setFont(QFont("SF Pro Display", 18, QFont.Weight.Bold))
        controls_label.setStyleSheet("color: white;")
        sidebar_layout.addWidget(controls_label)
        
        # Ticker Input
        ticker_label = QLabel("Ticker Symbol")
        ticker_label.setFont(QFont("SF Pro Text", 12))
        ticker_label.setStyleSheet("color: rgba(255, 255, 255, 0.7);")
        sidebar_layout.addWidget(ticker_label)
        
        self.ticker_input = QLineEdit("AAPL")
        self.ticker_input.setFont(QFont("SF Mono", 14))
        self.ticker_input.setStyleSheet("""
            QLineEdit {
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 10px;
                padding: 12px;
                color: white;
            }
            QLineEdit:focus {
                background: rgba(255, 255, 255, 0.15);
                border: 1px solid rgba(0, 122, 255, 0.5);
            }
        """)
        self.ticker_input.returnPressed.connect(self.load_data)
        sidebar_layout.addWidget(self.ticker_input)
        
        # Period
        period_label = QLabel("Time Period")
        period_label.setFont(QFont("SF Pro Text", 12))
        period_label.setStyleSheet("color: rgba(255, 255, 255, 0.7);")
        sidebar_layout.addWidget(period_label)
        
        self.period_combo = QComboBox()
        self.period_combo.addItems(["1y", "2y", "5y", "max"])
        self.period_combo.setCurrentText("2y")
        self.period_combo.setFont(QFont("SF Pro Text", 13))
        self.period_combo.setStyleSheet("""
            QComboBox {
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 10px;
                padding: 10px;
                color: white;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid white;
            }
        """)
        sidebar_layout.addWidget(self.period_combo)
        
        # Forecast Days
        self.days_label = QLabel("Forecast: 30 days")
        self.days_label.setFont(QFont("SF Pro Text", 12))
        self.days_label.setStyleSheet("color: rgba(255, 255, 255, 0.7);")
        sidebar_layout.addWidget(self.days_label)
        
        self.days_slider = QSlider(Qt.Orientation.Horizontal)
        self.days_slider.setRange(7, 90)
        self.days_slider.setValue(30)
        self.days_slider.valueChanged.connect(lambda v: self.days_label.setText(f"Forecast: {v} days"))
        self.days_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: rgba(255, 255, 255, 0.1);
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #007AFF;
                width: 20px;
                height: 20px;
                margin: -7px 0;
                border-radius: 10px;
            }
        """)
        sidebar_layout.addWidget(self.days_slider)
        
        # Buttons
        self.run_btn = GlassButton("🚀 Run Predictions", color="#007AFF")
        self.run_btn.clicked.connect(self.run_predictions)
        sidebar_layout.addWidget(self.run_btn)
        
        # Sentiment
        sidebar_layout.addSpacing(10)
        sentiment_title = QLabel("💭 Market Sentiment")
        sentiment_title.setFont(QFont("SF Pro Display", 16, QFont.Weight.Bold))
        sentiment_title.setStyleSheet("color: white;")
        sidebar_layout.addWidget(sentiment_title)
        
        self.sentiment_label = QLabel("Analyzing...")
        self.sentiment_label.setFont(QFont("SF Pro Display", 18, QFont.Weight.Bold))
        self.sentiment_label.setStyleSheet("color: #00FF88;")
        self.sentiment_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sidebar_layout.addWidget(self.sentiment_label)
        
        # AI Chat
        sidebar_layout.addSpacing(10)
        ai_title = QLabel("🤖 AI Assistant")
        ai_title.setFont(QFont("SF Pro Display", 16, QFont.Weight.Bold))
        ai_title.setStyleSheet("color: white;")
        sidebar_layout.addWidget(ai_title)
        
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setFont(QFont("SF Pro Text", 12))
        self.chat_history.setStyleSheet("""
            QTextEdit {
                background: rgba(0, 0, 0, 0.3);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 12px;
                padding: 12px;
                color: white;
            }
        """)
        self.chat_history.setText("💬 AI ready. Ask me anything!")
        sidebar_layout.addWidget(self.chat_history)
        
        # Chat Input
        chat_input_layout = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Ask a question...")
        self.chat_input.setFont(QFont("SF Pro Text", 12))
        self.chat_input.setStyleSheet("""
            QLineEdit {
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 10px;
                padding: 10px;
                color: white;
            }
        """)
        self.chat_input.returnPressed.connect(self.send_chat_message)
        
        self.send_btn = GlassButton("Send", color="#34C759")
        self.send_btn.setMaximumWidth(80)
        self.send_btn.clicked.connect(self.send_chat_message)
        
        chat_input_layout.addWidget(self.chat_input)
        chat_input_layout.addWidget(self.send_btn)
        sidebar_layout.addLayout(chat_input_layout)
        
        content_layout.addWidget(sidebar)
        content_container_layout.addLayout(content_layout)
        layout.addWidget(content_container, stretch=1)
        
        # Bottom Ticker
        self.bottom_ticker = TickerTape()
        layout.addWidget(self.bottom_ticker)
        
    def setup_styling(self):
        # Premium gradient background
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0A0E27,
                    stop:0.5 #1A1F3A,
                    stop:1 #0F1419
                );
            }
        """)

    def load_llm(self):
        self.llm_engine = LLMEngine()
        print("LLM Loaded")

    def update_ticker_tape_data(self):
        tickers = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD",
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS",
            "BTC-USD", "ETH-USD", "GC=F", "CL=F",
        ]
        
        items = []
        for i, ticker in enumerate(tickers * 15):
            price = 100 + (i * 10) % 500
            change = ((i * 7) % 20 - 10) / 10
            color = "#00FF88" if change > 0 else "#FF3B30"
            symbol = "▲" if change > 0 else "▼"
            items.append((f"{ticker}: ${price:.2f} {symbol} {abs(change):.1f}%", color))
        
        self.top_ticker.set_items(items)
        
    def load_data(self):
        ticker = self.ticker_input.text().upper()
        period = self.period_combo.currentText()
        
        self.sentiment_label.setText("Loading...")
        self.sentiment_label.setStyleSheet("color: #FFD60A;")
        QApplication.processEvents()
        
        try:
            self.df = DataLoader.fetch_history(ticker, period)
            self.news = DataLoader.fetch_news(ticker)
            self.ticker = ticker
            
            self.update_chart()
            self.update_sentiment()
            self.update_news_ticker()
            
            QMessageBox.information(self, "✅ Success", f"Loaded {len(self.df)} days of data for {ticker}")
        except Exception as e:
            QMessageBox.critical(self, "❌ Error", f"Failed to load data: {str(e)}")
            self.sentiment_label.setText("N/A")
        
    def update_sentiment(self):
        try:
            score = self.sentiment_analyzer.analyze_news(self.news)
            if score > 0.05:
                color, text, emoji = "#00FF88", "Bullish", "📈"
            elif score < -0.05:
                color, text, emoji = "#FF3B30", "Bearish", "📉"
            else:
                color, text, emoji = "#FFD60A", "Neutral", "➡️"
            
            self.sentiment_label.setText(f"{emoji} {text} ({score:.2f})")
            self.sentiment_label.setStyleSheet(f"color: {color};")
        except Exception as e:
            print(f"Sentiment error: {e}")
    
    def update_news_ticker(self):
        if self.news:
            news_items = [f"📰 {n['publisher']}: {n['title']}" for n in self.news]
            self.bottom_ticker.set_items(news_items * 3)
        else:
            self.bottom_ticker.set_items(["No recent news found."])

    def update_chart(self):
        try:
            self.plot_widget.clear()
            
            if self.df.empty:
                return
            
            timestamps = [d.timestamp() for d in self.df.index]
            
            # Historical data with glow effect
            self.plot_widget.plot(timestamps, self.df['Close'].values, 
                                pen=pg.mkPen(color='#00FF88', width=3), 
                                name='Historical Price',
                                shadowPen=pg.mkPen(color='#00FF88', width=6, alpha=50))
            
            # Predictions
            if self.predictions:
                future_dates = pd.date_range(start=self.df.index[-1] + pd.Timedelta(days=1), 
                                            periods=self.days_slider.value())
                future_timestamps = [d.timestamp() for d in future_dates]
                
                colors = ['#FF3B30', '#007AFF', '#34C759', '#FFD60A', '#FF9500']
                
                for i, (name, values) in enumerate(self.predictions.items()):
                    color = colors[i % len(colors)]
                    self.plot_widget.plot(future_timestamps, values,
                                        pen=pg.mkPen(color=color, width=2, style=Qt.PenStyle.DashLine),
                                        name=name)
                
                avg_pred = np.mean(list(self.predictions.values()), axis=0)
                self.plot_widget.plot(future_timestamps, avg_pred,
                                    pen=pg.mkPen(color='#FFD60A', width=4),
                                    name='ENSEMBLE',
                                    shadowPen=pg.mkPen(color='#FFD60A', width=8, alpha=50))
            
            self.plot_widget.setTitle(f"{self.ticker} Price Analysis", color='w', size='14pt')
            
        except Exception as e:
            print(f"Chart error: {e}")

    def run_predictions(self):
        if self.df.empty:
            QMessageBox.warning(self, "⚠️ No Data", "Please load ticker data first.")
            return
            
        days = self.days_slider.value()
        self.run_btn.setText("⏳ Running...")
        self.run_btn.setEnabled(False)
        QApplication.processEvents()
        
        try:
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
            
            preds = {}
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
                QMessageBox.information(self, "✅ Success", 
                    f"Generated predictions using {len(preds)} models!")
            else:
                QMessageBox.warning(self, "⚠️ No Predictions", "All models failed.")
        except Exception as e:
            QMessageBox.critical(self, "❌ Error", str(e))
        finally:
            self.run_btn.setText("🚀 Run Predictions")
            self.run_btn.setEnabled(True)

    def send_chat_message(self, preset_message=None):
        if not self.llm_engine:
            self.append_chat("🤖", "AI loading...")
            return
        
        message = preset_message if preset_message else self.chat_input.text().strip()
        if not message:
            return
        
        self.chat_input.clear()
        self.append_chat("👤", message)
        self.append_chat("🤖", "Thinking...")
        
        context = f"Stock: {self.ticker}\n"
        if not self.df.empty:
            context += f"Price: ${self.df['Close'].iloc[-1]:.2f}\n"
        
        prompt = f"{context}\nQ: {message}\nA:"
        
        self.ai_thread = AIThread(self.llm_engine, prompt)
        self.ai_thread.analysis_ready.connect(self.display_ai_response)
        self.ai_thread.start()
    
    def append_chat(self, sender, message):
        current = self.chat_history.toPlainText()
        self.chat_history.setText(f"{current}\n{sender}: {message}\n")
        self.chat_history.verticalScrollBar().setValue(
            self.chat_history.verticalScrollBar().maximum())
    
    def display_ai_response(self, response):
        text = self.chat_history.toPlainText()
        lines = text.split('\n')
        if lines and "Thinking..." in lines[-2]:
            lines = lines[:-2]
            self.chat_history.setText('\n'.join(lines))
        
        self.append_chat("🤖", response)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set app-wide font
    app.setFont(QFont("SF Pro Display", 11))
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
