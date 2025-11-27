import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import pandas as pd
import numpy as np
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QComboBox, 
                             QSlider, QPushButton, QFrame, QMessageBox, QTextEdit, QGraphicsDropShadowEffect, QScrollArea, QSizePolicy)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QPainter, QColor, QFont, QPalette, QLinearGradient, QPen, QBrush

import pyqtgraph as pg

from src.data_loader import DataLoader
from src.forecasting import ForecastEngine
from src.sentiment import SentimentEngine
from src.llm_engine import LLMEngine
from src.quant_analysis import QuantAnalyzer
from src.utils import setup_dirs

# Configure PyQtGraph
pg.setConfigOptions(antialias=True)
pg.setConfigOption('background', '#1C1B1F')
pg.setConfigOption('foreground', '#E6E1E5')

class MaterialCard(QFrame):
    """Material Design 3 elevated card"""
    def __init__(self, parent=None, elevation=2):
        super().__init__(parent)
        self.elevation = elevation
        self.setAutoFillBackground(True)
        
        # Material You surface color
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor("#1C1B1F"))  # Surface
        self.setPalette(palette)
        
        # Add shadow for elevation
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(elevation * 8)
        shadow.setColor(QColor(0, 0, 0, 60))
        shadow.setOffset(0, elevation * 2)
        self.setGraphicsEffect(shadow)
        
        self.setStyleSheet(f"""
            MaterialCard {{
                background-color: #1C1B1F;
                border-radius: 16px;
                border: 1px solid #49454F;
            }}
        """)

class MaterialButton(QPushButton):
    """Material Design 3 filled button"""
    def __init__(self, text, parent=None, color="#6750A4"):
        super().__init__(text, parent)
        self.base_color = QColor(color)
        self.hovered = False
        self.setMinimumHeight(48)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFont(QFont("Roboto", 14, QFont.Weight.Medium))
        
        self.update_style()
        
    def update_style(self):
        bg_color = self.base_color.name()
        hover_color = self.base_color.lighter(110).name()
        
        self.setStyleSheet(f"""
            MaterialButton {{
                background-color: {bg_color};
                color: white;
                border: none;
                border-radius: 24px;
                padding: 12px 24px;
                font-weight: 500;
            }}
            MaterialButton:hover {{
                background-color: {hover_color};
            }}
            MaterialButton:pressed {{
                background-color: {self.base_color.darker(110).name()};
            }}
        """)
class TickerTape(QWidget):
    def __init__(self, parent=None, speed=2):
        super().__init__(parent)
        self.setFixedHeight(50)
        self.items = []
        self.speed = speed
        self.offset = 0
        # Use a dot‑matrix style monospaced font. "Digital-7" is a common dot‑matrix font; fallback to Courier New.
        self.font = QFont("Digital-7", 12, QFont.Weight.Bold)
        if not self.font.exactMatch():
            self.font = QFont("Courier New", 12, QFont.Weight.Bold)
        self.font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 2)
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
        # Dark background matching Material theme
        painter.fillRect(self.rect(), QColor("#1C1B1F"))
        painter.setFont(self.font)
        if not self.items:
            return
        x = self.offset
        for item in self.items:
            if isinstance(item, tuple):
                text, color_str = item
                text_color = QColor(color_str)
            else:
                text = item
                text_color = QColor("#4CAF50")  # Material green
            text_width = painter.fontMetrics().horizontalAdvance(text)
            if x + text_width > 0 and x < self.width():
                painter.setPen(text_color)
                painter.drawText(x, 32, text)
            x += text_width + 50
        if x < 0:
            self.offset = self.width()

    def resizeEvent(self, event):
        # Reset offset on resize to avoid disappearing text
        if self.offset < -self.width():
            self.offset = self.width()
        super().resizeEvent(event)

class CustomTitleBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(40)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(15, 0, 15, 0)
        layout.setSpacing(8)
        
        # Window Controls (MacOS Style - Left Side)
        # Using fixed size and explicit background colors to ensure visibility
        self.btn_close = QPushButton()
        self.btn_close.setFixedSize(12, 12)
        self.btn_close.setStyleSheet("""
            QPushButton {
                background-color: #FF5F57;
                border: none;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #FF3B30;
            }
        """)
        self.btn_close.setToolTip("Close")
        
        self.btn_min = QPushButton()
        self.btn_min.setFixedSize(12, 12)
        self.btn_min.setStyleSheet("""
            QPushButton {
                background-color: #FFBD2E;
                border: none;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #FF9500;
            }
        """)
        self.btn_min.setToolTip("Minimize")
        
        self.btn_max = QPushButton()
        self.btn_max.setFixedSize(12, 12)
        self.btn_max.setStyleSheet("""
            QPushButton {
                background-color: #28C93F;
                border: none;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #007AFF;
            }
        """)
        self.btn_max.setToolTip("Maximize")
        
        layout.addWidget(self.btn_close)
        layout.addWidget(self.btn_min)
        layout.addWidget(self.btn_max)
        
        # Title (Centered)
        layout.addStretch()
        self.title_label = QLabel("Finance Predictor Pro")
        self.title_label.setFont(QFont("SF Pro Display", 12, QFont.Weight.Bold))
        self.title_label.setStyleSheet("color: rgba(255, 255, 255, 0.9);")
        layout.addWidget(self.title_label)
        layout.addStretch()
        
        # Spacer to balance the left controls
        # 3 buttons * 12px + 2 spaces * 8px = 52px approx width to balance
        dummy_spacer = QWidget()
        dummy_spacer.setFixedWidth(52)
        layout.addWidget(dummy_spacer)
        
        # Connect signals
        self.btn_min.clicked.connect(self.window().showMinimized)
        self.btn_max.clicked.connect(self.toggle_max)
        self.btn_close.clicked.connect(self.window().close)
        
        self.start_pos = None

    def toggle_max(self):
        if self.window().isMaximized():
            self.window().showNormal()
        else:
            self.window().showMaximized()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.start_pos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event):
        if self.start_pos:
            delta = event.globalPosition().toPoint() - self.start_pos
            self.window().move(self.window().pos() + delta)
            self.start_pos = event.globalPosition().toPoint()

    def mouseReleaseEvent(self, event):
        self.start_pos = None
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
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.resize(1600, 1000)
        
        # State
        self.ticker = "AAPL"
        self.df = pd.DataFrame()
        self.news = []
        self.llm_engine = None
        self.forecaster = ForecastEngine()
        self.sentiment_analyzer = SentimentEngine()
        self.quant_analyzer = QuantAnalyzer()
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
        main_widget.setObjectName("MainWidget")
        self.setCentralWidget(main_widget)
        
        # Main Layout
        layout = QVBoxLayout(main_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Custom Title Bar
        self.title_bar = CustomTitleBar(self)
        layout.addWidget(self.title_bar)
        
        # Top Ticker
        self.top_ticker = TickerTape()
        layout.addWidget(self.top_ticker)
        
        # Main Content with padding
        content_container = QWidget()
        content_container_layout = QVBoxLayout(content_container)
        content_container_layout.setContentsMargins(20, 20, 20, 20)
        
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)
        
        # Chart Area (Material Card)
        chart_panel = MaterialCard(elevation=3)
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
        
        content_layout.addWidget(chart_panel, stretch=1)
        
        # Sidebar (Tabbed Interface)
        from PyQt6.QtWidgets import QTabWidget
        self.sidebar_tabs = QTabWidget()
        self.sidebar_tabs.setFixedWidth(400)
        self.sidebar_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #49454F;
                border-radius: 12px;
                background: #1C1B1F;
            }
            QTabBar::tab {
                background: #2B2930;
                color: #CAC4D0;
                padding: 12px 20px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                margin-right: 4px;
                font-family: "SF Pro Text";
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background: #6750A4;
                color: white;
            }
            QTabBar::tab:hover {
                background: #49454F;
            }
        """)
        
        # --- TAB 1: DASHBOARD ---
        tab_dashboard = QWidget()
        dash_layout = QVBoxLayout(tab_dashboard)
        dash_layout.setSpacing(15)
        dash_layout.setContentsMargins(15, 20, 15, 20)
        
        # Controls Section
        controls_label = QLabel("⚙️ Controls")
        controls_label.setFont(QFont("SF Pro Display", 18, QFont.Weight.Bold))
        controls_label.setStyleSheet("color: white;")
        dash_layout.addWidget(controls_label)
        
        # Ticker Input
        ticker_label = QLabel("Ticker Symbol")
        ticker_label.setFont(QFont("SF Pro Text", 12))
        ticker_label.setStyleSheet("color: rgba(255, 255, 255, 0.7);")
        dash_layout.addWidget(ticker_label)
        
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
        dash_layout.addWidget(self.ticker_input)
        
        # Period
        period_label = QLabel("Time Period")
        period_label.setFont(QFont("SF Pro Text", 12))
        period_label.setStyleSheet("color: rgba(255, 255, 255, 0.7);")
        dash_layout.addWidget(period_label)
        
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
        dash_layout.addWidget(self.period_combo)
        
        # Forecast Days
        self.days_label = QLabel("Forecast: 30 days")
        self.days_label.setFont(QFont("SF Pro Text", 12))
        self.days_label.setStyleSheet("color: rgba(255, 255, 255, 0.7);")
        dash_layout.addWidget(self.days_label)
        
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
        dash_layout.addWidget(self.days_slider)
        
        # Buttons
        self.run_btn = MaterialButton("🚀 Run Predictions", color="#6750A4")
        self.run_btn.clicked.connect(self.run_predictions)
        dash_layout.addWidget(self.run_btn)
        
        # AI Chat (Moved to Dashboard for easy access)
        ai_title = QLabel("🤖 AI Assistant")
        ai_title.setFont(QFont("SF Pro Display", 16, QFont.Weight.Bold))
        ai_title.setStyleSheet("color: white; margin-top: 20px;")
        dash_layout.addWidget(ai_title)
        
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
        self.chat_history.setMinimumHeight(150)
        self.chat_history.setText("💬 AI ready. Ask me anything!")
        dash_layout.addWidget(self.chat_history)
        
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
        
        self.send_btn = MaterialButton("Send", color="#34C759")
        self.send_btn.setFixedSize(80, 40)
        self.send_btn.setFont(QFont("Roboto", 12, QFont.Weight.Medium))
        self.send_btn.clicked.connect(self.send_chat_message)
        
        chat_input_layout.addWidget(self.chat_input)
        chat_input_layout.addWidget(self.send_btn)
        dash_layout.addLayout(chat_input_layout)
        
        dash_layout.addStretch()
        
        # --- TAB 2: ANALYSIS ---
        tab_analysis = QWidget()
        analysis_layout = QVBoxLayout(tab_analysis)
        analysis_layout.setSpacing(15)
        analysis_layout.setContentsMargins(15, 20, 15, 20)
        
        # Quant Stats
        stats_title = QLabel("📊 Quant Stats")
        stats_title.setFont(QFont("SF Pro Display", 16, QFont.Weight.Bold))
        stats_title.setStyleSheet("color: white;")
        analysis_layout.addWidget(stats_title)

        self.stats_frame = QFrame()
        self.stats_frame.setStyleSheet("background: rgba(255, 255, 255, 0.05); border-radius: 8px; padding: 10px;")
        self.stats_layout_inner = QVBoxLayout(self.stats_frame)
        self.stats_labels = {}
        # Added Risk Metrics to this list
        metrics_list = ["Sharpe Ratio", "Volatility", "Max Drawdown", "VaR (95%)", "CVaR (95%)", "Kelly Criterion"]
        for metric in metrics_list:
            lbl = QLabel(f"{metric}: --")
            lbl.setFont(QFont("SF Pro Text", 12))
            lbl.setStyleSheet("color: rgba(255, 255, 255, 0.9);")
            self.stats_layout_inner.addWidget(lbl)
            self.stats_labels[metric] = lbl
        analysis_layout.addWidget(self.stats_frame)
        
        # Sentiment
        sentiment_title = QLabel("💭 Market Sentiment")
        sentiment_title.setFont(QFont("SF Pro Display", 16, QFont.Weight.Bold))
        sentiment_title.setStyleSheet("color: white; margin-top: 10px;")
        analysis_layout.addWidget(sentiment_title)
        
        self.sentiment_label = QLabel("Analyzing...")
        self.sentiment_label.setFont(QFont("SF Pro Display", 18, QFont.Weight.Bold))
        self.sentiment_label.setStyleSheet("color: #00FF88;")
        self.sentiment_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        analysis_layout.addWidget(self.sentiment_label)
        
        analysis_layout.addStretch()

        # --- TAB 3: CHARTING ---
        tab_charting = QWidget()
        chart_layout = QVBoxLayout(tab_charting)
        chart_layout.setSpacing(15)
        chart_layout.setContentsMargins(15, 20, 15, 20)
        
        ind_title = QLabel("📉 Indicators")
        ind_title.setFont(QFont("SF Pro Display", 16, QFont.Weight.Bold))
        ind_title.setStyleSheet("color: white;")
        chart_layout.addWidget(ind_title)

        from PyQt6.QtWidgets import QCheckBox
        self.chk_candle = QCheckBox("Candlestick Chart")
        self.chk_sma = QCheckBox("SMA (20)")
        self.chk_bb = QCheckBox("Bollinger Bands")
        self.chk_ichimoku = QCheckBox("Ichimoku Cloud")
        self.chk_patterns = QCheckBox("Show Patterns")
        self.chk_correlation = QCheckBox("Correlation Matrix")
        
        # Group checkboxes
        self.chart_toggles = [
            self.chk_candle, self.chk_sma, self.chk_bb, 
            self.chk_ichimoku, self.chk_patterns, self.chk_correlation
        ]
        
        for chk in self.chart_toggles:
            chk.setFont(QFont("SF Pro Text", 12))
            chk.setStyleSheet("""
                QCheckBox { color: white; spacing: 8px; }
                QCheckBox::indicator { width: 18px; height: 18px; border-radius: 4px; border: 1px solid #888; }
                QCheckBox::indicator:checked { background: #6750A4; border: 1px solid #6750A4; }
            """)
            chk.stateChanged.connect(self.update_chart)
            chart_layout.addWidget(chk)
            
        chart_layout.addStretch()

        # Add tabs
        self.sidebar_tabs.addTab(tab_dashboard, "Dashboard")
        self.sidebar_tabs.addTab(tab_analysis, "Analysis")
        self.sidebar_tabs.addTab(tab_charting, "Charting")
        
        content_layout.addWidget(self.sidebar_tabs)
        content_container_layout.addLayout(content_layout)
        layout.addWidget(content_container, stretch=1)
        content_container_layout.addLayout(content_layout)
        layout.addWidget(content_container, stretch=1)
        
        # Bottom Ticker
        self.bottom_ticker = TickerTape()
        layout.addWidget(self.bottom_ticker)
        
    def setup_styling(self):
        # Premium gradient background
        # Premium gradient background
        self.setStyleSheet("""
            QMainWindow, #MainWidget {
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
            self.update_quant_stats()
            
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

    def update_quant_stats(self):
        if self.df.empty:
            return
            
        # Calculate metrics
        metrics = self.quant_analyzer.calculate_metrics(self.df)
        risk_metrics = self.quant_analyzer.calculate_risk_metrics(self.df)
        regime = self.quant_analyzer.calculate_market_regime(self.df)
        
        metrics.update(risk_metrics)
        metrics["Market Regime"] = regime["Regime"]
        
        # Update labels
        for key, value in metrics.items():
            if key == "Annual Volatility" and "Volatility" in self.stats_labels:
                self.stats_labels["Volatility"].setText(f"Volatility: {value}")
            elif key in self.stats_labels:
                self.stats_labels[key].setText(f"{key}: {value}")
            elif key == "Market Regime":
                # Dynamic add if missing (hack for now)
                if "Market Regime" not in self.stats_labels:
                    lbl = QLabel(f"Market Regime: {value}")
                    lbl.setFont(QFont("SF Pro Text", 12))
                    lbl.setStyleSheet("color: #FFD60A; font-weight: bold;")
                    self.stats_layout_inner.addWidget(lbl)
                    self.stats_labels["Market Regime"] = lbl
                else:
                    self.stats_labels["Market Regime"].setText(f"Market Regime: {value}")
            
        # Add indicators to DF
        self.df = self.quant_analyzer.add_indicators(self.df)
        self.df = self.quant_analyzer.add_advanced_indicators(self.df)
        self.df = self.quant_analyzer.detect_patterns(self.df)

    def update_chart(self):
        try:
            self.plot_widget.clear()
            
            if self.df.empty:
                return
                
            # Check for Correlation Mode
            if hasattr(self, 'chk_correlation') and self.chk_correlation.isChecked():
                self.plot_correlation_matrix()
                return
            
            timestamps = [d.timestamp() for d in self.df.index]
            
            # Chart Type Selection
            if self.chk_candle.isChecked():
                # Candlestick Chart
                opens = self.df['Open'].values
                closes = self.df['Close'].values
                highs = self.df['High'].values
                lows = self.df['Low'].values
                
                bullish = closes >= opens
                bearish = ~bullish
                
                # Wicks
                for i in range(len(timestamps)):
                    t = timestamps[i]
                    color = '#00FF88' if bullish[i] else '#FF3B30'
                    self.plot_widget.plot([t, t], [lows[i], highs[i]], pen=pg.mkPen(color, width=1))
                
                # Bodies
                width = 24 * 60 * 60 * 0.8 
                
                if np.any(bullish):
                    self.plot_widget.addItem(pg.BarGraphItem(
                        x=np.array(timestamps)[bullish],
                        y0=opens[bullish],
                        height=closes[bullish] - opens[bullish],
                        width=width,
                        brush=pg.mkBrush('#00FF88'),
                        pen=pg.mkPen(None)
                    ))
                    
                if np.any(bearish):
                    self.plot_widget.addItem(pg.BarGraphItem(
                        x=np.array(timestamps)[bearish],
                        y0=closes[bearish],
                        height=opens[bearish] - closes[bearish],
                        width=width,
                        brush=pg.mkBrush('#FF3B30'),
                        pen=pg.mkPen(None)
                    ))
                    
            else:
                # Line Chart (Default)
                self.plot_widget.plot(timestamps, self.df['Close'].values, 
                                    pen=pg.mkPen(color='#00FF88', width=3), 
                                    name='Historical Price',
                                    shadowPen=pg.mkPen(color='#00FF88', width=6, alpha=50))
            
            # --- INDICATORS ---
            
            # SMA
            if self.chk_sma.isChecked() and 'SMA_20' in self.df.columns:
                self.plot_widget.plot(timestamps, self.df['SMA_20'].values,
                                    pen=pg.mkPen(color='#FFD60A', width=2),
                                    name='SMA (20)')
                                    
            # Bollinger Bands
            if self.chk_bb.isChecked() and 'BB_High' in self.df.columns:
                high_plot = self.plot_widget.plot(timestamps, self.df['BB_High'].values,
                                    pen=pg.mkPen(color='rgba(255,255,255,0.3)', width=1),
                                    name='BB High')
                low_plot = self.plot_widget.plot(timestamps, self.df['BB_Low'].values,
                                    pen=pg.mkPen(color='rgba(255,255,255,0.3)', width=1),
                                    name='BB Low')
                fill = pg.FillBetweenItem(low_plot, high_plot, brush=pg.mkBrush(255, 255, 255, 30))
                self.plot_widget.addItem(fill)
                
            # Ichimoku Cloud
            if hasattr(self, 'chk_ichimoku') and self.chk_ichimoku.isChecked() and 'Ichimoku_SpanA' in self.df.columns:
                # Plot lines
                self.plot_widget.plot(timestamps, self.df['Ichimoku_Conversion'].values, pen=pg.mkPen('#00B4D8', width=1.5), name="Tenkan")
                self.plot_widget.plot(timestamps, self.df['Ichimoku_Base'].values, pen=pg.mkPen('#D00000', width=1.5), name="Kijun")
                
                # Plot Spans and Fill
                span_a = self.plot_widget.plot(timestamps, self.df['Ichimoku_SpanA'].values, pen=pg.mkPen(None))
                span_b = self.plot_widget.plot(timestamps, self.df['Ichimoku_SpanB'].values, pen=pg.mkPen(None))
                
                # Fill Cloud (Green if A > B, Red if B > A)
                # pyqtgraph FillBetweenItem is simple, for complex crossovers we use a generic semi-transparent fill
                fill = pg.FillBetweenItem(span_a, span_b, brush=pg.mkBrush(100, 100, 255, 50))
                self.plot_widget.addItem(fill)

            # Patterns
            if hasattr(self, 'chk_patterns') and self.chk_patterns.isChecked():
                for i in range(len(timestamps)):
                    t = timestamps[i]
                    val = self.df['High'].iloc[i]
                    
                    if self.df['Pattern_Doji'].iloc[i]:
                        text = pg.TextItem("😒", anchor=(0.5, 1), color="#FFD60A")
                        text.setPos(t, val)
                        self.plot_widget.addItem(text)
                    elif self.df['Pattern_Hammer'].iloc[i]:
                        text = pg.TextItem("🔨", anchor=(0.5, 1))
                        text.setPos(t, val)
                        self.plot_widget.addItem(text)
                    elif self.df['Pattern_Engulfing'].iloc[i]:
                        text = pg.TextItem("🔥", anchor=(0.5, 1))
                        text.setPos(t, val)
                        self.plot_widget.addItem(text)

            # Predictions
            # Predictions
            if self.predictions:
                future_dates = pd.date_range(start=self.df.index[-1] + pd.Timedelta(days=1), 
                                            periods=self.days_slider.value())
                future_timestamps = [d.timestamp() for d in future_dates]
                
                colors = ['#FF3B30', '#007AFF', '#34C759', '#FFD60A', '#FF9500']
                
                # Plot individual models
                idx = 0
                for name, values in self.predictions.items():
                    if name.startswith("ENSEMBLE"):
                        continue
                        
                    color = colors[idx % len(colors)]
                    self.plot_widget.plot(future_timestamps, values,
                                        pen=pg.mkPen(color=color, width=1, style=Qt.PenStyle.DotLine),
                                        name=name)
                    idx += 1
                
                # Plot Ensembles
                if 'ENSEMBLE (Technical)' in self.predictions:
                    self.plot_widget.plot(future_timestamps, self.predictions['ENSEMBLE (Technical)'],
                                        pen=pg.mkPen(color='#FFD60A', width=2, style=Qt.PenStyle.DashLine),
                                        name='Technical Avg')
                                        
                if 'ENSEMBLE (Sentiment Adjusted)' in self.predictions:
                    self.plot_widget.plot(future_timestamps, self.predictions['ENSEMBLE (Sentiment Adjusted)'],
                                        pen=pg.mkPen(color='#00FF88', width=4), # Green/Gold for main
                                        name='Sentiment Adjusted',
                                        shadowPen=pg.mkPen(color='#00FF88', width=8, alpha=50))
            
            self.plot_widget.setTitle(f"{self.ticker} Price Analysis", color='w', size='14pt')
            
        except Exception as e:
            print(f"Chart error: {e}")

    def plot_correlation_matrix(self):
        """Fetches and plots correlation matrix."""
        self.plot_widget.clear()
        
        # Show loading
        text = pg.TextItem("Calculating Correlation Matrix...\n(Fetching data for SPY, BTC, GLD, etc.)", anchor=(0.5, 0.5), color="white")
        text.setFont(QFont("SF Pro Display", 16))
        self.plot_widget.addItem(text)
        self.plot_widget.autoRange() # Ensure loading text is visible
        QApplication.processEvents()
        
        try:
            # Calculate
            corr_df = self.quant_analyzer.calculate_correlation(self.ticker)
            
            self.plot_widget.clear()
            if corr_df.empty:
                text = pg.TextItem("Failed to calculate correlation.\nCheck internet connection.", anchor=(0.5, 0.5), color="#FF3B30")
                self.plot_widget.addItem(text)
                self.plot_widget.autoRange()
                return
                
            # Plot Heatmap
            # Prepare data (flip y for correct orientation)
            data = corr_df.values
            
            # Create ImageItem
            img = pg.ImageItem(data)
            
            # Colormap (Red to Blue)
            pos = np.array([0.0, 0.5, 1.0])
            color = np.array([[255, 59, 48, 255], [255, 255, 255, 255], [0, 122, 255, 255]], dtype=np.ubyte)
            map = pg.ColorMap(pos, color)
            img.setLookupTable(map.getLookupTable(0.0, 1.0, 256))
            
            self.plot_widget.addItem(img)
            
            # Add labels
            cols = corr_df.columns
            for i in range(len(cols)):
                # X Axis
                lbl = pg.TextItem(cols[i], anchor=(0.5, 0), color="white", angle=0)
                lbl.setPos(i + 0.5, len(cols))
                self.plot_widget.addItem(lbl)
                
                # Y Axis
                lbl = pg.TextItem(cols[i], anchor=(1, 0.5), color="white")
                lbl.setPos(0, i + 0.5)
                self.plot_widget.addItem(lbl)
                
                # Values in cells
                for j in range(len(cols)):
                    val = data[j, i] # Transposed access for visual
                    txt = pg.TextItem(f"{val:.2f}", anchor=(0.5, 0.5), color="black")
                    txt.setPos(i + 0.5, j + 0.5)
                    self.plot_widget.addItem(txt)
            
            # CRITICAL: Reset view range to show the 7x7 matrix
            self.plot_widget.autoRange()
            self.plot_widget.setTitle(f"{self.ticker} Correlation Matrix", color='w', size='14pt')
            
        except Exception as e:
            print(f"Correlation plot error: {e}")
            self.plot_widget.clear()
            text = pg.TextItem(f"Error: {str(e)}", anchor=(0.5, 0.5), color="red")
            self.plot_widget.addItem(text)
            self.plot_widget.autoRange()

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
                # 1. Calculate Raw Ensemble
                ensemble_values = np.mean(list(preds.values()), axis=0)
                preds['ENSEMBLE (Technical)'] = ensemble_values
                
                # 2. Apply Sentiment Adjustment (Bayesian Drift)
                try:
                    sentiment_score = self.sentiment_analyzer.analyze_news(self.news)
                    
                    # Calculate volatility (annualized)
                    returns = self.df['Close'].pct_change().dropna()
                    volatility = returns.std() * np.sqrt(252) if not returns.empty else 0.2
                    
                    adjusted_ensemble = self.forecaster.apply_sentiment_adjustment(
                        ensemble_values, sentiment_score, volatility
                    )
                    
                    preds['ENSEMBLE (Sentiment Adjusted)'] = adjusted_ensemble
                    
                    # Log for user visibility
                    print(f"Sentiment Score: {sentiment_score:.2f}, Volatility: {volatility:.2%}")
                    print(f"Applied drift adjustment to ensemble.")
                    
                except Exception as e:
                    print(f"Sentiment adjustment failed: {e}")
                    preds['ENSEMBLE (Sentiment Adjusted)'] = ensemble_values

                self.predictions = preds
                self.update_chart()
                QMessageBox.information(self, "✅ Success", 
                    f"Generated predictions using {len(preds)} models!\nSentiment Drift Applied.")
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
