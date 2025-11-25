import tkinter as tk
from tkinter import ttk, messagebox
import threading
import pandas as pd
import numpy as np
import webbrowser
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data_loader import DataLoader
from src.forecasting import ForecastEngine
from src.sentiment import SentimentEngine
from src.llm_engine import LLMEngine
from src.utils import setup_dirs


class FinancePredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Finance Predictor Pro")
        self.root.geometry("1400x900")
        self.root.configure(bg='#0e1117')
        
        # State
        self.ticker = "AAPL"
        self.df = pd.DataFrame()
        self.news = []
        self.llm_engine = None
        self.forecaster = ForecastEngine()
        self.sentiment_analyzer = SentimentEngine()
        
        # Setup
        setup_dirs()
        self.setup_ui()
        
        # Load LLM in background
        threading.Thread(target=self.load_llm, daemon=True).start()
        
        # Preload data
        self.load_data()
        
    def setup_ui(self):
        # Top Ticker Frame
        top_ticker_frame = tk.Frame(self.root, bg='#000000', height=40)
        top_ticker_frame.pack(fill=tk.X, side=tk.TOP)
        
        self.top_ticker_label = tk.Label(
            top_ticker_frame,
            text="S&P 500: 4500 ▲ 0.5%  |  DOW: 35000 ▼ 0.2%  |  AAPL: 180.50 ▲ 1.2%  |  MSFT: 320.00 ▲ 0.8%",
            bg='#000000',
            fg='#00FF00',
            font=('Consolas', 10, 'bold')
        )
        self.top_ticker_label.pack(pady=10)
        self.animate_ticker(self.top_ticker_label)
        
        # Main Content Frame
        main_frame = tk.Frame(self.root, bg='#0e1117')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left: Controls
        left_frame = tk.Frame(main_frame, bg='#161b22', width=300)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_frame.pack_propagate(False)
        
        # Title
        tk.Label(left_frame, text="Configuration", bg='#161b22', fg='white', 
                font=('Arial', 14, 'bold')).pack(pady=10)
        
        # Ticker Input
        tk.Label(left_frame, text="Ticker:", bg='#161b22', fg='white').pack(pady=(10, 0))
        self.ticker_entry = tk.Entry(left_frame, bg='#333', fg='white', insertbackground='white')
        self.ticker_entry.insert(0, "AAPL")
        self.ticker_entry.pack(pady=5, padx=10, fill=tk.X)
        self.ticker_entry.bind('<Return>', lambda e: self.load_data())
        
        # Period
        tk.Label(left_frame, text="Period:", bg='#161b22', fg='white').pack(pady=(10, 0))
        self.period_var = tk.StringVar(value="2y")
        period_combo = ttk.Combobox(left_frame, textvariable=self.period_var, 
                                    values=["1y", "2y", "5y", "max"], state='readonly')
        period_combo.pack(pady=5, padx=10, fill=tk.X)
        
        # Forecast Days
        tk.Label(left_frame, text="Forecast Days:", bg='#161b22', fg='white').pack(pady=(10, 0))
        self.days_var = tk.IntVar(value=30)
        self.days_label = tk.Label(left_frame, text="30", bg='#161b22', fg='white')
        self.days_label.pack()
        days_slider = tk.Scale(left_frame, from_=7, to=90, orient=tk.HORIZONTAL,
                              variable=self.days_var, bg='#161b22', fg='white',
                              highlightthickness=0, command=self.update_days_label)
        days_slider.pack(pady=5, padx=10, fill=tk.X)
        
        # Load Data Button
        load_btn = tk.Button(left_frame, text="Load Data", bg='#1f6feb', fg='white',
                            font=('Arial', 10, 'bold'), command=self.load_data)
        load_btn.pack(pady=10, padx=10, fill=tk.X)
        
        # Run Predictions Button
        pred_btn = tk.Button(left_frame, text="Run Predictions", bg='#1f6feb', fg='white',
                            font=('Arial', 10, 'bold'), command=self.run_predictions)
        pred_btn.pack(pady=10, padx=10, fill=tk.X)
        
        # Sentiment Display
        tk.Label(left_frame, text="Sentiment:", bg='#161b22', fg='white',
                font=('Arial', 12, 'bold')).pack(pady=(20, 5))
        self.sentiment_label = tk.Label(left_frame, text="N/A", bg='#161b22', fg='orange',
                                       font=('Arial', 14, 'bold'))
        self.sentiment_label.pack()
        
        # AI Analysis Button
        tk.Label(left_frame, text="AI Analyst:", bg='#161b22', fg='white',
                font=('Arial', 12, 'bold')).pack(pady=(20, 5))
        ai_btn = tk.Button(left_frame, text="Generate Analysis", bg='#238636', fg='white',
                          font=('Arial', 10, 'bold'), command=self.generate_ai_analysis)
        ai_btn.pack(pady=5, padx=10, fill=tk.X)
        
        # AI Output
        self.ai_text = tk.Text(left_frame, bg='#0d1117', fg='white', wrap=tk.WORD,
                              height=10, font=('Arial', 9))
        self.ai_text.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        self.ai_text.insert('1.0', "AI analysis will appear here...")
        self.ai_text.config(state=tk.DISABLED)
        
        # Right: Chart Info
        right_frame = tk.Frame(main_frame, bg='#0e1117')
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Chart Button
        self.chart_btn = tk.Button(right_frame, text="📊 Open Interactive Chart in Browser",
                                   bg='#1f6feb', fg='white', font=('Arial', 12, 'bold'),
                                   command=self.open_chart, height=2)
        self.chart_btn.pack(pady=20, padx=20, fill=tk.X)
        
        # Info Text
        info_text = tk.Text(right_frame, bg='#161b22', fg='white', wrap=tk.WORD,
                           font=('Arial', 10))
        info_text.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        info_text.insert('1.0', """
📈 Finance Predictor Pro

Features:
• Fully interactive Plotly charts (zoom, pan, hover)
• Multiple prediction models (ARIMA, Prophet, XGBoost, etc.)
• Real-time sentiment analysis
• AI-powered analysis with Phi-4

Instructions:
1. Enter a ticker symbol (e.g., AAPL, MSFT, NVDA)
2. Click "Load Data" or press Enter
3. Click "Run Predictions" to see forecasts
4. Click the chart button to open the interactive visualization
5. Use "Generate Analysis" for AI insights

The chart opens in your default web browser with full interactivity!
        """)
        info_text.config(state=tk.DISABLED)
        
        # Bottom Ticker Frame
        bottom_ticker_frame = tk.Frame(self.root, bg='#000000', height=40)
        bottom_ticker_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.bottom_ticker_label = tk.Label(
            bottom_ticker_frame,
            text="Loading news...",
            bg='#000000',
            fg='#00FFFF',
            font=('Consolas', 10, 'bold')
        )
        self.bottom_ticker_label.pack(pady=10)
        self.animate_ticker(self.bottom_ticker_label)
        
    def animate_ticker(self, label):
        """Animate ticker by rotating text"""
        def rotate():
            text = label.cget("text")
            if len(text) > 0:
                label.config(text=text[1:] + text[0])
            label.after(100, rotate)
        rotate()
        
    def update_days_label(self, value):
        self.days_label.config(text=str(int(float(value))))
        
    def load_llm(self):
        self.llm_engine = LLMEngine()
        print("LLM Loaded")
        
    def load_data(self):
        ticker = self.ticker_entry.get().strip().upper()
        period = self.period_var.get()
        
        def fetch():
            try:
                self.df = DataLoader.fetch_history(ticker, period)
                self.news = DataLoader.fetch_news(ticker)
                self.ticker = ticker
                
                # Update sentiment
                score = self.sentiment_analyzer.analyze_news(self.news)
                color = "green" if score > 0.05 else ("red" if score < -0.05 else "orange")
                text = "Bullish" if score > 0.05 else ("Bearish" if score < -0.05 else "Neutral")
                
                self.root.after(0, lambda: self.sentiment_label.config(
                    text=f"{text} ({score:.2f})", fg=color))
                
                # Update news ticker
                if self.news:
                    news_text = "  |  ".join([f"{n['publisher']}: {n['title']}" for n in self.news[:10]])
                    self.root.after(0, lambda: self.bottom_ticker_label.config(text=news_text))
                
                self.root.after(0, lambda: messagebox.showinfo("Success", 
                    f"Loaded {len(self.df)} days of data for {ticker}"))
                    
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
                
        threading.Thread(target=fetch, daemon=True).start()
        
    def run_predictions(self):
        if self.df.empty:
            messagebox.showwarning("No Data", "Please load ticker data first.")
            return
            
        days = self.days_var.get()
        
        def predict():
            try:
                preds = {}
                for name, model_func in self.forecaster.models.items():
                    try:
                        pred_series = model_func(self.df, days)
                        values = pred_series.values if hasattr(pred_series, 'values') else pred_series
                        
                        # Flatten completely
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
                            preds[name] = values
                    except Exception as e:
                        print(f"Model {name} failed: {e}")
                
                self.predictions = preds
                self.root.after(0, lambda: messagebox.showinfo("Success", 
                    f"Generated predictions using {len(preds)} models. Click 'Open Chart' to view!"))
                    
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
                
        threading.Thread(target=predict, daemon=True).start()
        
    def open_chart(self):
        """Create and open interactive Plotly chart"""
        if self.df.empty:
            messagebox.showwarning("No Data", "Please load data first.")
            return
            
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=self.df.index,
            y=self.df['Close'],
            mode='lines',
            name='Historical Price',
            line=dict(color='white', width=2)
        ))
        
        # Predictions
        if self.predictions:
            future_dates = pd.date_range(
                start=self.df.index[-1] + pd.Timedelta(days=1),
                periods=self.days_var.get()
            )
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD']
            
            for i, (name, values) in enumerate(self.predictions.items()):
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=values,
                    mode='lines',
                    name=name,
                    line=dict(color=colors[i % len(colors)], width=2, dash='dash')
                ))
            
            # Ensemble average
            avg_pred = np.mean(list(self.predictions.values()), axis=0)
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=avg_pred,
                mode='lines',
                name='ENSEMBLE AVERAGE',
                line=dict(color='#FFD700', width=3)
            ))
        
        # Layout
        fig.update_layout(
            title=f"{self.ticker} Price Analysis",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            template="plotly_dark",
            hovermode='x unified',
            legend=dict(x=0, y=1),
            height=800
        )
        
        # Save and open
        html_file = "chart.html"
        fig.write_html(html_file)
        webbrowser.open('file://' + os.path.abspath(html_file))
        
    def generate_ai_analysis(self):
        if not self.llm_engine:
            self.ai_text.config(state=tk.NORMAL)
            self.ai_text.delete('1.0', tk.END)
            self.ai_text.insert('1.0', "AI Engine loading...")
            self.ai_text.config(state=tk.DISABLED)
            return
            
        def analyze():
            try:
                prompt = f"Analyze {self.ticker} stock based on recent data."
                response = self.llm_engine.analyze(prompt)
                
                self.root.after(0, lambda: self.update_ai_text(response))
            except Exception as e:
                self.root.after(0, lambda: self.update_ai_text(f"Error: {e}"))
                
        self.ai_text.config(state=tk.NORMAL)
        self.ai_text.delete('1.0', tk.END)
        self.ai_text.insert('1.0', "Analyzing...")
        self.ai_text.config(state=tk.DISABLED)
        
        threading.Thread(target=analyze, daemon=True).start()
        
    def update_ai_text(self, text):
        self.ai_text.config(state=tk.NORMAL)
        self.ai_text.delete('1.0', tk.END)
        self.ai_text.insert('1.0', text)
        self.ai_text.config(state=tk.DISABLED)


if __name__ == "__main__":
    root = tk.Tk()
    app = FinancePredictorApp(root)
    root.mainloop()
