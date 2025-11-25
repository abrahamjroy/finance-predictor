import flet as ft
from flet.plotly_chart import PlotlyChart
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from src.data_loader import DataLoader
from src.forecasting import ForecastEngine
from src.sentiment import SentimentEngine
from src.llm_engine import LLMEngine
from src.utils import setup_dirs
import threading
import time

# Global State
class AppState:
    def __init__(self):
        self.ticker = "AAPL"
        self.period = "2y"
        self.forecast_days = 30
        self.selected_models = []
        self.df = pd.DataFrame()
        self.news = []
        self.predictions = {}
        self.sentiment_score = 0.0
        self.llm_engine = None
        self.chat_history = []
        self.analysis_context = ""

state = AppState()

def main(page: ft.Page):
    page.title = "Finance Predictor Pro"
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 0
    page.spacing = 0
    page.bgcolor = "#0e1117"

    # Initialize Engines
    setup_dirs()
    forecaster = ForecastEngine()
    sentiment_analyzer = SentimentEngine()
    
    # --- UI Components ---

    # 1. Ticker Tape (Top)
    # Using a Row with scroll=HIDDEN, but we will animate by rotating controls
    # 1. Ticker Tape (Top)
    top_ticker_content = ft.Row(spacing=20, scroll=ft.ScrollMode.HIDDEN)
    
    # Animation Logic for Tickers (Smooth Scrolling)
    def animate_tickers():
        offset = 0
        while True:
            try:
                offset += 2
                if top_ticker_content.scroll_control:
                    try:
                        top_ticker_content.scroll_to(offset=offset, duration=50)
                    except: pass
                
                if bottom_ticker_content.scroll_control:
                    try:
                        bottom_ticker_content.scroll_to(offset=offset, duration=50)
                    except: pass
                
                time.sleep(0.05)
            except Exception:
                time.sleep(1)

    threading.Thread(target=animate_tickers, daemon=True).start()
    
    def update_ticker_data():
        assets = {
            "^GSPC": "S&P 500", "^DJI": "DOW", "AAPL": "APPLE", "MSFT": "MICROSOFT", 
            "GOOGL": "GOOGLE", "BTC-USD": "BITCOIN", "ETH-USD": "ETHEREUM"
        }
        items = []
        for symbol, name in assets.items():
            try:
                import yfinance as yf
                t = yf.Ticker(symbol)
                # Fetch 1mo to be safe, but we only need last 2 days
                hist = t.history(period="5d") 
                if len(hist) >= 2:
                    curr = hist['Close'].iloc[-1]
                    prev = hist['Close'].iloc[-2]
                    change = curr - prev
                    pct = (change / prev) * 100
                    color = "green" if change >= 0 else "red"
                    arrow = "▲" if change >= 0 else "▼"
                    items.append(
                        ft.Container(
                            content=ft.Row([
                                ft.Text(f"{name}:", color="white", weight=ft.FontWeight.BOLD, size=12),
                                ft.Text(f"{curr:.2f}", color=color, weight=ft.FontWeight.BOLD, size=12),
                                ft.Text(f"{arrow} {abs(pct):.2f}%", color=color, size=12),
                            ], spacing=5),
                            padding=ft.padding.symmetric(horizontal=10),
                            border=ft.border.all(1, "#333"),
                            border_radius=5,
                            bgcolor="#161b22"
                        )
                    )
            except:
                continue
        
        if items:
            # Duplicate items for scrolling effect
            top_ticker_content.controls = items * 20
            page.update()

    # 1b. Bottom Ticker (News)
    bottom_ticker_content = ft.Row(spacing=40, scroll=ft.ScrollMode.HIDDEN)
    
    def update_news_ticker():
        if not state.news:
            return
        
        items = []
        for item in state.news:
            items.append(
                ft.Container(
                    content=ft.Text(f"{item['publisher']}: {item['title']}", color="cyan", size=12, weight=ft.FontWeight.BOLD),
                    padding=ft.padding.symmetric(horizontal=10)
                )
            )
        bottom_ticker_content.controls = items * 20
        page.update()

    # 3. Main Content Area & UI Components
    chart_container = ft.Container(expand=True)
    sentiment_gauge = ft.Container(padding=10)
    news_list = ft.Column(scroll=ft.ScrollMode.AUTO, height=200)
    chat_history_view = ft.Column(scroll=ft.ScrollMode.AUTO, expand=True)
    
    llm_status_indicator = ft.Icon(name="circle", color="red")
    llm_status_text = ft.Text("AI Model Loading...", color="red")

    def update_sidebar_status():
        if state.llm_engine and state.llm_engine.model:
            llm_status_indicator.name = "check_circle"
            llm_status_indicator.color = "green"
            llm_status_text.value = "AI Model Ready"
            llm_status_text.color = "green"
        else:
            llm_status_indicator.name = "error"
            llm_status_indicator.color = "red"
            llm_status_text.value = "AI Model Offline"
            llm_status_text.color = "red"
        page.update()

    def load_llm():
        if not state.llm_engine:
            state.llm_engine = LLMEngine()
            update_sidebar_status()

    threading.Thread(target=load_llm, daemon=True).start()

    def update_chart():
        df = state.df
        preds = state.predictions
        days = state.forecast_days
        
        if df.empty:
            return

        # Create Plotly Figure
        fig = go.Figure()

        # 1. Historical Candlestick
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='History'
        ))

        # 2. Predictions
        if preds:
            future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=days)
            pred_df = pd.DataFrame(preds, index=future_dates)
            
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD']
            
            for i, col in enumerate(pred_df.columns):
                color = colors[i % len(colors)]
                fig.add_trace(go.Scatter(
                    x=pred_df.index,
                    y=pred_df[col],
                    mode='lines',
                    name=f"Pred: {col}",
                    line=dict(color=color, width=2, dash='dash')
                ))
            
            # Average Prediction
            avg_pred = pred_df.mean(axis=1)
            fig.add_trace(go.Scatter(
                x=pred_df.index,
                y=avg_pred,
                mode='lines',
                name='Average Forecast',
                line=dict(color='#FFD700', width=3)
            ))

        # Layout Configuration
        fig.update_layout(
            title=f"{state.ticker} Price Analysis",
            template="plotly_dark",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            height=600,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis_rangeslider_visible=False 
        )

        # Enable Range Slider for zooming if desired, or just standard zoom
        # fig.update_layout(xaxis_rangeslider_visible=True)

        chart_container.content = PlotlyChart(fig, expand=True)
        page.update()

    def update_sentiment():
        score = sentiment_analyzer.analyze_news(state.news)
        state.sentiment_score = score
        
        color = "green" if score > 0.05 else ("red" if score < -0.05 else "orange")
        text = "Bullish" if score > 0.05 else ("Bearish" if score < -0.05 else "Neutral")
        
        sentiment_gauge.content = ft.Column([
            ft.Text("Market Sentiment", size=20, weight=ft.FontWeight.BOLD),
            ft.Container(
                content=ft.Text(f"{text} ({score:.2f})", size=24, weight=ft.FontWeight.BOLD, color="white"),
                bgcolor=color,
                padding=20,
                border_radius=10,
                alignment=ft.alignment.center
            )
        ])
        
        news_items = []
        for item in state.news[:5]:
            news_items.append(ft.Text(f"• {item['title']}", size=12, selectable=True))
        news_list.controls = news_items
        page.update()

    def load_data(e=None):
        ticker = ticker_input.value
        period = period_dropdown.value
        
        if not ticker:
            return

        state.ticker = ticker
        state.period = period
        
        page.snack_bar = ft.SnackBar(ft.Text(f"Loading data for {ticker}..."))
        page.snack_bar.open = True
        page.update()
        
        try:
            df = DataLoader.fetch_history(ticker, period)
            news = DataLoader.fetch_news(ticker)
            
            if df.empty:
                page.snack_bar = ft.SnackBar(ft.Text(f"No data found for {ticker}"), bgcolor="red")
                page.snack_bar.open = True
                page.update()
                return

            state.df = df
            state.news = news
            
            try: update_chart()
            except Exception as e: print(f"Chart update error: {e}")
                
            try: update_sentiment()
            except Exception as e: print(f"Sentiment update error: {e}")
                
            try: update_news_ticker()
            except Exception as e: print(f"News ticker update error: {e}")
            
            page.snack_bar = ft.SnackBar(ft.Text(f"Loaded {ticker}"), bgcolor="green")
            page.snack_bar.open = True
            page.update()
            
        except Exception as ex:
            print(f"Error loading data: {ex}")
            page.snack_bar = ft.SnackBar(ft.Text(f"Error: {ex}"), bgcolor="red")
            page.snack_bar.open = True
            page.update()

    def run_predictions(e):
        if state.df.empty:
            page.snack_bar = ft.SnackBar(ft.Text("Please load data first (enter ticker)."), bgcolor="orange")
            page.snack_bar.open = True
            page.update()
            return

        page.snack_bar = ft.SnackBar(ft.Text("Running AI Predictions..."))
        page.snack_bar.open = True
        page.update()
        
        selected_models = []
        for cb in model_checkboxes.controls:
            if isinstance(cb, ft.Checkbox) and cb.value:
                selected_models.append(cb.label)
        
        state.selected_models = selected_models
        days = int(forecast_slider.value)
        state.forecast_days = days
        
        preds = {}

        for model_name in selected_models:
            try:
                pred_series = forecaster.models[model_name](state.df, days)
                values = pred_series.values
                
                if len(values) > 0:
                    if isinstance(values[0], (list, np.ndarray, pd.Series)):
                        flat_values = []
                        for v in values:
                            if hasattr(v, '__iter__') and len(v) > 0:
                                flat_values.append(float(v[0]))
                            elif hasattr(v, '__iter__') and len(v) == 0:
                                flat_values.append(0.0)
                            else:
                                flat_values.append(float(v))
                        values = np.array(flat_values)
                    else:
                        values = values.ravel()
                
                preds[model_name] = values
            except Exception as ex:
                print(f"Model {model_name} failed: {ex}")
        
        state.predictions = preds
        
        try: update_chart()
        except Exception as e: print(f"Chart update failed: {e}")
        
        try: update_sentiment()
        except Exception as e: print(f"Sentiment update failed: {e}")

        try: update_news_ticker()
        except Exception as e: print(f"News ticker update failed: {e}")
        
        page.snack_bar = ft.SnackBar(ft.Text("Analysis Complete!"), bgcolor="green")
        page.snack_bar.open = True
        page.update()

    def generate_ai_analysis(e):
        if not state.llm_engine or not state.llm_engine.model:
            page.snack_bar = ft.SnackBar(ft.Text("AI Model not ready yet!"), bgcolor="red")
            page.snack_bar.open = True
            page.update()
            return
            
        if state.df.empty:
            page.snack_bar = ft.SnackBar(ft.Text("Please run predictions first."), bgcolor="orange")
            page.snack_bar.open = True
            page.update()
            return

        page.snack_bar = ft.SnackBar(ft.Text("AI is analyzing..."))
        page.snack_bar.open = True
        page.update()
        
        last_price = state.df['Close'].iloc[-1]
        price_change = state.df['Close'].iloc[-1] - state.df['Close'].iloc[0]
        pct_change = (price_change / state.df['Close'].iloc[0]) * 100
        
        prompt = f"""
        You are a financial analyst. Analyze the following data for {state.ticker}:
        - Current Price: ${last_price:.2f}
        - Period Change: {pct_change:.2f}%
        - News Sentiment Score: {state.sentiment_score:.2f} (-1 to 1)
        
        Recent Headlines:
        {[n['title'] for n in state.news[:3]]}
        
        Provide a concise market outlook and trading recommendation.
        """
        
        analysis = state.llm_engine.analyze(prompt)
        state.analysis_context = analysis
        state.chat_history = [] 
        
        chat_history_view.controls.clear()
        chat_history_view.controls.append(
            ft.Container(
                content=ft.Markdown(analysis),
                bgcolor="#14532d",
                padding=10,
                border_radius=5,
                margin=ft.margin.only(bottom=10)
            )
        )
        page.update()

    def send_chat_message(e):
        if not chat_input.value: return
        
        user_msg = chat_input.value
        chat_input.value = ""
        
        chat_history_view.controls.append(
            ft.Container(
                content=ft.Text(f"You: {user_msg}"),
                bgcolor="#1e3a8a",
                padding=10,
                border_radius=5,
                alignment=ft.alignment.center_right,
                margin=ft.margin.only(bottom=5, left=50)
            )
        )
        page.update()
        
        messages = [{"role": "system", "content": f"Context: {state.analysis_context}"}]
        for msg in state.chat_history[-6:]:
            messages.append(msg)
        messages.append({"role": "user", "content": user_msg})
        
        response = state.llm_engine.chat(messages)
        
        state.chat_history.append({"role": "user", "content": user_msg})
        state.chat_history.append({"role": "assistant", "content": response})
        
        chat_history_view.controls.append(
            ft.Container(
                content=ft.Markdown(f"**AI:** {response}"),
                bgcolor="#14532d",
                padding=10,
                border_radius=5,
                margin=ft.margin.only(bottom=10, right=50)
            )
        )
        page.update()

    # Define inputs and bind events
    ticker_input = ft.TextField(label="Ticker Symbol", value="AAPL", width=200, on_submit=load_data)
    period_dropdown = ft.Dropdown(
        label="History Period",
        width=200,
        options=[
            ft.dropdown.Option("1y"),
            ft.dropdown.Option("2y"),
            ft.dropdown.Option("5y"),
            ft.dropdown.Option("max"),
        ],
        value="2y"
    )
    forecast_slider = ft.Slider(min=7, max=90, divisions=83, value=30, label="{value} days")
    
    model_checkboxes = ft.Column(scroll="auto")
    for model in forecaster.models.keys():
        model_checkboxes.controls.append(ft.Checkbox(label=model, value=True))
        
    chat_input = ft.TextField(hint_text="Ask about the analysis...", expand=True, on_submit=send_chat_message)

    # Sidebar
    sidebar = ft.Container(
        width=300,
        bgcolor="#161b22",
        padding=20,
        content=ft.Column([
            ft.Text("Configuration", size=20, weight=ft.FontWeight.BOLD),
            ft.Divider(),
            ticker_input,
            period_dropdown,
            ft.Text("Forecast Days"),
            forecast_slider,
            ft.Divider(),
            ft.Text("Models", weight=ft.FontWeight.BOLD),
            ft.Container(content=model_checkboxes, height=150),
            ft.Divider(),
            ft.Row([llm_status_indicator, llm_status_text]),
            ft.ElevatedButton("Run Predictions", on_click=run_predictions, width=260, bgcolor="blue"),
        ])
    )
    
    # Main Area
    main_content = ft.Column([
        ft.Container(
            content=top_ticker_content,
            height=40,
            bgcolor="#000000",
            border=ft.border.only(bottom=ft.border.BorderSide(1, "green")),
            padding=ft.padding.only(left=10)
        ),
        ft.Container(
            expand=True,
            padding=20,
            content=ft.Row([
                ft.Column([
                    ft.Text(f"Financial Analysis", size=24, weight=ft.FontWeight.BOLD),
                    chart_container
                ], expand=2),
                ft.Column([
                    ft.Text("Market Intelligence", size=20, weight=ft.FontWeight.BOLD),
                    sentiment_gauge,
                    ft.Text("Recent News", weight=ft.FontWeight.BOLD),
                    news_list,
                    ft.Divider(),
                    ft.Text("AI Analyst", weight=ft.FontWeight.BOLD),
                    ft.ElevatedButton("Generate Analysis", on_click=generate_ai_analysis),
                    ft.Container(content=chat_history_view, expand=True, border=ft.border.all(1, "#424242"), border_radius=5, padding=10),
                    ft.Row([chat_input, ft.IconButton("send", on_click=send_chat_message)])
                ], expand=1)
            ], expand=True)
        ),
        ft.Container(
            content=bottom_ticker_content,
            height=30,
            bgcolor="#000000",
            border=ft.border.only(top=ft.border.BorderSide(1, "orange")),
            padding=ft.padding.only(left=10)
        )
    ], expand=True)

    page.add(
        ft.Row([
            sidebar,
            main_content
        ], expand=True)
    )
    
    threading.Thread(target=update_ticker_data, daemon=True).start()
    
    def initial_load():
        time.sleep(1)
        load_data()
        
    threading.Thread(target=initial_load, daemon=True).start()

if __name__ == "__main__":
    ft.app(target=main)
