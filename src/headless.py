import sys
import json
import re
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from .data_loader import DataLoader
from .forecasting import ForecastEngine
from .sentiment import SentimentEngine
from .llm_engine import LLMEngine
from .quant_analysis import QuantAnalyzer

class HeadlessSession:
    def __init__(self):
        self.ticker = "AAPL"
        self.df = pd.DataFrame()
        self.news = []
        self.predictions = {}
        self.forecast_days = 30
        
        print("\n" + "="*50)
        print("   FINANCE PREDICTOR PRO - HEADLESS MODE")
        print("="*50)
        print("Initializing Engines...")
        
        self.forecaster = ForecastEngine()
        self.sentiment_analyzer = SentimentEngine()
        self.quant_analyzer = QuantAnalyzer()
        self.llm = LLMEngine()
        
        print("✅ System Ready.")
        print("Type 'exit' or 'quit' to stop.")
        print("-" * 50)

    def start(self):
        # Initial load
        self.load_data("AAPL")
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                if not user_input:
                    continue
                    
                if user_input.lower() in ['exit', 'quit']:
                    print("Goodbye!")
                    break
                
                self.handle_interaction(user_input)
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    def handle_interaction(self, user_input):
        # Build Context
        context = f"Stock: {self.ticker}\n"
        if not self.df.empty:
            context += f"Current Price: ${self.df['Close'].iloc[-1]:.2f}\n"
            
        if self.news:
            context += "\nRecent News:\n"
            for n in self.news[:3]:
                context += f"- {n['title']} ({n['publisher']})\n"
                
        if self.predictions:
            context += "\nModel Predictions:\n"
            if 'ENSEMBLE (Sentiment Adjusted)' in self.predictions:
                preds = self.predictions['ENSEMBLE (Sentiment Adjusted)']
                context += f"- Sentiment Adjusted Forecast (Next {len(preds)} days): Starts ${float(preds[0]):.2f}, Ends ${float(preds[-1]):.2f}\n"
        
        prompt = f"{context}\nUser Request: {user_input}\n\nResponse (if action needed, provide JSON):"
        
        print("🤖 AI: Thinking...", end="", flush=True)
        result = self.llm.analyze(prompt)
        print("\r" + " " * 20 + "\r", end="") # Clear line
        
        if "error" in result:
            print(f"🤖 Error: {result['error']}")
            return

        response = result.get("response", "")
        self.process_response(response)

    def process_response(self, response):
        # Extract JSON commands
        commands_to_execute = []
        
        # 1. Markdown blocks
        code_blocks = re.findall(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        for block in code_blocks:
            try:
                parsed = json.loads(block)
                if isinstance(parsed, list): commands_to_execute.extend(parsed)
                elif isinstance(parsed, dict): commands_to_execute.append(parsed)
            except: pass
            
        # 2. Raw JSON
        if not commands_to_execute:
            raw_matches = re.finditer(r'(\{.*?\"tool\":.*?\})', response, re.DOTALL)
            for match in raw_matches:
                try:
                    commands_to_execute.append(json.loads(match.group(1)))
                except: pass

        # Print AI text (removing JSON)
        clean_response = response
        for block in code_blocks:
            clean_response = clean_response.replace(f"```json{block}```", "")
            clean_response = clean_response.replace(f"```json {block} ```", "")
        
        # Simple cleanup of raw JSON if any remains
        clean_response = re.sub(r'\{.*?\"tool\":.*?\}', '', clean_response, flags=re.DOTALL)
        
        if clean_response.strip():
            print(f"🤖 AI: {clean_response.strip()}")

        # Execute Commands
        for cmd in commands_to_execute:
            tool = cmd.get("tool")
            params = cmd.get("params", {})
            print(f"⚙️ Executing: {tool}...")
            
            if tool == "load_ticker":
                self.load_data(params.get("symbol", "AAPL"))
            elif tool == "run_predictions":
                self.run_predictions()
            elif tool == "set_forecast_days":
                self.forecast_days = int(params.get("days", 30))
                print(f"✅ Forecast horizon set to {self.forecast_days} days.")
            elif tool == "show_chart_indicator":
                print(f"ℹ️ Indicator '{params.get('indicator')}' enabled (internal state).")

    def load_data(self, ticker):
        print(f"📥 Loading data for {ticker}...")
        try:
            self.df = DataLoader.fetch_history(ticker, "2y")
            self.news = DataLoader.fetch_news(ticker)
            self.ticker = ticker
            print(f"✅ Loaded {len(self.df)} days of data.")
            
            # Quick Stats
            current = self.df['Close'].iloc[-1]
            print(f"   Current Price: ${current:.2f}")
            
            # Sentiment
            score = self.sentiment_analyzer.analyze_news(self.news)
            print(f"   Sentiment Score: {score:.2f}")
            
        except Exception as e:
            print(f"❌ Failed to load data: {e}")

    def run_predictions(self):
        if self.df.empty:
            print("⚠️ No data loaded.")
            return
            
        print(f"🚀 Running predictions for {self.forecast_days} days...")
        
        try:
            preds = {}
            # Simplified synchronous run for headless to avoid complexity
            for name, func in self.forecaster.models.items():
                try:
                    res = func(self.df, self.forecast_days)
                    # Handle Series vs Array
                    vals = res.values if hasattr(res, 'values') else res
                    # Flatten
                    vals = np.array(vals).flatten()
                    if len(vals) >= self.forecast_days:
                         preds[name] = vals[:self.forecast_days]
                except Exception as e:
                    # print(f"   Model {name} failed: {e}")
                    pass
            
            if preds:
                ensemble = np.mean(list(preds.values()), axis=0)
                
                # Sentiment Adjust
                score = self.sentiment_analyzer.analyze_news(self.news)
                returns = self.df['Close'].pct_change().dropna()
                vol = returns.std() * np.sqrt(252) if not returns.empty else 0.2
                
                adj_ensemble = self.forecaster.apply_sentiment_adjustment(ensemble, score, vol)
                self.predictions['ENSEMBLE (Sentiment Adjusted)'] = adj_ensemble
                
                start_price = float(self.df['Close'].iloc[-1])
                end_price = float(adj_ensemble[-1])
                change = float(((end_price - start_price) / start_price) * 100)
                
                print(f"✅ Predictions Generated!")
                print(f"   Forecast: ${start_price:.2f} -> ${end_price:.2f} ({change:+.2f}%)")
                print(f"   Volatility: {vol:.2%}")
            else:
                print("❌ All models failed.")
                
        except Exception as e:
            print(f"❌ Prediction error: {e}")

def run_headless():
    session = HeadlessSession()
    session.start()
