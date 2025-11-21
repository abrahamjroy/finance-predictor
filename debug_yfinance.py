import yfinance as yf
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_ticker(ticker_symbol):
    print(f"Testing {ticker_symbol}...")
    try:
        # Test 1: Info
        ticker = yf.Ticker(ticker_symbol)
        # Force a request
        _ = ticker.info
        print(f"Info fetch successful for {ticker_symbol}")
    except Exception as e:
        print(f"Info fetch failed: {e}")

    try:
        # Test 2: History
        df = yf.download(ticker_symbol, period="1mo", progress=False)
        if not df.empty:
            print(f"History fetch successful. Shape: {df.shape}")
        else:
            print("History fetch returned empty DataFrame")
    except Exception as e:
        print(f"History fetch failed: {e}")

if __name__ == "__main__":
    test_ticker("AAPL")
    test_ticker("NVDA")
