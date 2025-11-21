from src.data_loader import DataLoader
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_fix():
    print("Testing DataLoader with custom session...")
    
    # Test 1: History
    df = DataLoader.fetch_history("AAPL", period="1mo")
    if not df.empty:
        print(f"✅ History fetch successful. Shape: {df.shape}")
    else:
        print("❌ History fetch failed.")

    # Test 2: News
    news = DataLoader.fetch_news("AAPL")
    if news:
        print(f"✅ News fetch successful. Count: {len(news)}")
    else:
        print("❌ News fetch failed or empty.")

if __name__ == "__main__":
    test_fix()
