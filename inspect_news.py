import yfinance as yf
import json

def inspect_news():
    ticker = yf.Ticker("AAPL")
    news = ticker.news
    if news:
        item = news[0]
        content = item.get('content', {})
        print(f"Content Keys: {list(content.keys())}")
        
        if 'canonicalUrl' in content:
            print(f"URL found: {content['canonicalUrl']}")
        elif 'link' in item:
            print(f"Link found: {item['link']}")
        
    else:
        print("No news found.")

if __name__ == "__main__":
    inspect_news()
