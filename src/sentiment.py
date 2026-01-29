import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from typing import List, Dict
from .utils import get_logger

logger = get_logger(__name__)

# Download VADER lexicon if not already downloaded
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    logger.info("Downloading VADER lexicon...")
    nltk.download('vader_lexicon', quiet=True)


class SentimentAnalyzer:
    """
    Sentiment Analysis using VADER (Valence Aware Dictionary and sEntiment Reasoner).
    """
    
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
    
    def analyze(self, text: str) -> float:
        """
        Analyze sentiment of a text string.
        
        Args:
            text: Text to analyze (e.g., news headline)
        
        Returns:
            Compound sentiment score (-1 to +1)
            Positive: > 0.05
            Neutral: -0.05 to 0.05
            Negative: < -0.05
        """
        scores = self.sia.polarity_scores(text)
        return scores['compound']
    
    def analyze_news(self, news_list: List[Dict]) -> float:
        """
        Analyze sentiment of a list of news articles.
        
        Args:
            news_list: List of news dictionaries with 'title' key
        
        Returns:
            Average compound sentiment score
        """
        if not news_list:
            return 0.0
        
        scores = [self.analyze(article.get('title', '')) for article in news_list]
        return sum(scores) / len(scores) if scores else 0.0


# Backward compatibility alias for existing app.py
SentimentEngine = SentimentAnalyzer
