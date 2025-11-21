import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from typing import List, Dict
from .utils import get_logger

logger = get_logger(__name__)

class SentimentEngine:
    """
    Handles sentiment analysis of text data.
    """
    
    def __init__(self):
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            nltk.download('vader_lexicon', quiet=True)
            
        self.sia = SentimentIntensityAnalyzer()

    def analyze_news(self, news_items: List[Dict]) -> float:
        """
        Returns an average compound sentiment score (-1 to 1) for a list of news items.
        """
        if not news_items:
            return 0.0
            
        scores = []
        for item in news_items:
            title = item.get('title', '')
            if title:
                score = self.sia.polarity_scores(title)['compound']
                scores.append(score)
                
        return sum(scores) / len(scores) if scores else 0.0
