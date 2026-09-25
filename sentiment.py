"""Fetch recent headlines from NewsAPI and score them for sentiment with VADER.

Uses NewsAPI's free "Developer" plan: 100 requests/day, articles delayed up
to 24 hours, and search limited to the last month. Fine for a live "current
sentiment" read; not enough depth to backtest sentiment historically.
"""

import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


def get_news_sentiment(query, api_key, page_size=15):
    """Return (average_compound_score, headlines) or (None, []) if unavailable."""
    if not api_key:
        return None, []

    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": query,
                "sortBy": "publishedAt",
                "language": "en",
                "pageSize": page_size,
                "apiKey": api_key,
            },
            timeout=10,
        )
        data = resp.json()
        if data.get("status") != "ok":
            return None, []

        headlines = []
        for article in data.get("articles", []):
            title = (article.get("title") or "").strip()
            if not title:
                continue
            score = _analyzer.polarity_scores(title)["compound"]
            headlines.append({"title": title, "url": article.get("url"), "score": score})

        if not headlines:
            return None, []

        avg_score = sum(h["score"] for h in headlines) / len(headlines)
        return avg_score, headlines
    except Exception:
        return None, []
