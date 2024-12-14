import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import pandas as pd

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

def fetch_news(api_key, query, from_date, to_date):
    url = f"https://newsapi.org/v2/everything?q={query}&from={from_date}&to={to_date}&sortBy=popularity&apiKey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json().get('articles', [])
        return [article['title'] + " " + article['description'] for article in articles]
    else:
        print("Failed to fetch news")
        return []

def fetch_social_media():
    # Placeholder: Replace with Twitter API or Reddit API fetching code
    posts = [
        "This stock is doing great! Time to buy!",
        "Market sentiment looks bad for this stock. Maybe sell it.",
        "Hold for now. Unclear what the market is doing."
    ]
    return posts

def analyze_sentiment(text):
    # Use VADER for sentiment analysis
    vader_score = analyzer.polarity_scores(text)['compound']
    # Use TextBlob for comparison (optional)
    blob_score = TextBlob(text).sentiment.polarity
    # Combine scores (optional: you can weigh them differently)
    return (vader_score + blob_score) / 2

def compute_overall_sentiment(sentiments):
    avg_score = sum(sentiments) / len(sentiments) if sentiments else 0
    scaled_score = int((avg_score + 1) * 50)  # Scale to 1-100
    if scaled_score <= 40:
        return scaled_score, "Negative Sentiment (Strong Sell)"
    elif scaled_score <= 60:
        return scaled_score, "Neutral Sentiment"
    else:
        return scaled_score, "Positive Sentiment (Strong Buy)"

# Main program
if __name__ == "__main__":
    # Configuration
    NEWS_API_KEY = "your_news_api_key_here"
    STOCK_QUERY = "TSLA"  # Example stock query
    FROM_DATE = "2024-12-01"
    TO_DATE = "2024-12-14"

    # Fetch and analyze news data
    news_articles = fetch_news(NEWS_API_KEY, STOCK_QUERY, FROM_DATE, TO_DATE)
    news_sentiments = [analyze_sentiment(article) for article in news_articles]

    # Fetch and analyze social media data
    social_posts = fetch_social_media()
    social_sentiments = [analyze_sentiment(post) for post in social_posts]

    # Combine all sentiments
    all_sentiments = news_sentiments + social_sentiments

    # Compute overall sentiment
    overall_score, sentiment_category = compute_overall_sentiment(all_sentiments)

    # Display results
    print(f"Overall Sentiment Score: {overall_score}")
    print(f"Sentiment Category: {sentiment_category}")

    # Optional: Save to CSV
    data = {
        'Source': ['News'] * len(news_sentiments) + ['Social Media'] * len(social_sentiments),
        'Text': news_articles + social_posts,
        'Sentiment Score': news_sentiments + social_sentiments
    }
    df = pd.DataFrame(data)
    df.to_csv("sentiment_analysis_results.csv", index=False)
    print("Sentiment analysis results saved to 'sentiment_analysis_results.csv'")
