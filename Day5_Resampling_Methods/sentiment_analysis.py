import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

# Natural Language Processing libraries
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import re

# Machine Learning libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

# Download required NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('punkt_tab')


class SentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def clean_text(self, text):
        """Basic text cleaning"""
        # Remove URLs, mentions, hashtags
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def vader_sentiment(self, text):
        """VADER sentiment analysis"""
        text = self.clean_text(text)
        scores = self.sia.polarity_scores(text)

        # Determine overall sentiment
        if scores['compound'] >= 0.05:
            sentiment = 'Positive'
        elif scores['compound'] <= -0.05:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'

        return {
            'text': text,
            'compound': scores['compound'],
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu'],
            'sentiment': sentiment
        }

    def textblob_sentiment(self, text):
        """TextBlob sentiment analysis"""
        text = self.clean_text(text)
        blob = TextBlob(text)

        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        # Determine sentiment
        if polarity > 0.1:
            sentiment = 'Positive'
        elif polarity < -0.1:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'

        return {
            'text': text,
            'polarity': polarity,
            'subjectivity': subjectivity,
            'sentiment': sentiment
        }

    def lexicon_based_sentiment(self, text):
        """Simple lexicon-based sentiment analysis"""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful',
                          'fantastic', 'awesome', 'love', 'perfect', 'best',
                          'happy', 'pleased', 'satisfied', 'brilliant', 'outstanding']

        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst',
                          'hate', 'disgusting', 'disappointing', 'poor', 'useless',
                          'sad', 'angry', 'frustrated', 'annoying', 'boring']

        text = self.clean_text(text.lower())
        words = text.split()

        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)

        score = positive_count - negative_count

        if score > 0:
            sentiment = 'Positive'
        elif score < 0:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'

        return {
            'text': text,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'score': score,
            'sentiment': sentiment
        }

    def analyze_sentiment_comprehensive(self, texts):
        """Comprehensive sentiment analysis using multiple methods"""
        results = []

        for text in texts:
            vader_result = self.vader_sentiment(text)
            textblob_result = self.textblob_sentiment(text)
            lexicon_result = self.lexicon_based_sentiment(text)

            result = {
                'original_text': text,
                'cleaned_text': self.clean_text(text),

                # VADER results
                'vader_compound': vader_result['compound'],
                'vader_sentiment': vader_result['sentiment'],

                # TextBlob results
                'textblob_polarity': textblob_result['polarity'],
                'textblob_subjectivity': textblob_result['subjectivity'],
                'textblob_sentiment': textblob_result['sentiment'],

                # Lexicon-based results
                'lexicon_score': lexicon_result['score'],
                'lexicon_sentiment': lexicon_result['sentiment'],
            }

            results.append(result)

        return pd.DataFrame(results)

    def train_ml_sentiment_classifier(self, texts, labels):
        """Train a machine learning sentiment classifier"""
        # Create pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
            ('classifier', LogisticRegression(random_state=42))
        ])

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )

        # Train model
        pipeline.fit(X_train, y_train)

        # Predict
        y_pred = pipeline.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        return {
            'model': pipeline,
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred),
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred
        }

    def visualize_sentiment_analysis(self, df):
        """Create visualizations for sentiment analysis results"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. VADER Sentiment Distribution
        vader_counts = df['vader_sentiment'].value_counts()
        axes[0, 0].pie(vader_counts.values, labels=vader_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('VADER Sentiment Distribution')

        # 2. TextBlob Sentiment Distribution
        textblob_counts = df['textblob_sentiment'].value_counts()
        axes[0, 1].pie(textblob_counts.values, labels=textblob_counts.index, autopct='%1.1f%%')
        axes[0, 1].set_title('TextBlob Sentiment Distribution')

        # 3. Lexicon-based Sentiment Distribution
        lexicon_counts = df['lexicon_sentiment'].value_counts()
        axes[0, 2].pie(lexicon_counts.values, labels=lexicon_counts.index, autopct='%1.1f%%')
        axes[0, 2].set_title('Lexicon-based Sentiment Distribution')

        # 4. VADER Compound Score Distribution
        axes[1, 0].hist(df['vader_compound'], bins=20, alpha=0.7, color='skyblue')
        axes[1, 0].set_title('VADER Compound Score Distribution')
        axes[1, 0].set_xlabel('Compound Score')
        axes[1, 0].set_ylabel('Frequency')

        # 5. TextBlob Polarity vs Subjectivity
        scatter = axes[1, 1].scatter(df['textblob_polarity'], df['textblob_subjectivity'],
                                     c=df['textblob_sentiment'].map({'Positive': 'green',
                                                                     'Negative': 'red',
                                                                     'Neutral': 'gray'}),
                                     alpha=0.6)
        axes[1, 1].set_title('TextBlob: Polarity vs Subjectivity')
        axes[1, 1].set_xlabel('Polarity')
        axes[1, 1].set_ylabel('Subjectivity')

        # 6. Method Comparison
        methods = ['VADER', 'TextBlob', 'Lexicon']
        positive_counts = [
            (df['vader_sentiment'] == 'Positive').sum(),
            (df['textblob_sentiment'] == 'Positive').sum(),
            (df['lexicon_sentiment'] == 'Positive').sum()
        ]
        negative_counts = [
            (df['vader_sentiment'] == 'Negative').sum(),
            (df['textblob_sentiment'] == 'Negative').sum(),
            (df['lexicon_sentiment'] == 'Negative').sum()
        ]
        neutral_counts = [
            (df['vader_sentiment'] == 'Neutral').sum(),
            (df['textblob_sentiment'] == 'Neutral').sum(),
            (df['lexicon_sentiment'] == 'Neutral').sum()
        ]

        x = np.arange(len(methods))
        width = 0.25

        axes[1, 2].bar(x - width, positive_counts, width, label='Positive', color='green', alpha=0.7)
        axes[1, 2].bar(x, neutral_counts, width, label='Neutral', color='gray', alpha=0.7)
        axes[1, 2].bar(x + width, negative_counts, width, label='Negative', color='red', alpha=0.7)

        axes[1, 2].set_title('Sentiment Distribution Comparison')
        axes[1, 2].set_xlabel('Methods')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(methods)
        axes[1, 2].legend()

        plt.tight_layout()
        plt.show()


def main():
    # Sample dataset - various types of text with different sentiments
    sample_texts = [
        "I absolutely love this product! It's amazing and works perfectly.",
        "This is the worst service I've ever experienced. Completely terrible!",
        "The movie was okay, nothing special but not bad either.",
        "Absolutely fantastic! I would highly recommend this to everyone.",
        "Poor quality and overpriced. Definitely not worth the money.",
        "Great customer service and super fast delivery. Very satisfied!",
        "The book was boring and hard to follow. Wasted my time.",
        "Excellent value for money. Will definitely buy again!",
        "Very disappointing experience. Expected much better quality.",
        "Outstanding product with beautiful design and great functionality!",
        "Meh, it's just average. Nothing to write home about.",
        "Incredible! This exceeded all my expectations. Perfect!",
        "Awful experience. Staff was rude and unhelpful.",
        "Good product but delivery was slow. Mixed feelings.",
        "Love the design but functionality could be better.",
        "Hate everything about this. Complete waste of money!",
        "Pretty good overall. Would recommend with some reservations.",
        "This is exactly what I was looking for. Very happy!",
        "Not impressed. Quality doesn't match the price point.",
        "Amazing customer support! They solved my problem quickly."
    ]

    # Initialize sentiment analyzer
    analyzer = SentimentAnalyzer()

    print("=== SENTIMENT ANALYSIS EXAMPLES ===\n")

    # 1. Individual Analysis Examples
    print("1. INDIVIDUAL SENTIMENT ANALYSIS EXAMPLES:\n")

    test_sentences = [
        "I love this amazing product!",
        "This is terrible and I hate it.",
        "It's okay, nothing special."
    ]

    for sentence in test_sentences:
        print(f"Text: '{sentence}'")

        # VADER analysis
        vader_result = analyzer.vader_sentiment(sentence)
        print(f"VADER: {vader_result['sentiment']} (compound: {vader_result['compound']:.3f})")

        # TextBlob analysis
        textblob_result = analyzer.textblob_sentiment(sentence)
        print(f"TextBlob: {textblob_result['sentiment']} (polarity: {textblob_result['polarity']:.3f})")

        # Lexicon-based analysis
        lexicon_result = analyzer.lexicon_based_sentiment(sentence)
        print(f"Lexicon: {lexicon_result['sentiment']} (score: {lexicon_result['score']})")
        print("-" * 50)

    # 2. Comprehensive Analysis
    print("\n2. COMPREHENSIVE SENTIMENT ANALYSIS:")
    df_results = analyzer.analyze_sentiment_comprehensive(sample_texts)

    print(f"\nDataset size: {len(df_results)} texts")
    print(f"Columns: {list(df_results.columns)}")

    # Display sample results
    print("\nSample Results:")
    display_cols = ['original_text', 'vader_sentiment', 'textblob_sentiment', 'lexicon_sentiment']
    print(df_results[display_cols].head(10).to_string(index=False))

    # 3. Statistical Summary
    print("\n3. STATISTICAL SUMMARY:")

    print("VADER Sentiment Distribution:")
    print(df_results['vader_sentiment'].value_counts())
    print(f"Average compound score: {df_results['vader_compound'].mean():.3f}")

    print("\nTextBlob Sentiment Distribution:")
    print(df_results['textblob_sentiment'].value_counts())
    print(f"Average polarity: {df_results['textblob_polarity'].mean():.3f}")
    print(f"Average subjectivity: {df_results['textblob_subjectivity'].mean():.3f}")

    print("\nLexicon-based Sentiment Distribution:")
    print(df_results['lexicon_sentiment'].value_counts())
    print(f"Average score: {df_results['lexicon_score'].mean():.3f}")

    # 4. Agreement Analysis
    print("\n4. METHOD AGREEMENT ANALYSIS:")

    # Check how often methods agree
    vader_textblob_agreement = (df_results['vader_sentiment'] == df_results['textblob_sentiment']).mean()
    vader_lexicon_agreement = (df_results['vader_sentiment'] == df_results['lexicon_sentiment']).mean()
    textblob_lexicon_agreement = (df_results['textblob_sentiment'] == df_results['lexicon_sentiment']).mean()

    print(f"VADER-TextBlob agreement: {vader_textblob_agreement:.2%}")
    print(f"VADER-Lexicon agreement: {vader_lexicon_agreement:.2%}")
    print(f"TextBlob-Lexicon agreement: {textblob_lexicon_agreement:.2%}")

    # All three methods agree
    all_agree = ((df_results['vader_sentiment'] == df_results['textblob_sentiment']) &
                 (df_results['textblob_sentiment'] == df_results['lexicon_sentiment'])).mean()
    print(f"All three methods agree: {all_agree:.2%}")

    # 5. Machine Learning Classifier (if we had labeled data)
    print("\n5. MACHINE LEARNING CLASSIFIER EXAMPLE:")

    # Create synthetic labels based on VADER (for demonstration)
    synthetic_labels = df_results['vader_sentiment'].tolist()

    # Train classifier
    ml_results = analyzer.train_ml_sentiment_classifier(sample_texts, synthetic_labels)
    print(f"ML Classifier Accuracy: {ml_results['accuracy']:.3f}")

    # 6. Visualizations
    print("\n6. GENERATING VISUALIZATIONS...")
    analyzer.visualize_sentiment_analysis(df_results)

    # 7. Detailed Analysis for Specific Cases
    print("\n7. DETAILED ANALYSIS FOR DISAGREEMENT CASES:")

    # Find cases where methods disagree
    disagreement_mask = ~((df_results['vader_sentiment'] == df_results['textblob_sentiment']) &
                          (df_results['textblob_sentiment'] == df_results['lexicon_sentiment']))

    disagreement_cases = df_results[disagreement_mask]

    if len(disagreement_cases) > 0:
        print(f"\nFound {len(disagreement_cases)} disagreement cases:")
        for idx, row in disagreement_cases.head(5).iterrows():
            print(f"\nText: '{row['original_text']}'")
            print(f"VADER: {row['vader_sentiment']} ({row['vader_compound']:.3f})")
            print(f"TextBlob: {row['textblob_sentiment']} ({row['textblob_polarity']:.3f})")
            print(f"Lexicon: {row['lexicon_sentiment']} ({row['lexicon_score']})")

    print("\n=== ANALYSIS COMPLETE ===")
    print("Check the generated visualizations for detailed insights!")


if __name__ == "__main__":
    main()