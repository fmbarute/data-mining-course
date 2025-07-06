import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

# Text processing libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from textblob import TextBlob

# Machine Learning libraries
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation

# Word cloud
from wordcloud import WordCloud

# Download required NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer


class TextMiningAnalyzer:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.sia = SentimentIntensityAnalyzer()

    def preprocess_text(self, text):
        """Comprehensive text preprocessing"""
        # Convert to lowercase
        text = text.lower()

        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Tokenize
        tokens = word_tokenize(text)

        # Remove stopwords and lemmatize
        processed_tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]

        return ' '.join(processed_tokens)

    def analyze_sentiment(self, texts):
        """Analyze sentiment using VADER and TextBlob"""
        sentiments = []

        for text in texts:
            # VADER sentiment
            vader_scores = self.sia.polarity_scores(text)

            # TextBlob sentiment
            blob = TextBlob(text)
            textblob_polarity = blob.sentiment.polarity
            textblob_subjectivity = blob.sentiment.subjectivity

            sentiments.append({
                'vader_compound': vader_scores['compound'],
                'vader_positive': vader_scores['pos'],
                'vader_negative': vader_scores['neg'],
                'vader_neutral': vader_scores['neu'],
                'textblob_polarity': textblob_polarity,
                'textblob_subjectivity': textblob_subjectivity
            })

        return pd.DataFrame(sentiments)

    def extract_features_tfidf(self, texts, max_features=1000):
        """Extract TF-IDF features"""
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # Include unigrams and bigrams
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.8  # Ignore terms that appear in more than 80% of documents
        )

        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()

        return tfidf_matrix, feature_names, vectorizer

    def perform_topic_modeling(self, texts, n_topics=5):
        """Perform topic modeling using LDA"""
        # Create document-term matrix
        vectorizer = CountVectorizer(
            max_features=1000,
            min_df=2,
            max_df=0.8,
            stop_words='english'
        )
        doc_term_matrix = vectorizer.fit_transform(texts)

        # LDA topic modeling
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=10
        )
        lda.fit(doc_term_matrix)

        # Get topics
        feature_names = vectorizer.get_feature_names_out()
        topics = []

        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics.append({
                'topic_id': topic_idx,
                'words': top_words,
                'weights': topic[top_words_idx]
            })

        return topics, lda, vectorizer

    def text_classification(self, texts, labels):
        """Perform text classification"""
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]

        # Extract features
        tfidf_matrix, _, vectorizer = self.extract_features_tfidf(processed_texts)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            tfidf_matrix, labels, test_size=0.2, random_state=42
        )

        # Train models
        models = {
            'Naive Bayes': MultinomialNB(),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }

        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            results[name] = {
                'model': model,
                'predictions': y_pred,
                'accuracy': model.score(X_test, y_test),
                'classification_report': classification_report(y_test, y_pred)
            }

        return results, vectorizer, X_test, y_test

    def generate_wordcloud(self, texts, title="Word Cloud"):
        """Generate word cloud visualization"""
        # Combine all texts
        combined_text = ' '.join([self.preprocess_text(text) for text in texts])

        # Generate word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=100
        ).generate(combined_text)

        # Plot
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(title, fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def analyze_word_frequency(self, texts, top_n=20):
        """Analyze word frequency"""
        # Preprocess and combine texts
        all_words = []
        for text in texts:
            processed = self.preprocess_text(text)
            all_words.extend(processed.split())

        # Count frequencies
        word_freq = Counter(all_words)
        top_words = word_freq.most_common(top_n)

        # Visualize
        words, frequencies = zip(*top_words)

        plt.figure(figsize=(12, 6))
        plt.bar(words, frequencies)
        plt.title(f'Top {top_n} Most Frequent Words')
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

        return word_freq


def main():
    # Sample data - replace with your own dataset
    sample_texts = [
        "I love this product! It's amazing and works perfectly.",
        "This is the worst service I've ever experienced. Terrible!",
        "The movie was okay, not great but not bad either.",
        "Absolutely fantastic! Highly recommend to everyone.",
        "Poor quality and expensive. Not worth the money.",
        "Great customer service and fast delivery. Very satisfied.",
        "The book was boring and hard to follow.",
        "Excellent value for money. Will buy again!",
        "Disappointing experience. Expected much better.",
        "Outstanding quality and beautiful design."
    ]

    # Sample labels for classification (positive=1, negative=0)
    sample_labels = [1, 0, 0, 1, 0, 1, 0, 1, 0, 1]

    # Initialize analyzer
    analyzer = TextMiningAnalyzer()

    print("=== TEXT MINING ANALYSIS ===\n")

    # 1. Text Preprocessing Example
    print("1. TEXT PREPROCESSING EXAMPLE:")
    original_text = sample_texts[0]
    processed_text = analyzer.preprocess_text(original_text)
    print(f"Original: {original_text}")
    print(f"Processed: {processed_text}\n")

    # 2. Sentiment Analysis
    print("2. SENTIMENT ANALYSIS:")
    sentiment_df = analyzer.analyze_sentiment(sample_texts)
    print(sentiment_df.head())
    print(f"Average sentiment (VADER): {sentiment_df['vader_compound'].mean():.3f}")
    print(f"Average sentiment (TextBlob): {sentiment_df['textblob_polarity'].mean():.3f}\n")

    # 3. TF-IDF Feature Extraction
    print("3. TF-IDF FEATURE EXTRACTION:")
    processed_texts = [analyzer.preprocess_text(text) for text in sample_texts]
    tfidf_matrix, feature_names, vectorizer = analyzer.extract_features_tfidf(processed_texts)
    print(f"TF-IDF Matrix Shape: {tfidf_matrix.shape}")
    print(f"Sample features: {feature_names[:10]}\n")

    # 4. Topic Modeling
    print("4. TOPIC MODELING:")
    topics, lda_model, topic_vectorizer = analyzer.perform_topic_modeling(processed_texts, n_topics=3)
    for topic in topics:
        print(f"Topic {topic['topic_id']}: {', '.join(topic['words'][:5])}")
    print()

    # 5. Text Classification
    print("5. TEXT CLASSIFICATION:")
    classification_results, class_vectorizer, X_test, y_test = analyzer.text_classification(
        sample_texts, sample_labels
    )

    for model_name, result in classification_results.items():
        print(f"{model_name} Accuracy: {result['accuracy']:.3f}")
    print()

    # 6. Word Frequency Analysis
    print("6. WORD FREQUENCY ANALYSIS:")
    word_freq = analyzer.analyze_word_frequency(sample_texts, top_n=10)
    print("Top 10 words:", dict(word_freq.most_common(10)))

    # 7. Generate visualizations
    print("\n7. GENERATING VISUALIZATIONS...")

    # Word frequency plot
    analyzer.analyze_word_frequency(sample_texts, top_n=15)

    # Word cloud
    analyzer.generate_wordcloud(sample_texts, "Sample Text Word Cloud")

    # Sentiment distribution
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.hist(sentiment_df['vader_compound'], bins=10, alpha=0.7)
    plt.title('VADER Sentiment Distribution')
    plt.xlabel('Compound Score')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(sentiment_df['textblob_polarity'], bins=10, alpha=0.7)
    plt.title('TextBlob Sentiment Distribution')
    plt.xlabel('Polarity Score')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    print("\nAnalysis complete! Check the generated plots and results above.")


if __name__ == "__main__":
    main()