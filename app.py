import streamlit as st
import requests
from newspaper import Article
from textblob import TextBlob
from urllib.parse import urlparse
import validators
from PIL import Image
import io
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

# Suppress NLTK download messages
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class NewsNugget:
    def __init__(self):
        # Configure page settings
        st.set_page_config(
            page_title="NewsNugget: AI News Companion",
            page_icon="🗞️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Enhanced custom CSS for modern styling
        st.markdown("""
        <style>
        .main-title {
            font-size: 3rem;
            color: #2C3E50;
            text-align: center;
            background: linear-gradient(to right, #3498DB, #2980B9);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 30px;
            font-weight: bold;
        }
        .stApp {
            background-color: #F0F4F8;
        }
        .stButton>button {
            background-color: #3498DB !important;
            color: white !important;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #2980B9 !important;
            transform: scale(1.05);
        }
        .stExpander {
            border: 1px solid #3498DB;
            border-radius: 5px;
        }
        .metric-container {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        </style>
        """, unsafe_allow_html=True)

    def get_website_name(self, url):
        """Extract clean website name from URL"""
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        return domain.replace('www.', '')

    def fetch_article(self, url):
        """Fetch and parse article from given URL"""
        try:
            article = Article(url)
            article.download()
            article.parse()
            article.nlp()
            return article
        except Exception as e:
            st.error(f"Error parsing the article: {e}")
            return None

    def analyze_text(self, text):
        """Perform comprehensive text analysis"""
        # Sentiment Analysis
        blob = TextBlob(text)
        sentiment_score = blob.sentiment.polarity

        # Advanced Keyword Extraction
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text.lower())
        filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
        word_freq = Counter(filtered_words)
        top_keywords = word_freq.most_common(10)

        # Readability Metrics
        words_count = len(word_tokenize(text))
        sentences_count = len(nltk.sent_tokenize(text))
        avg_words_per_sentence = words_count / sentences_count if sentences_count > 0 else 0

        return {
            'sentiment_score': sentiment_score,
            'top_keywords': top_keywords,
            'words_count': words_count,
            'sentences_count': sentences_count,
            'avg_words_per_sentence': avg_words_per_sentence
        }

    def create_keyword_chart(self, keywords):
        """Create interactive keyword frequency bar chart"""
        df = pd.DataFrame(keywords, columns=['Keyword', 'Frequency'])
        fig = px.bar(
            df, 
            x='Keyword', 
            y='Frequency', 
            title='Top Keywords in the Article',
            color='Frequency',
            color_continuous_scale='blues'
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        return fig

    def create_sentiment_gauge(self, sentiment_score):
        """Create interactive sentiment gauge"""
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = sentiment_score * 50 + 50,  # Map -1 to 1 range to 0 to 100
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Sentiment Score"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps' : [
                    {'range': [0, 33], 'color': "red"},
                    {'range': [33, 66], 'color': "yellow"},
                    {'range': [66, 100], 'color': "green"}
                ]
            }
        ))
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        return fig

    def run(self):
        """Main application runner"""
        st.markdown('<h1 class="main-title"> NewsNugget: Your AI News Companion</h1>', unsafe_allow_html=True)
        
        # URL Input with validation
        url = st.text_input("Enter Article URL", placeholder="https://example.com/news-article")
        
        if st.button("Analyze Article"):
            # Validate URL
            if not url or not validators.url(url):
                st.error("Please enter a valid URL")
                return

            with st.spinner('Fetching and analyzing the article...'):
                article = self.fetch_article(url)
                
                if article:
                    # Article Overview with styled metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f'<div class="metric-container"><h4>Website</h4><p>{self.get_website_name(url)}</p></div>', unsafe_allow_html=True)
                    with col2:
                        st.markdown(f'<div class="metric-container"><h4>Authors</h4><p>{", ".join(article.authors) if article.authors else "N/A"}</p></div>', unsafe_allow_html=True)
                    with col3:
                        st.markdown(f'<div class="metric-container"><h4>Publish Date</h4><p>{article.publish_date.strftime("%B %d, %Y") if article.publish_date else "N/A"}</p></div>', unsafe_allow_html=True)

                    # Top Image with updated parameter
                    if article.top_image:
                        st.image(article.top_image, caption='Article Image', use_container_width=True)

                    # Summary
                    st.subheader("Article Summary")
                    summary = ' '.join(article.summary.split()[:100]) + '...'
                    st.write(summary)

                    # Text Analysis
                    analysis = self.analyze_text(article.text)

                    # Sentiment and Keyword Visualization
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(self.create_sentiment_gauge(analysis['sentiment_score']))
                    with col2:
                        st.plotly_chart(self.create_keyword_chart(analysis['top_keywords']))

                    # Detailed Text Stats
                    st.subheader("Text Statistics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Words", analysis['words_count'])
                    with col2:
                        st.metric("Total Sentences", analysis['sentences_count'])
                    with col3:
                        st.metric("Avg Words/Sentence", f"{analysis['avg_words_per_sentence']:.2f}")

                    # Full Text and Advanced Details
                    with st.expander("Full Article Text"):
                        st.write(article.text)

def main():
    """Initialize and run the NewsNugget application"""
    news_nugget = NewsNugget()
    news_nugget.run()

if __name__ == "__main__":
    main()