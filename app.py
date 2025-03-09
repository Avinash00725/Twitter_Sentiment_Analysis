# app.py
import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Load stopwords
stop_words = set(stopwords.words('english'))

# Load the pre-trained model and TF-IDF vectorizer
lr_model = joblib.load('lr_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Prediction function
def predict_sentiment(tweet):
    cleaned_tweet = preprocess_text(tweet)
    if not cleaned_tweet.strip():
        return "Error: The tweet is too short or contains no meaningful words after preprocessing."
    tweet_tfidf = tfidf.transform([cleaned_tweet]).toarray()
    prediction = lr_model.predict(tweet_tfidf)[0]
    return "Positive" if prediction == 1 else "Negative"

# Streamlit app
st.title("Twitter Sentiment Analysis")
st.write("Enter a tweet to predict its sentiment (Positive or Negative).")
tweet_input = st.text_input("Tweet:", "Type your tweet here...")
if st.button("Predict"):
    if tweet_input:
        result = predict_sentiment(tweet_input)
        st.write("**Predicted Sentiment:**", result)
    else:
        st.write("Please enter a tweet to predict its sentiment.")