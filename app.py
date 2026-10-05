import streamlit as st
import joblib
import re
import string


# -----------------------------
# Load Model and TF-IDF
# -----------------------------

model = joblib.load("amazon_review_svm.pkl")
tfidf = joblib.load("amazon_review_tfidf.pkl")


# -----------------------------
# Stopwords and Stemmer
# -----------------------------

import nltk

nltk.download("stopwords", quiet=True)

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


# -----------------------------
# Text Preprocessing
# Same preprocessing used
# during model training
# -----------------------------

def preprocess_text(text):

    # Convert to string
    text = str(text)

    # Lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)

    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Remove numbers
    text = re.sub(r"\d+", " ", text)

    # Tokenization
    tokens = text.split()

    # Remove stopwords
    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    # Stemming
    tokens = [
        stemmer.stem(word)
        for word in tokens
    ]

    # Join tokens
    return " ".join(tokens)


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(
    page_title="Amazon Review Sentiment Analyzer",
    page_icon="🛒",
    layout="centered"
)

st.title("🛒 Amazon Fine Food Review Sentiment Analyzer")

st.write(
    "Enter an Amazon food product review and "
    "the machine learning model will predict its sentiment."
)


# -----------------------------
# User Input
# -----------------------------

review = st.text_area(
    "Enter your review:",
    placeholder="Example: This product tastes amazing and I really loved it!",
    height=150
)


# -----------------------------
# Prediction
# -----------------------------

if st.button("Analyze Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")

    else:

        # Preprocess review
        clean_review = preprocess_text(review)

        # Convert text to TF-IDF
        review_tfidf = tfidf.transform([clean_review])

        # Predict sentiment
        prediction = model.predict(review_tfidf)[0]

        # Display result
        if prediction == "Positive":
            st.success("🟢 Positive Review")

        elif prediction == "Neutral":
            st.warning("🟡 Neutral Review")

        else:
            st.error("🔴 Negative Review")


# -----------------------------
# Model Information
# -----------------------------

st.divider()

st.subheader("Model Information")

st.write("**Model:** Linear SVM")
st.write("**Vectorization:** TF-IDF")
st.write("**Accuracy:** 92.99%")
st.write("**Macro F1 Score:** 77.01%")