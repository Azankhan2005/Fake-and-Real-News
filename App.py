# app.py
import streamlit as st
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

# Load the model
model = joblib.load("fake_news_model.pkl")

# Preprocessing
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [stemmer.stem(w) for w in tokens if w.isalpha() and w not in stop_words]
    return " ".join(tokens)

# Streamlit UI
st.title("📰 Fake News Detector")
st.write("Enter a news article below to check if it's **Fake** or **Real**.")

input_text = st.text_area("News Text", height=200)

if st.button("Detect"):
    cleaned = clean_text(input_text)
    prediction = model.predict([cleaned])[0]
    st.success(f"🔍 This news is **{prediction}**")
