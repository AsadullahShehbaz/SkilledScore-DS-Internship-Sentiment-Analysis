# app.py
# -------------------------------
# Streamlit Sentiment Analysis App
# Models: Naive Bayes & LSTM
# Author: Your Name (SkilledScore Internship Project)
# -------------------------------

import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

# -------------------------------
# Load Pre-trained Models
# -------------------------------
@st.cache_resource
def load_nb_model():
    """Load Naive Bayes model and TF-IDF vectorizer"""
    nb_pipeline = joblib.load(open("models/nb_pipeline.pkl", "rb"))
    return nb_pipeline

@st.cache_resource
def load_lstm_model():
    """Load LSTM model and tokenizer"""
    lstm_model = load_model("models/lstm_model.h5")
    tokenizer = joblib.load(open("models/lstm_tokenizer.pkl", "rb"))
    return lstm_model, tokenizer
# Preprocessing function
stop_words = set(stopwords.words('english'))
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    filtered = [word for word in tokens if word not in stop_words]
    return " ".join(filtered)

# -------------------------------
# Prediction Functions
# -------------------------------
def predict_with_nb(text, model):
    """Predict sentiment using Naive Bayes"""
    prediction = model.predict([text])[0]
    return prediction

def predict_with_lstm(text, model, tokenizer, max_len=100):
    """Predict sentiment using LSTM"""
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len)
    prediction = model.predict(padded)
    label = np.argmax(prediction, axis=1)[0]
    return label

# -------------------------------
# Utility: Label Mapping
# -------------------------------
label_map = {0: "Negative 😡", 1: "Neutral 😐", 2: "Positive 😊"}

# -------------------------------
# Streamlit App UI
# -------------------------------
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("💬 Sentiment Analysis App")
st.write("Choose a model (Naive Bayes or LSTM), enter text, and see sentiment prediction!")

# Sidebar for model selection
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Naive Bayes", "LSTM"]
)

# Input text
user_text = st.text_area("📝 Enter your review here:", "", height=150)

if st.button("🔎 Predict Sentiment"):
    if not user_text.strip():
        st.warning("⚠️ Please enter some text to analyze!")
    else:
        if model_choice == "Naive Bayes":
            nb_model = load_nb_model()
            user_text = preprocess(user_text)
            pred = predict_with_nb(user_text, nb_model)
            st.success(f"### Predicted Sentiment: {label_map[pred]}")
        
        elif model_choice == "LSTM":
            lstm_model, tokenizer = load_lstm_model()
            user_text = preprocess(user_text)
            pred = predict_with_lstm(user_text, lstm_model, tokenizer)
            st.success(f"### Predicted Sentiment: {label_map[pred]}")

# Footer
st.markdown(
    """
    ---
    ✅ Built with Streamlit | Internship Project @ **SkilledScore.com**  
    ✨ Models: Naive Bayes & LSTM | Dataset: Amazon Product Reviews
    """
)
 # --- Connect With Me Section ---
st.sidebar.markdown("### 🔗 Connect With Me")

st.sidebar.markdown(
    """
    <div style="display: flex; gap: 15px; flex-wrap: wrap;">
        <a href="https://www.linkedin.com/in/asadullah-shehbaz-18172a2bb/" target="_blank">
            <img src="https://img.shields.io/badge/-LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white">
        </a>
        <a href="https://github.com/AsadullahShehbaz" target="_blank">
            <img src="https://img.shields.io/badge/-GitHub-181717?style=for-the-badge&logo=github&logoColor=white">
        </a>
        <a href="https://www.kaggle.com/asadullahcreative" target="_blank">
            <img src="https://img.shields.io/badge/-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white">
        </a>
        <a href="https://web.facebook.com/profile.php?id=61576230402114" target="_blank">
            <img src="https://img.shields.io/badge/-Facebook-1877F2?style=for-the-badge&logo=facebook&logoColor=white">
        </a>
        <a href="https://www.instagram.com/asad_ullahshehbaz/" target="_blank">
            <img src="https://img.shields.io/badge/-Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white">
        </a>
    </div>
    """,
    unsafe_allow_html=True
)
