import os
import string
import pickle
import streamlit as st

st.title("🧠 Smart News Classifier")
st.write("App is loading...")

BASE_DIR = os.path.dirname(__file__)

model_path = os.path.join(BASE_DIR, "model/model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "model/vectorizer.pkl")
try:
    model = pickle.load(open(model_path, "rb"))
    vectorizer = pickle.load(open(vectorizer_path, "rb"))
except Exception as e:
    st.error(f"Error: {e}")

label_map = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Technology"
}

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('','',string.punctuation))
    return text

def predict_news(text):
    text = clean_text(text)
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    probs = model.predict_proba(text_vec)[0]
    confidence = max(probs)
    return label_map.get(prediction,"Unknown"), confidence

# UI
st.title("🧠 Smart News Classifier")
user_input = st.text_area("Enter news text:")

if st.button("Classify"):
    if user_input:
        label, confidence  = predict_news(user_input)
        st.success(f"Category: {label}")
        st.info(f"Confidence: {confidence:.2%}")
    else:
        st.warning("Please enter some text.")