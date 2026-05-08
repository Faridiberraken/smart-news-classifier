import streamlit as st
import string
import pickle

model = pickle.load(open("model/model.pkl","rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl","rb"))

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