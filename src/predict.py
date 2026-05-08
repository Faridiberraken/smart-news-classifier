import pickle
import string

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

sample1 = "Algerian president arrived in Turkey to discuss bilateral trade agreements."
sample2 = "Microsoft to cancel its program of stock options for C-level staff worldwide."
sample3 = "Haney won't train for his upcoming fight against the Mexican champion."
label, confidence  = predict_news(sample1)
print(f"Prediction: {label} (confidence: {confidence:.2%})")
