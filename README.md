# 🧠 Smart News Classifier

An AI-powered web app that automatically classifies news articles into categories like **World, Sports, Business, and Technology** using Natural Language Processing (NLP).

---

## 🚀 Demo
You can test the app here:
https://smart-news-classifier-mhs8tekqtapckihpn5avtp.streamlit.app/

👉 Paste a news headline or article and instantly get:

* 📂 Predicted category
* 📊 Confidence score

---

## 🎯 Features

* 🧹 Text preprocessing (cleaning, normalization)
* 🔤 TF-IDF vectorization for feature extraction
* 🤖 Machine Learning model (Logistic Regression)
* 📊 Prediction confidence scoring
* 🌐 Interactive web app UI
* ⚡ Fast and lightweight

---

## 🧠 How It Works

1. Input text is cleaned (lowercase, punctuation removed)
2. Text is converted into numerical features using TF-IDF
3. A trained model predicts the category
4. The app displays the result with confidence level

---

## 🛠️ Tech Stack

* Python
* scikit-learn
* pandas
* Streamlit

---

## 📁 Project Structure

```
smart-news-classifier/
│
├── data/              # Dataset
├── model/             # Saved model + vectorizer
├── src/               # Training & prediction scripts
│   ├── train.py
│   ├── predict.py
│   └── utils.py
│
├── app.py             # Streamlit app
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/smart-news-classifier.git
cd smart-news-classifier

python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

---

## ▶️ Usage

### Train the model:

```bash
python src/train.py
```

### Run the app:

```bash
streamlit run app.py
```

---

## 🧪 Example

**Input:**

```
Apple launches new AI-powered iPhone
```

**Output:**

```
Category: Technology 💻  
Confidence: 92.3%
```

---

## 📊 Model Performance

* Algorithm: Logistic Regression
* Feature extraction: TF-IDF
* Accuracy: 90.5%

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first.

---

## 📬 Contact

If you liked this project or want to collaborate, feel free to reach out!

---

⭐ **If you found this useful, give it a star!**



How to run:

pip install -r requirements.txt
python train.py
streamlit run app.py