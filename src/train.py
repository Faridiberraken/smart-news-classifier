import pandas as pd
import string
import pickle
import logging

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO)

df = pd.read_csv('AG News dataset/train.csv')
logging.info("-----Data imported: shape: %s", df.shape)

df.columns = ['label', 'text', 'description']
df['text'] = df['text'] + " " + df['description']
df = df.drop('description', axis =1)

logging.info("New column names: %s", df.columns)
logging.info("Data first 5 rows: \n %s",df.head(5))
logging.info("Distribution: %s",(df['label'].value_counts(normalize = True)*100).round(0).astype(int).astype(str)+ "%")

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('','',string.punctuation))
    return text

df['text'] = df['text'].apply(clean_text)

logging.info("-----Text cleaned.")

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'],test_size= 0.2, random_state= 42)
logging.info("X_train shape: %s", X_train.shape)
logging.info("X_train first rows: %s", X_train.head())
logging.info("X_train count NA: %s", X_train.isna().sum())
vectorizer = TfidfVectorizer(max_features= 5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

logging.info("-----Vectorizer created")

model = LogisticRegression(max_iter= 100)
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_true= y_test, y_pred=y_pred)
logging.info("----- Model trained")
logging.info("Accuracy: %s", accuracy)

with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)