import pandas as pd
import string
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Load data
fake_df = pd.read_csv("data/Fake.csv")
true_df = pd.read_csv("data/True.csv")

# Label data
fake_df["label"] = 1
true_df["label"] = 0

# Merge and shuffle
df = pd.concat([fake_df, true_df]).sample(frac=1).reset_index(drop=True)
df["content"] = df["title"].fillna('') + " " + df["text"].fillna('')

# Clean text
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

df['content'] = df['content'].apply(clean_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['content'], df['label'], test_size=0.2, random_state=42)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=7000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Models
nb_model = MultinomialNB()
lr_model = LogisticRegression(max_iter=200)

nb_model.fit(X_train_tfidf, y_train)
lr_model.fit(X_train_tfidf, y_train)

# Print accuracy
print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_model.predict(X_test_tfidf)))
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_model.predict(X_test_tfidf)))

# Save models
os.makedirs("models", exist_ok=True)
pickle.dump(nb_model, open("models/naive_bayes.pkl", "wb"))
pickle.dump(lr_model, open("models/logistic_model.pkl", "wb"))
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))
