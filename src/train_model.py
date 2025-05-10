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

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Load datasets
fake_df = pd.read_csv("data/Fake.csv")
true_df = pd.read_csv("data/True.csv")

# Label the datasets
fake_df["label"] = 1  # 1 for fake
true_df["label"] = 0  # 0 for real

# Combine and shuffle the data
df = pd.concat([fake_df, true_df], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

# Combine title and text into a single content field
df["content"] = (df["title"].fillna('') + " " + df["text"].fillna('')).str.strip()

# Preprocessing: clean and lemmatize text
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()  # Lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

df['content'] = df['content'].apply(clean_text)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(df['content'], df['label'], test_size=0.2, random_state=42)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=7000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train models
nb_model = MultinomialNB()
lr_model = LogisticRegression(max_iter=200)

nb_model.fit(X_train_tfidf, y_train)
lr_model.fit(X_train_tfidf, y_train)

# Evaluate
print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_model.predict(X_test_tfidf)))
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_model.predict(X_test_tfidf)))

# Save models and vectorizer
os.makedirs("models", exist_ok=True)
with open("models/naive_bayes.pkl", "wb") as f:
    pickle.dump(nb_model, f)
with open("models/logistic_model.pkl", "wb") as f:
    pickle.dump(lr_model, f)
with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
