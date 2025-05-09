import pandas as pd
import string
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# Load both datasets
fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

# Add labels
fake_df["label"] = 1  # Fake = 1
true_df["label"] = 0  # Real = 0

# Combine
df = pd.concat([fake_df, true_df], axis=0).sample(frac=1).reset_index(drop=True)

# Use only the text column (you can also use title + text if needed)
df['content'] = df['title'] + " " + df['text']

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

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['content'], df['label'], test_size=0.2, random_state=42)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=7000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Models
nb = MultinomialNB()
lr = LogisticRegression(max_iter=200)

nb.fit(X_train_tfidf, y_train)
lr.fit(X_train_tfidf, y_train)

# Evaluate
print("Naive Bayes Accuracy:", accuracy_score(y_test, nb.predict(X_test_tfidf)))
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr.predict(X_test_tfidf)))

# Save models
pickle.dump(nb, open("naive_bayes.pkl", "wb"))
pickle.dump(lr, open("logistic_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
