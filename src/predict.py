import pickle
import string
import sys

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data (if not already downloaded)
nltk.download('stopwords')
nltk.download('wordnet')

# Load saved models and vectorizer
with open("models/logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Preprocessing
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Input: either from command line or hardcoded
if len(sys.argv) > 1:
    input_text = " ".join(sys.argv[1:])
else:
    input_text = input("Enter a news article (title + body): ")

# Clean and vectorize
cleaned_text = clean_text(input_text)
vectorized_input = vectorizer.transform([cleaned_text])

# Predict
prediction = model.predict(vectorized_input)[0]
proba = model.predict_proba(vectorized_input)[0]

# Display result
label = "FAKE" if prediction == 1 else "REAL"
print(f"\nPrediction: {label}")
print(f"Fake news probability: {proba[1] * 100:.2f}%")
print(f"Real news probability: {proba[0] * 100:.2f}%")
