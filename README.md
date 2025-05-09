# 📰 Fake News Detection with Streamlit

This project is a web application that detects whether a given news article is **real** or **fake** using Natural Language Processing (NLP) and machine learning. It is built using Python, scikit-learn, and Streamlit to provide a clean and interactive interface.

## 📌 Features

- Input news text and detect if it's fake or real
- Built-in models: **Naive Bayes** and **Logistic Regression**
- Utilizes **TF-IDF** vectorization for text preprocessing
- User-friendly web interface with **Streamlit**
- Based on a real-world dataset from **Kaggle**

## 🧠 Models Used

- **TF-IDF Vectorizer**: Transforms text data into feature vectors
- **Multinomial Naive Bayes**: A probabilistic classifier
- **Logistic Regression**: A linear model for binary classification

## 📁 Dataset

- Source: [Fake and Real News Dataset on Kaggle](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
- The dataset consists of two CSV files: `Fake.csv` and `True.csv`
- Place both files in the root directory or modify the file paths in the script accordingly

## 🖥️ Technologies

- Python 3.9+
- pandas
- numpy
- scikit-learn
- nltk
- Streamlit

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection
