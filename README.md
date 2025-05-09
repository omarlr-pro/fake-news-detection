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
- streamlit

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the virtual environment:

- On Windows:
  ```bash
  .venv\Scripts\activate
  ```
- On macOS/Linux:
  ```bash
  source .venv/bin/activate
  ```

### 3. Install Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

If you face a `distutils` error, run this (Ubuntu/Debian only):

```bash
sudo apt install python3-distutils
```

## 🚀 Run the App

```bash
streamlit run app.py
```

The app will launch in your browser at: [http://localhost:8501](http://localhost:8501)

## 🧪 Example Usage

1. Enter any news text into the input box
2. Select the model (Naive Bayes or Logistic Regression)
3. Click **Predict**
4. Get an instant result: ✅ Real or ❌ Fake

## 📄 requirements.txt

```
numpy==1.24.4
pandas==1.5.3
scikit-learn
nltk
streamlit
```

## 🙋‍♂️ Author

Made with ❤️ by [Omar Laraje](https://github.com/omarlr-pro)


