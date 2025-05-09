import streamlit as st
import pickle

# Load trained models
nb_model = pickle.load(open("naive_bayes.pkl", "rb"))
lr_model = pickle.load(open("logistic_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detection")

st.markdown("Enter a news article below to check whether it is **Fake** or **Real**.")

classifier = st.radio("Choose a model:", ["Logistic Regression", "Naive Bayes"], horizontal=True)

news_text = st.text_area("📝 News Content", height=300)

if st.button("🔍 Analyze"):                              
    if news_text.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        tfidf_input = vectorizer.transform([news_text])

        if classifier == "Naive Bayes":
            result = nb_model.predict(tfidf_input)[0]
        else:
            result = lr_model.predict(tfidf_input)[0]

        if result == 0:
            st.success("✅ The news is **Real**.")
        else:
            st.error("🚨 The news is **Fake**.")
