import streamlit as st
import pickle
import pandas as pd
import time
import matplotlib.pyplot as plt
import altair as alt

# Set page configuration with wider layout
st.set_page_config(
    page_title="Fake News Detector", 
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.3rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    .stProgress > div > div > div {
        height: 10px;
    }
    .info-box {
        background-color: #B1B7D1;
        border-left: 5px solid #0EA5E9;
        padding: 1rem;
        border-radius: 5px;
    }
    .stTextArea textarea {
        border-radius: 10px;
        border: 1px solid #D1D5DB;
    }
    .stButton button {
        border-radius: 5px;
        font-weight: bold;
        width: 100%;
        height: 3rem;
    }
    .metric-container {
        background-color: #F9FAFB;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .report-section {
        margin-top: 2rem;
        padding: 1.5rem;
        background-color: #F9FAFB;
        border-radius: 10px;
    }
    hr {
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    try:
        nb_model = pickle.load(open("models/naive_bayes.pkl", "rb"))
        lr_model = pickle.load(open("models/logistic_model.pkl", "rb"))
        vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
        return nb_model, lr_model, vectorizer
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

nb_model, lr_model, vectorizer = load_models()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/news.png", width=80)
    st.title("About this app")
    st.markdown("""
    This application uses machine learning to detect fake news based on content analysis.
    
    **How it works:**
    - Enter the news content in the text area
    - Choose a classification model
    - Click the analyze button
    - Review the results and analysis
    
    **Available Models:**
    - **Logistic Regression**: Good for balanced accuracy
    - **Naive Bayes**: Faster prediction with good recall
    """)
    
    st.markdown("---")
    st.subheader("📊 Model Information")
    
    model_metrics = pd.DataFrame({
        "Model": ["Logistic Regression", "Naive Bayes"],
        "Accuracy": ["87%", "84%"],
        "Speed": ["Moderate", "Fast"]
    })
    st.dataframe(model_metrics, hide_index=True, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 🛠️ Developed by")
    st.markdown("AI-powered news analysis team")
    
# Main content
st.markdown('<h1 class="main-header">📰 Fake News Detector</h1>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Advanced text analysis to identify potentially misleading news</p>', unsafe_allow_html=True)

# Create two columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    news_text = st.text_area("📝 Enter News Content", 
                           height=250, 
                           placeholder="Paste or type news content here to analyze...")

with col2:
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("### How to get best results")
    st.markdown("""
    - Include the full article text when possible
    - The more content provided, the more accurate the analysis
    - Try both models for comparison
    - Consider the context and source of the news
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("### Model Selection")
    classifier = st.radio("Choose a classification model:", 
                         ["Logistic Regression", "Naive Bayes"], 
                         horizontal=True)
    
    analyze_button = st.button("🔍 Analyze Content", type="primary", use_container_width=True)

# Handle analysis
if analyze_button:
    if news_text.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing content..."):
            # Create a progress bar for visual effect
            progress_bar = st.progress(0)
            for percent in range(100):
                time.sleep(0.01)  # Simulate processing time
                progress_bar.progress(percent + 1)
            
            # Transform the input text
            tfidf_input = vectorizer.transform([news_text])
            
            # Get probabilities along with predictions
            if classifier == "Naive Bayes":
                result = nb_model.predict(tfidf_input)[0]
                proba = nb_model.predict_proba(tfidf_input)[0]
            else:
                result = lr_model.predict(tfidf_input)[0]
                proba = lr_model.predict_proba(tfidf_input)[0]
            
            # Display results
            st.markdown("## Analysis Results")
            
            cols = st.columns(2)
            
            # Main result display
            with cols[0]:
                if result == 0:
                    st.markdown(f"""
                    <div style="background-color: #ECFDF5; padding: 20px; border-radius: 10px; text-align: center;">
                        <h2 style="color: #059669; margin: 0;">✅ Real News</h2>
                        <p style="margin-top: 10px; font-size: 1.1rem;">This content has characteristics consistent with legitimate news articles.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background-color: #FEF2F2; padding: 20px; border-radius: 10px; text-align: center;">
                        <h2 style="color: #DC2626; margin: 0;">🚨 Fake News</h2>
                        <p style="margin-top: 10px; font-size: 1.1rem;">This content has patterns similar to misleading or fabricated news.</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Confidence metrics
            with cols[1]:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.subheader("Confidence Scores")
                
                real_prob = proba[0] * 100
                fake_prob = proba[1] * 100
                
                # Create chart data
                confidence_data = pd.DataFrame({
                    'Category': ['Real News', 'Fake News'],
                    'Probability': [real_prob, fake_prob]
                })
                
                chart = alt.Chart(confidence_data).mark_bar().encode(
                    x=alt.X('Probability', axis=alt.Axis(format=',.1f'), title='Confidence (%)'),
                    y=alt.Y('Category', title=None),
                    color=alt.condition(
                        alt.datum.Category == 'Real News',
                        alt.value('#059669'),  # green for real
                        alt.value('#DC2626')   # red for fake
                    )
                ).properties(height=150)
                
                st.altair_chart(chart, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Analysis details section
            st.markdown('<div class="report-section">', unsafe_allow_html=True)
            st.subheader("📊 Detailed Analysis")
            
            tabs = st.tabs(["Text Statistics", "Content Analysis", "Interpretation"])
            
            with tabs[0]:
                word_count = len(news_text.split())
                sentence_count = len(news_text.split('.'))
                
                stats_cols = st.columns(3)
                with stats_cols[0]:
                    st.metric("Word Count", f"{word_count}")
                with stats_cols[1]:
                    st.metric("Sentence Count", f"{sentence_count}")
                with stats_cols[2]:
                    st.metric("Avg. Words per Sentence", f"{word_count/max(1, sentence_count):.1f}")
                    
            with tabs[1]:
                st.write("Model used for classification:", classifier)
                st.write(f"Real news probability: {real_prob:.2f}%")
                st.write(f"Fake news probability: {fake_prob:.2f}%")
                st.write(f"Confidence margin: {abs(real_prob - fake_prob):.2f}%")
                
                # Show confidence level interpretation
                confidence_diff = abs(real_prob - fake_prob)
                if confidence_diff > 40:
                    confidence_level = "Very High"
                elif confidence_diff > 20:
                    confidence_level = "High"
                elif confidence_diff > 10:
                    confidence_level = "Moderate"
                else:
                    confidence_level = "Low"
                    
                st.write(f"Confidence level: {confidence_level}")
                
            with tabs[2]:
                st.write("### Interpretation Guide")
                if confidence_level == "Very High":
                    st.write("The model is very confident in its prediction.")
                elif confidence_level == "High":
                    st.write("The model shows good confidence in this classification.")
                elif confidence_level == "Moderate":
                    st.write("The model has moderate confidence. Consider checking other sources.")
                else:
                    st.write("The model has low confidence. This content has mixed signals - further verification is strongly recommended.")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Disclaimer
            st.markdown("""
            <div style="margin-top: 2rem; padding: 10px; background-color: #F3F4F6; border-radius: 5px; font-size: 0.9rem;">
                <strong>Disclaimer:</strong> This tool provides an algorithmic analysis based on text patterns and should not be the sole determinant of news credibility. 
                Always verify information from multiple reputable sources.
            </div>
            """, unsafe_allow_html=True)

# Tips section at the bottom
st.markdown("---")
expander = st.expander("💡 Tips for identifying fake news manually")
with expander:
    st.markdown("""
    ### Key indicators that may suggest fake news:
    1. **Sensationalist headlines** that make extraordinary claims
    2. **Emotional language** designed to trigger strong reactions
    3. **Poor source attribution** or anonymous sources
    4. **Website design** that looks unprofessional or mimics legitimate news sites
    5. **Lack of author information** or publishing date
    6. **Content not reported** by other mainstream news outlets
    7. **Grammatical errors** and typos throughout the article
    """)