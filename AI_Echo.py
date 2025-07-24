import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from collections import Counter
import re

# Set page config
st.set_page_config(page_title="AI Echo - Sentiment Analyzer", layout="wide")

# Styling
st.markdown(
    """
    <style>
    .big-font {
        font-size:22px !important;
        font-weight: 500;
    }
    .title-font {
        font-size:28px !important;
        font-weight: 700;
        color: #2c3e50;
    }
    .sub-title {
        font-size:18px !important;
        color: #555;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown("<div class='title-font'>🤖 AI Echo</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Your Smartest Conversational Partner for Sentiment Analysis</div>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3208/3208728.png", width=100)
st.sidebar.title("🔎 Navigation")
option = st.sidebar.radio("Go to", ["🏠 Home", "📊 EDA", "📝 Sentiment Prediction", "📈 Model Evaluation"])

# Load model & vectorizer
with open("C:/Users/Ganesh Baskar/DS_AI Echo/best_sentimental_model.pkl", "rb") as file:
    model = pickle.load(file)
with open("tf_idf_vectoriser.pkl", "rb") as file:
    vectorizer = pickle.load(file)

# HOME
if option == "🏠 Home":
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/4004/4004730.png", width=220)
    with col2:
        st.markdown("### ✨ Welcome to AI Echo!")
        st.markdown("""
        - 📌 **Analyze** sentiment of ChatGPT user reviews in real time  
        - 📊 Visualize key **insights & trends** across time, platform, and users  
        - 🤖 Powered by **Machine Learning + NLP**  
        - 🧠 Useful for: **UX Teams**, **Moderators**, **Product Developers**
        """)
        st.success("Start by selecting a section from the sidebar!")

# SENTIMENT PREDICTION
elif option == "📝 Sentiment Prediction":
    st.markdown("### 🎯 Real-Time Sentiment Prediction")
    user_input = st.text_area("🧾 Enter user review or feedback:")
    if st.button("🔍 Analyze Sentiment"):
        if user_input:
            input_vectorized = vectorizer.transform([user_input])
            prediction = model.predict(input_vectorized)[0]
            prediction_prob = model.predict_proba(input_vectorized)

            st.success(f"**Predicted Sentiment:** {prediction}")
            labels = model.classes_
            prob_df = pd.DataFrame(prediction_prob, columns=labels)
            st.bar_chart(prob_df.T)
        else:
            st.warning("❗ Please enter some text to analyze.")

# EDA
elif option == "📊 EDA":
    st.markdown("### 📊 Exploratory Data Analysis (EDA)")
    df = pd.read_csv("C:/Users/Ganesh Baskar/DS_AI Echo/processed_reviews.csv")
    
    if "text" in df.columns and "sentiment" in df.columns:
        st.subheader("📌 Sample Data")
        st.dataframe(df.head())

        # Sentiment distribution
        st.subheader("📊 Sentiment Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x="sentiment", data=df, palette="viridis", ax=ax)
        st.pyplot(fig)

        # Sentiment by rating
        st.subheader("🎯 Sentiment by Rating")
        fig, ax = plt.subplots(figsize=(8, 5))
        sentiment_by_rating = df.groupby(['rating', 'sentiment']).size().unstack(fill_value=0)
        sentiment_by_rating.plot(kind='barh', colormap='viridis', ax=ax)
        st.pyplot(fig)

        # WordCloud per sentiment
        st.subheader("☁️ WordClouds per Sentiment")
        fig, axes = plt.subplots(3, 1, figsize=(10, 10))
        sentiments = df["sentiment"].unique()
        for i, sentiment in enumerate(sentiments):
            reviews = df[df["sentiment"] == sentiment]["text"].astype(str)
            all_text = " ".join(reviews)
            cleaned = re.sub(r"[^\w\s]", "", all_text)
            cleaned = re.sub(r"\d+", "", cleaned).lower()
            words = cleaned.split()
            unwanted = ["sentences", "tokenized_words", "stemmed_words", "lemmatized_words"]
            filtered_words = [w for w in words if w not in unwanted and len(w) > 2]
            filtered_text = " ".join(filtered_words)
            wordcloud = WordCloud(width=600, height=300, background_color="white").generate(filtered_text)
            axes[i].imshow(wordcloud, interpolation="bilinear")
            axes[i].set_title(f"{sentiment} Sentiment")
            axes[i].axis("off")
        st.pyplot(fig)

        # Sentiment trend over time
        st.subheader("📅 Sentiment Trend Over Time")
        fig, ax = plt.subplots(figsize=(10, 5))
        df["date"] = pd.to_datetime(df["date"])
        trend = df.groupby([df["date"].dt.to_period("M"), "sentiment"]).size().unstack(fill_value=0)
        trend.index = trend.index.to_timestamp()
        trend.plot(marker='o', ax=ax)
        st.pyplot(fig)

        # Verified users
        st.subheader("🧍 Verified Users vs Reviews")
        fig, ax = plt.subplots()
        verified = df.groupby(['verified_purchase', 'sentiment']).size().unstack(fill_value=0)
        verified.plot(kind='bar', colormap='viridis', ax=ax)
        st.pyplot(fig)

        # Review length vs sentiment
        st.subheader("✏️ Review Length vs Sentiment")
        fig, ax = plt.subplots()
        df['review_length'] = df['text'].apply(lambda x: len(str(x).split()))
        avg_length = df.groupby("sentiment")["review_length"].mean()
        avg_length.plot(kind='bar', color=["red", "gray", "green"], ax=ax)
        st.pyplot(fig)

        # Location vs Sentiment
        st.subheader("📍 Location-wise Sentiment")
        fig, ax = plt.subplots(figsize=(8, 5))
        loc_sentiments = df.groupby(["location", "sentiment"]).size().unstack(fill_value=0)
        loc_sentiments.plot(kind='barh', colormap='viridis', ax=ax)
        st.pyplot(fig)

        # Platform vs Sentiment
        st.subheader("📱 Platform-wise Sentiment")
        fig, ax = plt.subplots()
        platform = df.groupby(["platform", "sentiment"]).size().unstack(fill_value=0)
        platform.plot(kind='bar', colormap='coolwarm', ax=ax)
        st.pyplot(fig)

        # Version vs Sentiment
        st.subheader("🔢 ChatGPT Version-wise Sentiment")
        fig, ax = plt.subplots()
        version = df.groupby(["version", "sentiment"]).size().unstack(fill_value=0)
        version.plot(kind='bar', colormap='viridis', ax=ax)
        st.pyplot(fig)

        # Negative themes
        st.subheader("🚨 Common Negative Feedback Themes")
        negative_reviews = df[df["sentiment"] == "Negative"]["text"].astype(str)
        all_text = " ".join(negative_reviews)
        cleaned = re.sub(r"[^\w\s]", "", all_text)
        cleaned = re.sub(r"\d+", "", cleaned).lower()
        words = cleaned.split()
        unwanted = ["sentences", "tokenized_words", "stemmed_words", "lemmatized_words"]
        filtered_words = [w for w in words if w not in unwanted and len(w) > 2]
        word_freq = Counter(filtered_words)
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate_from_frequencies(word_freq)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

# MODEL EVALUATION
elif option == "📈 Model Evaluation":
    st.markdown("### 📈 Model Evaluation Report")
    test_df = pd.read_csv("C:/Users/Ganesh Baskar/DS_AI Echo/processed_reviews.csv")
    if "text" in test_df.columns and "sentiment" in test_df.columns:
        X_test = vectorizer.transform(test_df["text"])
        y_test = test_df["sentiment"]

        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_test)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        st.subheader("📋 Classification Report")
        report = classification_report(y_encoded, le.transform(y_pred), target_names=le.classes_, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}"}))

        st.subheader("🔍 Confusion Matrix")
        cm = confusion_matrix(y_encoded, le.transform(y_pred))
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        if len(le.classes_) == 2:
            st.subheader("🧠 ROC Curve & AUC")
            fpr, tpr, _ = roc_curve(y_encoded, y_proba[:, 1])
            auc_score = roc_auc_score(y_encoded, y_proba[:, 1])
            fig2, ax2 = plt.subplots()
            ax2.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
            ax2.plot([0, 1], [0, 1], "k--")
            ax2.set_xlabel("False Positive Rate")
            ax2.set_ylabel("True Positive Rate")
            ax2.set_title("ROC Curve")
            ax2.legend()
            st.pyplot(fig2)
            st.success(f"✅ ROC-AUC Score: **{auc_score:.2f}**")
    else:
        st.warning("🚫 Dataset must contain 'text' and 'sentiment' columns.")
