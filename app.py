

import streamlit as st
import joblib

# Load vectorizer and model
vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("lr_model.jb")

# Page config
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# Custom CSS for gradient background & styling
st.markdown(
    """
    <style>
    /* Gradient background */
    .stApp {
        background: linear-gradient(135deg, #3b82f6, #9333ea);
        color: white;
    }

    /* Title styling */
    h1 {
        text-align: center;
        font-size: 3em;
        font-weight: bold;
        color: white;
    }

    /* Text area */
    textarea {
        border-radius: 12px !important;
        padding: 12px !important;
        font-size: 1em !important;
    }

    /* Buttons */
    button {
        background: linear-gradient(90deg,#2563eb,#7c3aed);
        color: white;
        padding: 12px 20px;
        border-radius: 12px;
        font-weight: 600;
        border: none;
        cursor: pointer;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App Title
st.title("📰 Fake News Detector")
st.write("Enter a News Article below to check whether it is Fake or Real.")

# Input text box
inputn = st.text_area("News Article:", "")

# Prediction button
if st.button("Check News"):
    if inputn.strip():
        transform_input = vectorizer.transform([inputn])
        prediction = model.predict(transform_input)

        if prediction[0] == 1:
            st.success("✅ The News is Real!")
        else:
            st.error("❌ The News is Fake!")
    else:
        st.warning("⚠ Please enter some text to analyze.")