import streamlit as st
import pytesseract
from PIL import Image
import pandas as pd
import numpy as np
import shap
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

MODEL_PATH = "fake_news_model.pkl"
VEC_PATH = "vectorizer.pkl"

# Load or train model
def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(VEC_PATH):
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VEC_PATH)
    else:
        data = pd.DataFrame({
            "text": [
                "The earth is flat",
                "NASA confirms earth is round",
                "Aliens landed in Ohio",
                "Government launches satellite"
            ],
            "label": [1, 0, 1, 0]
        })
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(data["text"])
        model = LogisticRegression()
        model.fit(X, data["label"])
        joblib.dump(model, MODEL_PATH)
        joblib.dump(vectorizer, VEC_PATH)
    return model, vectorizer

model, vectorizer = load_model()

# Prediction
def predict(text):
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    return pred, proba, X

# Explanation using SHAP
def get_explanation(X):
    explainer = shap.LinearExplainer(model, vectorizer.transform(vectorizer.get_feature_names_out()), feature_perturbation="interventional")
    shap_values = explainer(X)
    return shap_values

# OCR from image
def extract_text(image):
    return pytesseract.image_to_string(image)

# Streamlit UI
st.set_page_config(page_title="Fake News Detection", layout="centered")
st.title("📰 Fake News Detection with Explainable AI")

option = st.radio("Choose input type:", ("Text Input", "Image Upload"))

if option == "Text Input":
    user_text = st.text_area("Enter News Article Text", height=200)
    if st.button("Analyze"):
        if user_text.strip():
            pred, proba, X = predict(user_text)
            label = "🟢 Real" if pred == 0 else "🔴 Fake"
            st.markdown(f"### Prediction: {label} (Confidence: {max(proba):.2f})")

            shap_values = get_explanation(X)
            st.markdown("#### Explanation (Top words):")
            tokens = vectorizer.get_feature_names_out()
            contrib = shap_values.values[0]
            top_indices = np.argsort(np.abs(contrib))[-5:][::-1]
            for idx in top_indices:
                word = tokens[idx]
                value = contrib[idx]
                st.write(f"• *{word}*: {value:.4f}")
        else:
            st.warning("Please enter some text.")

elif option == "Image Upload":
    uploaded_file = st.file_uploader("Upload an image (e.g. screenshot or photo of article)", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        extracted_text = extract_text(image)
        st.text_area("Extracted Text", extracted_text, height=150)
        if st.button("Analyze Image Text"):
            if extracted_text.strip():
                pred, proba, X = predict(extracted_text)
                label = "🟢 Real" if pred == 0 else "🔴 Fake"
                st.markdown(f"### Prediction: {label} (Confidence: {max(proba):.2f})")

                shap_values = get_explanation(X)
                st.markdown("#### Explanation (Top words):")
                tokens = vectorizer.get_feature_names_out()
                contrib = shap_values.values[0]
                top_indices = np.argsort(np.abs(contrib))[-5:][::-1]
                for idx in top_indices:
                    word = tokens[idx]
                    value = contrib[idx]
                    st.write(f"• *{word}*: {value:.4f}")
            else:
                st.warning("No readable text extracted from image.")
