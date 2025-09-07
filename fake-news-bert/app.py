# 📦 Core libraries
import os
import re
import pickle
import numpy as np
import pandas as pd

# 🎛 App interface
import streamlit as st

# 🤖 Machine learning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 🖼️ Image processing & OCR
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract

# 🔧 Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 💾 Load model and vectorizer (from training step)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# 👀 Show class labels in sidebar for debugging
st.sidebar.write("🧪 Model classes:", model.classes_)

# 🧹 Clean text
def clean(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# 🔮 Predict label and confidence
def predict_news(text):
    cleaned_text = clean(text)
    st.write("🧹 Cleaned Text:", cleaned_text)
    if len(cleaned_text.split()) < 2:
        return "⚠ Not enough content to make a reliable prediction."

    vec = vectorizer.transform([cleaned_text])
    proba = model.predict_proba(vec)[0]
    label = model.predict(vec)[0]
    label_index = np.where(model.classes_ == label)[0][0]
    result = "🧠 *Prediction*: Fake" if label == 1 else "🧠 *Prediction*: Genuine"
    return f"{result}  \n✅ **Confidence**: {proba[label_index]:.2%}"

# 🖼 Safe image loader with OpenCV
def read_image_bytes(uploaded_file):
    image_bytes = uploaded_file.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

# 📈 Preprocess image for OCR
def preprocess_pil(img):
    img = img.convert("L")
    img = img.filter(ImageFilter.MedianFilter())
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(2)

# 🎛 Streamlit UI
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detector")

mode = st.radio("Choose input method:", ["📝 Text", "🖼 Image"])

if mode == "📝 Text":
    user_input = st.text_area("Enter news content:")
    if st.button("Predict"):
        result = predict_news(user_input)
        st.markdown(result)

elif mode == "🖼 Image":
    uploaded_image = st.file_uploader("Upload a news image", type=["png", "jpg", "jpeg"])
    if uploaded_image:
        try:
            img = read_image_bytes(uploaded_image)
            st.image(img, caption="Uploaded Image", use_container_width=True)
            pil_img = preprocess_pil(img)
            extracted_text = pytesseract.image_to_string(pil_img)

            # 🔍 Display raw OCR output
            st.code(extracted_text, language='text')
            st.text_area("Extracted Text", value=extracted_text, height=150)

            if st.button("Predict from Image"):
                result = predict_news(extracted_text)
                st.markdown(result)
        except Exception as e:
            st.error(f"Something went wrong processing the image: {e}")
