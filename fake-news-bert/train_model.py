import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np


# 🚚 Load and label both datasets
fake_df = pd.read_csv(r"C:\Users\manog\Downloads\archive (1)\News _dataset\Fake.csv")
true_df = pd.read_csv(r"C:\Users\manog\Downloads\archive (1)\News _dataset\True.csv")

fake_df['label'] = 1  # Fake
true_df['label'] = 0  # Genuine

# 🧩 Combine and shuffle
data = pd.concat([fake_df, true_df], ignore_index=True)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# 📝 Combine title and text for better predictions
data['content'] = data['title'].astype(str) + " " + data['text'].astype(str)

# ✂ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    data['content'], data['label'], test_size=0.2, random_state=42
)

# 🔠 TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 🤖 Logistic Regression model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# 📊 Optional: Print accuracy for feedback
print(f"Training Accuracy: {model.score(X_train_vec, y_train):.2f}")
print(f"Test Accuracy: {model.score(X_test_vec, y_test):.2f}")
print(f"Model Classes: {model.classes_}")  # [0 1] → 0 = Genuine, 1 = Fake

# 💾 Save the model and vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model and vectorizer saved successfully.")
