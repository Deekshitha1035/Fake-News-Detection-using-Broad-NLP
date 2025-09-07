import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load and prepare dataset
df_fake = pd.read_csv(r'C:\Users\manog\Downloads\archive (1)\News _dataset\Fake.csv')
df_true = pd.read_csv(r'C:\Users\manog\Downloads\archive (1)\News _dataset\True.csv')
df_fake['label'] = 'Fake'
df_true['label'] = 'Genuine'
df = pd.concat([df_fake, df_true], ignore_index=True)
df['text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
df['text'] = df['text'].str.lower().str.strip()

# Train model
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['text'])
y = df['label']
model = LogisticRegression()
model.fit(X, y)

# Save trained vectorizer and model
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model and vectorizer saved successfully.")
