# Fake News Detection using BERT / NLP

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT)

---

## 🔹 Overview
A **Fake News Detection system** built using **BERT** and NLP techniques to classify news as **fake** or **real**.  
The project includes data preprocessing, model training, and prediction scripts.

---

## 🔹 Features
- Train a BERT-based classifier
- Preprocess and merge multiple datasets
- Predict fake/real news from new text
- Lightweight repo (datasets and models stored externally)

---

## 🔹 Project Structure

fake-news-detection/
│── fake-news-bert/
│ ├── app.py
│ ├── bert_train.py
│ ├── train_model.py
│ ├── save_model.py
│ ├── merge_datasets.py
│ ├── bert_fake_news_model/
│ ├── config.json
│ ├── tokenizer_config.json
│ ├── vocab.txt
│── README.md
│── .gitignore


---

## 🔹 Datasets & Models
**Not included** in the repo due to size.  

Download and place them in `fake-news-bert/`:

| File | Description |
|------|------------|
| `Fake.csv` / `True.csv` | Original news datasets |
| `model.safetensors` | Trained BERT model |
| `vectorizer.pkl` | Saved vectorizer |

> Example link: [Google Drive / Hugging Face](#)

---

## 🔹 Installation

1. Clone the repo:
```bash
git clone https://github.com/Deekshitha1035/Fake-News-Detection-using-Broad-NLP.git
cd Fake-News-Detection-using-Broad-NLP
