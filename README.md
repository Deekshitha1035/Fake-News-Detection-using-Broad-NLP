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
│
├── fake-news-bert/             
│   ├── app.py                  
│   ├── bert_train.py           
│   ├── save_model.py           
│   ├── merge_datasets.py       
│   ├── bert_fake_news_model/   
│       ├── config.json
│       ├── tokenizer_config.json
│       ├── vocab.txt
│
├── predict_bert.py             
├── requirements.txt            
├── README.md                   
└── .gitignore                  


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

## 🔹 Tech Stack

- **Language:** Python 3.8+
- **Deep Learning & NLP:** BERT, Hugging Face Transformers, PyTorch
- **Data Processing & Analysis:** Pandas, NumPy, scikit-learn, NLTK
- **Web/App (Optional):** Flask or Streamlit
- **Model Storage:** `safetensors`, Pickle (`.pkl`)
- **Version Control:** Git & GitHub


## 🔹 Screenshots

### Web/App Interface
![App Screenshot](<img width="918" height="558" alt="f1" src="https://github.com/user-attachments/assets/eedfa2de-57f9-47a6-9b44-b34ba4bcff6c" />)


### Sample Prediction
![Prediction Screenshot](<img width="925" height="557" alt="f3" src="https://github.com/user-attachments/assets/e5a6c24e-c5f3-45c3-907a-0681fa530f1e" />

<img width="925" height="547" alt="f4" src="https://github.com/user-attachments/assets/d56ff06a-8b1e-465c-8034-ffcb999c87fc" />


<img width="967" height="558" alt="f2" src="https://github.com/user-attachments/assets/9c443cf7-bf39-4d19-b0c4-73c43aa1d273" />

)




## 🔹 Installation

1. Clone the repo:
```bash
git clone https://github.com/Deekshitha1035/Fake-News-Detection-using-Broad-NLP.git
cd Fake-News-Detection-using-Broad-NLP
