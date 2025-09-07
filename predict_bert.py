from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load your saved model and tokenizer
model_path = "D:/fake-news-bert/bert_fake_news_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        return "Genuine" if predicted_class == 1 else "Fake"


#  Batch test samples
samples = [
    "Scientists confirm that eating mangoes cures cancer.",
    "The government announced a new tax reform today.",
    "Aliens have been spotted at an IPL match."
]

for text in samples:
    print(f"Input: {text}")
    print(f"Prediction: {predict(text)}")
    print()
