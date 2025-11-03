from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Load model
tokenizer = DistilBertTokenizer.from_pretrained("models/bert_email_classifier")
model = DistilBertForSequenceClassification.from_pretrained("models/bert_email_classifier")

def predict_email(text):
    inputs = tokenizer(text, truncation=True, padding="max_length", max_length=256, return_tensors="pt")
    outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    return "Spam" if pred == 1 else "Ham"

# Test
emails = [
    "Congratulations! You won a free iPhone.",
    "Hey, can we meet tomorrow for lunch?"
]

for email in emails:
    print(email, "=>", predict_email(email))
