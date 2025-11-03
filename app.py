# app/app.py
import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import os
import torch.nn.functional as F

st.set_page_config(page_title="Email Spam Classifier", layout="wide")

# Absolute path to your BERT model
MODEL_DIR = os.path.join(os.getcwd(), "models", "bert_email_classifier")

# Load model and tokenizer
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_DIR):
        st.error(f"Model folder not found at {MODEL_DIR}")
        st.stop()
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    return tokenizer, model

tokenizer, model = load_model()

# Prediction function with probability
def predict_email(text, threshold=0.5):
    inputs = tokenizer(text, truncation=True, padding="max_length", max_length=256, return_tensors="pt")
    outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)
    spam_prob = probs[0][1].item()
    if spam_prob > threshold:
        return "Spam", spam_prob
    else:
        return "Ham", 1 - spam_prob

# Streamlit UI
st.title("Email Spam Classifier")
st.write("Classify emails as Ham or Spam. Enter your email message below:")

email_text = st.text_area("Email Text:", height=200)

# Prediction button
if st.button("Predict"):
    if email_text.strip() == "":
        st.warning("Please enter email text!")
    else:
        prediction, probability = predict_email(email_text)
        st.success(f"Prediction: {prediction}")
        st.info(f"Confidence: {probability:.2f}")
