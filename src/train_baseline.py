# src/train_baseline.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
import os

# Load data
df = pd.read_csv("data/spam_Emails_data.csv")
df = df[df['label'].isin(['Ham', 'Spam']) & df['text'].notna()]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# Pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
    ('clf', LogisticRegression(max_iter=200))
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
os.makedirs("../models", exist_ok=True)
joblib.dump(pipeline, "../models/baseline_spam.joblib")
print("Baseline model saved successfully.")
