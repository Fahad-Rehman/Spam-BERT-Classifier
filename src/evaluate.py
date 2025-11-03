import joblib
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("data/spam_Emails_data.csv")
df = df[df['label'].isin(['Ham', 'Spam']) & df['text'].notna()]

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# Load baseline model
pipeline = joblib.load("models/baseline_spam.joblib")
y_pred = pipeline.predict(X_test)

print(classification_report(y_test, y_pred))
