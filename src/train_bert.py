import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score
import pandas as pd
import numpy as np


# Load dataset
df = pd.read_csv("data/spam_Emails_data.csv")
df = df.dropna()
df = df[df["label"].isin(["Ham", "Spam"])]

# Encode labels
df["label"] = df["label"].map({"Ham": 0, "Spam": 1})

# Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].tolist(),
    df["label"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df["label"].tolist()
)

# Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

train_dataset = EmailDataset(train_texts, train_labels, tokenizer)
val_dataset = EmailDataset(val_texts, val_labels, tokenizer)

# Model
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)

#Compatible TrainingArguments for Transformers 4.57.1
training_args = TrainingArguments(
    output_dir="models/bert_output",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_dir="logs",
    logging_steps=100,
    eval_strategy="epoch",       #Use this instead of 'evaluation_strategy'
    save_strategy="epoch",       #supported in 4.57.1
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none"
)

# Define evaluation function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds)
    }


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Train
trainer.train()

# Evaluate
preds = trainer.predict(val_dataset)
pred_labels = np.argmax(preds.predictions, axis=1)
print("\nClassification Report:\n")
print(classification_report(val_labels, pred_labels, target_names=["Ham", "Spam"]))

# Save model
model.save_pretrained("models/bert_email_classifier")
tokenizer.save_pretrained("models/bert_email_classifier")
print("\nModel and tokenizer saved successfully.")
