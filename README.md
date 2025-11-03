# Email Spam Classifier

A simple **Email Spam Classifier** using **BERT (DistilBERT)** and **Streamlit**.  
It allows users to input **raw email text** and predicts whether the email is **Ham** (not spam) or **Spam**.

This project demonstrates building a machine learning pipeline from **data preprocessing**, **model training**, to **deployment** in a user-friendly web app.

---

## Features

- Classifies emails as **Ham** or **Spam** using a fine-tuned DistilBERT model.
- Shows **prediction confidence** to indicate how certain the model is.
- User-friendly **Streamlit interface**.
- Lightweight and easy to run on a local machine.

---

## Folder Structure

Here’s a compact overview of the project:
```yaml
EmailAutomation/
├─ app/ # Streamlit application
│ └─ app.py
├─ models/ # Saved BERT model and tokenizer
│ └─ bert_email_classifier/
├─ src/ # Training and inference scripts
│ ├─ train_bert.py
│ ├─ train_baseline.py
│ └─ infer_bert.py
├─ data/ # Dataset CSV
│ └─ spam_Emails_data.csv
├─ requirements.txt # Python dependencies
└─ README.md
```
### Dataset

The dataset is too large to include in this repository.  
Download it here: [spam_Emails_data.csv](https://www.kaggle.com/datasets/meruvulikith/190k-spam-ham-email-dataset-for-classification)  

After downloading, place the file in the `data/` folder before running the app or training scripts.

## Installation
### 1. Clone the repository

```bash
git clone <https://github.com/Fahad-Rehman/Spam-BERT-Classifier>
cd EmailAutomation
```
### 2. Create a virtual environment
```bash
python -m venv venv
```
### 3. Activate the environment
* Windows:
```bash
venv\Scripts\activate
```
* Mac/Linux
```bash
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Run the Streamlit App
```bash
streamlit run app/app.py
```
* Enter raw email text in the text area.
* Click Predict to see the classification (Ham/Spam) and confidence score.

### Run inference via Script
```bash
python src/infer_bert.py
```
* Modify the script to input your own email text for testing.

## Training the BERT Model
* Training is done using src/train_bert.py
* Make sure the dataset data/spam_Emails_data.csv exists.
* After training, the model and tokenizer are saved in models/bert_email_classifier.
Example snippet from training:

```bash
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
```

## Notes
* The model works best on clean English text.
* PDFs, images, or romanized/non-English text are not supported in this version.
* Confidence scores are provided to indicate the certainty of predictions.
* Streamlit app uses absolute paths for model loading.

## License
This project is open-source. You are free to use, modify, and share it for educational and non-commercial purposes.

