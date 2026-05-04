# 🚨 SpamShield — SMS Spam Detection System

Real-time SMS threat detection system using TF-IDF + Logistic Regression, exposed via REST API and interactive web interface.

---

## 🎯 Problem

Spam and phishing SMS messages are a major security threat, often used for fraud, identity theft, and financial scams. This project demonstrates how classical NLP techniques can effectively detect such threats in real time.

---

## ✨ Features

- **Dataset**: SMS Spam Collection (5,574 messages, balanced ham/spam)
- **Model**: TF-IDF unigrams + bigrams with Logistic Regression
- **Serialization**: Pickle-based model persistence
- **REST API**: POST `/api/predict` for programmatic classification
- **Web UI**: Interactive interface with confidence meter and example messages
- **High Accuracy**: ~98%+ classification accuracy on test set

---

## 🛠️ Tech Stack

### 💻 Programming Language
<p>
  <img src="https://skillicons.dev/icons?i=python" />
</p>

### 🧠 Machine Learning
- scikit-learn  
- Pandas  
- NumPy  

### 🌐 Web Development
<p>
  <img src="https://skillicons.dev/icons?i=html" style="margin-right:10px;" />
  <img src="https://skillicons.dev/icons?i=css" style="margin-right:10px;" />
  <img src="https://skillicons.dev/icons?i=flask" style="margin-right:10px;" />
  <img src="https://skillicons.dev/icons?i=js" />
</p>

### ⚙️ Backend Framework
- Flask

---

## 🚀 Quick Start

```bash
git clone https://github.com/yourusername/spam-classifier.git
cd spam-classifier
pip install -r requirements.txt
python train.py
```

Then start the Flask server:

```bash
python app.py
```

Open your browser to `http://localhost:5000` and start classifying messages.

---

## 🔌 API Usage

**Request:**

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"message": "URGENT: Verify your account now at http://fake-bank.com"}'
```

**Response:**

```json
{
  "message": "URGENT: Verify your account now at http://fake-bank.com",
  "label": "spam",
  "confidence": 97.42,
  "is_spam": true
}
```

---

## 💻 CLI Usage

```bash
python app.py
```

### Example

→ You won a FREE iPhone!!!
🚨 SPAM (confidence: 98.7%)

---

## 🏗️ Architecture

User Input
     ↓
TF-IDF Vectorizer
     ↓
Logistic Regression Model
     ↓
Prediction (Spam/Ham + Confidence)
     ↓
Flask API / Web UI

---

## 📁 Project Structure

```
spam-classifier/
├── train.py
├── predict.py
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
├── data/
│   └── sms.tsv
├── model/
│   ├── model.pkl
│   └── vectorizer.pkl
└── templates/
│   └── index.html
└── static/
    └── favicon.ico

```

---

## 🧠 How It Works

1. **Preprocessing**: Messages mapped to numeric labels (ham=0, spam=1), stratified 80/20 train/test split
2. **Vectorization**: TF-IDF extracts unigram + bigram features (max 5000 features) from training messages only
3. **Model Training**: Logistic Regression (C=5, max_iter=1000) fitted on vectorized training data
4. **Prediction**: Input text vectorized using the same fitted vectorizer, then classified with confidence probability

---

## 📈 Model Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Ham   | 0.99      | 0.98   | 0.98     |
| Spam  | 0.97      | 0.98   | 0.97     |

**Overall Accuracy: 0.9809** (98.09% on test set)

---

## 🏆 Highlights

- End-to-end ML pipeline (data → training → deployment)
- High accuracy on real-world dataset
- Fast inference (<50ms)
- Clean REST API + UI integration
- Production-ready structure

---

## 🛠️ Possible Improvements

🚀 Add explainability (highlight spam words)
🚀 Try SVM / Naive Bayes comparisons
🚀 Add user authentication
🚀 Store prediction history

---

## 📜 Dataset

SMS Spam Collection from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) — 5,574 real SMS messages.

---
