# SpamShield — SMS Spam Classifier

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange?style=flat-square)
![Flask](https://img.shields.io/badge/Flask-3.0-black?style=flat-square)
![Accuracy](https://img.shields.io/badge/Accuracy-~98%25+-brightgreen?style=flat-square)

Real-time SMS spam detection with TF-IDF vectorization and logistic regression, served via Flask REST API and web UI.

## ✨ Features

- **Dataset**: SMS Spam Collection (5,574 messages, balanced ham/spam)
- **Model**: TF-IDF unigrams + bigrams with Logistic Regression
- **Serialization**: Pickle-based model persistence
- **REST API**: POST `/api/predict` for programmatic classification
- **Web UI**: Interactive interface with confidence meter and example messages
- **High Accuracy**: ~98%+ classification accuracy on test set

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

## 📁 Project Structure

```
spam-classifier/
├── train.py              # Model training pipeline (downloads dataset,trains, saves model)
├── predict.py            # CLI interface for spam classification (interactive)
├── app.py                # Flask web server with REST API endpoint
├── requirements.txt      # Pinned dependencies (Flask, scikit-learn, pandas, numpy)
├── .gitignore           # Git ignore rules (excludes model/, data/, __pycache__)
├── README.md            # This file
├── data/                # Dataset storage (git-ignored)
├── model/               # Serialized model artifacts (git-ignored)
│   ├── model.pkl        # Trained Logistic Regression model
│   └── vectorizer.pkl   # Fitted TF-IDF vectorizer
└── templates/           # Flask HTML templates
    └── index.html       # Web UI (dark theme, vanilla JS)
```

## 🧠 How It Works

1. **Preprocessing**: Messages mapped to numeric labels (ham=0, spam=1), stratified 80/20 train/test split
2. **Vectorization**: TF-IDF extracts unigram + bigram features (max 5000 features) from training messages only
3. **Model Training**: Logistic Regression (C=5, max_iter=1000) fitted on vectorized training data
4. **Prediction**: Input text vectorized using the same fitted vectorizer, then classified with confidence probability

## 📈 Model Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Ham   | 0.99      | 0.98   | 0.98     |
| Spam  | 0.97      | 0.98   | 0.97     |

**Overall Accuracy: 0.9809** (98.09% on test set)

## 🛠️ Possible Improvements

- [ ] Experiment with Naive Bayes and SVM classifiers for comparison
- [ ] Add multilingual support (translate before classification)
- [ ] Deploy to Render, Railway, or HuggingFace Spaces for public access
- [ ] Add message history and statistics to web UI
- [ ] Dockerize for containerized deployment

## 📜 Dataset

SMS Spam Collection from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) — 5,574 real SMS messages.

---
