import os
import urllib.request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# ── DATASET DOWNLOAD ──
data_path = "data/sms.tsv"
if not os.path.exists(data_path):
    print("📥 Downloading SMS Spam Collection dataset...")
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    os.makedirs("data", exist_ok=True)
    urllib.request.urlretrieve(url, data_path)
    print("✅ Dataset downloaded successfully!")
else:
    print("✅ Dataset already exists!")

# ── LOAD AND PREPROCESS ──
print("\n📊 Loading and preprocessing dataset...")
df = pd.read_csv(data_path, sep="\t", header=None, names=["label", "message"])
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

print(f"📋 Dataset shape: {df.shape}")
print("\n🔢 Label distribution:")
print(df["label"].value_counts())

# ── TRAIN/TEST SPLIT ──
print("\n📊 Splitting dataset (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label"], 
    test_size=0.2, random_state=42, stratify=df["label"]
)

# ── VECTORIZATION ──
print("\n📥 Vectorizing text data...")
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000,
    ngram_range=(1, 2)
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print(f"✅ Vectorization complete! Vector shape: {X_train_vec.shape}")

# ── MODEL TRAINING ──
print("\n🚀 Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000, C=5)
model.fit(X_train_vec, y_train)
print("✅ Model trained successfully!")

# ── EVALUATION ──
print("\n📊 Model Evaluation:")
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy:.4f}")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

print("\n🔢 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ── SAVE MODEL AND VECTORIZER ──
print("\n💾 Saving model and vectorizer...")
os.makedirs("model", exist_ok=True)

with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)
print("✅ Model saved to model/model.pkl")

with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
print("✅ Vectorizer saved to model/vectorizer.pkl")

print("\n🚀 Training pipeline complete!")
