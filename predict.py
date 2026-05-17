import os
import sys
import pickle

from sqlalchemy import label

# ── LOAD MODEL AND VECTORIZER ──
if not os.path.exists("model/model.pkl") or not os.path.exists("model/vectorizer.pkl"):
    print("Model not found. Run python train.py first.")
    sys.exit(1)

with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

print("✅ Model ready. Type a message to classify (or 'quit' to exit).")

# ── PREDICTION FUNCTION ──
def predict_message(text):
    X_vec = vectorizer.transform([text])
    pred = model.predict(X_vec)[0]
    probs = model.predict_proba(X_vec)[0]
    confidence = max(probs)

    label = pred
    is_spam = label == "spam"
    
    return {
        "label": label,
        "confidence": round(confidence * 100, 2),
        "is_spam": is_spam
    }

# ── INTERACTIVE LOOP ──
if __name__ == "__main__":
    while True:
        user_input = input("→ ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ["quit", "exit"]:
            break
        
        result = predict_message(user_input)
        
        if result["is_spam"]:
            print(f"🚨 SPAM  (confidence: {result['confidence']}%)")
        else:
            print(f"✅ HAM   (confidence: {result['confidence']}%)")
