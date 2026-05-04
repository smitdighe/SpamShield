import pickle
from flask import Flask, render_template, request, jsonify

# ── INITIALIZE FLASK APP ──
app = Flask(__name__, template_folder="templates")

# ── LOAD MODEL AND VECTORIZER ──
try:
    with open("model/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    print("✅ Model loaded successfully.")
except FileNotFoundError:
    raise FileNotFoundError("Model not found. Run python train.py first.")

# ── PREDICTION HELPER ──
def predict_message(text: str) -> dict:
    X_vec = vectorizer.transform([text])
    
    pred = model.predict(X_vec)[0]
    probs = model.predict_proba(X_vec)[0]
    
    confidence = max(probs)  # FIXED
    
    label = pred
    is_spam = label == "spam"
    
    return {
        "label": label,
        "confidence": round(confidence * 100, 2),
        "is_spam": is_spam
    }

# ── ROUTES ──
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    POST /api/predict
    Body: { "message": "string" }
    Returns: { "message": "string", "label": "spam"/"ham", "confidence": float, "is_spam": bool }
    """
    data = request.get_json(silent=True)
    
    if data is None or "message" not in data:
        return jsonify({"error": "Missing 'message' field in JSON body."}), 400
    
    message = data["message"].strip()
    
    if not message:
        return jsonify({"error": "Message cannot be empty."}), 400
    
    result = predict_message(message)
    
    return jsonify({
        "message": data["message"],
        "label": result["label"],
        "confidence": result["confidence"],
        "is_spam": result["is_spam"]
    }), 200

# ── RUN APP ──
if __name__ == "__main__":
    app.run(debug=True, port=5000)
