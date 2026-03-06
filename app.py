import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ── Load model & metadata once at startup ─────────────────────────────────────
model    = joblib.load("model/irrigation_model.pkl")
with open("model/metadata.json") as f:
    META = json.load(f)
FEATURES = META["features"]
LABELS   = META["labels"]


@app.route("/")
def index():
    return render_template("index.html", model_name=META["model_name"])


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    try:
        values = [
            float(data["soil_moisture"]),
            float(data["et0"]),
            float(data["crop_coefficient"]),
            float(data["days_planted"]),
            float(data["temperature"]),
        ]
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    X = pd.DataFrame([values], columns=FEATURES)
    label = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0].tolist()

    return jsonify({
        "label"      : label,
        "prediction" : LABELS[str(label)],
        "prob_irrigate"    : round(proba[1], 4),
        "prob_no_irrigate" : round(proba[0], 4),
        "model_used" : META["model_name"],
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)
