from flask import Flask, render_template, request
import pickle
import re
import string

app = Flask(__name__)

# Load vectorizer and model
with open("saved_model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("saved_model/model.pkl", "rb") as f:
    model = pickle.load(f)

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(rf"[{string.punctuation}]", "", text)
    text = text.strip()
    return text

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    input_text = ""

    if request.method == "POST":
        input_text = request.form.get("news_text", "").strip()

        if input_text == "":
            prediction = "⚠️ Please enter some text."
        else:
            cleaned = clean_text(input_text)
            X = vectorizer.transform([cleaned])

            pred = model.predict(X)[0]  # prediction (label)
            proba = model.predict_proba(X)[0]  # probability array

            # Convert pred to index
            if hasattr(model, 'classes_'):
                try:
                    label_index = list(model.classes_).index(pred)
                    prob = proba[label_index]
                except ValueError:
                    prob = max(proba)
            else:
                prob = max(proba)

            # Display label
            if str(pred).upper() == "FAKE" or int(pred) == 1:
                label_display = "🔴 FAKE NEWS"
            else:
                label_display = "🟢 REAL NEWS"

            prediction = f"📰 Prediction: {label_display}"
            confidence = f"📊 Confidence: {round(prob * 100, 2)}%"

    return render_template("index.html", prediction=prediction, confidence=confidence, input_text=input_text)

if __name__ == "__main__":
    app.run(debug=True)
