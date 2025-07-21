import sys
import pickle
import re
import string
import numpy as np

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"\d+", "", text)      # remove numbers
    text = re.sub(rf"[{string.punctuation}]", "", text)  # remove punctuation
    text = text.strip()
    return text

def main():
    # Step 1: Check input
    if len(sys.argv) < 2:
        print("⚠️ Please provide a news text.")
        print("👉 Usage: python predict_cli.py \"<your news text>\"")
        sys.exit()

    input_text = sys.argv[1]

    # Step 2: Clean the text
    cleaned = clean_text(input_text)

    # Step 3: Load vectorizer and model
    with open("saved_model/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    with open("saved_model/model.pkl", "rb") as f:
        model = pickle.load(f)

    # Step 4: Vectorize
    X = vectorizer.transform([cleaned])

    # Step 5: Predict
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][pred]
    label = "FAKE" if pred == 1 else "REAL"

    # Step 6: Output
    print(f"\n📰 Prediction: {label}")
    print(f"📊 Confidence: {round(prob * 100, 2)}%")

if __name__ == "__main__":
    main()
