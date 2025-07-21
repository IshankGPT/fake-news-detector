# 📰 Fake News Detection App

A machine learning-based web application that classifies news statements as Real or Fake.

Built with:
- 🐍 Python
- 🤖 Scikit-learn
- 🌐 Flask
- 🧹 Custom text preprocessing
- 🖥️ HTML/CSS for frontend

---

## 🚀 Features

- Logistic Regression classifier trained on TF-IDF vectorized text
- Cleaned text preprocessing
- Web interface using Flask
- CLI tool for quick testing in terminal
- Prediction confidence score
- Logging of user inputs and predictions

---

## 📁 Project Structure

```

llm-fake-news-detector/
│
├── predict_web.py           # Main Flask application
├── predict_cli.py          # Command-line interface for predictions
├── train_model.py          # Script to train and save model/vectorizer
├── clean_text.py           # Text cleaning utility
├── requirements.txt        # Python dependencies
├── logs.csv                # Optional: stores user inputs & results
│
├── saved_model/
│   ├── model.pkl           # Trained ML model
│   └── tfidf_vectorizer.pkl  
│   └── train_test_split.pkl  
│   └── vectorizer.pkl      # TF-IDF vectorizer
├── templates/
│   └── index.html          # HTML for web app frontend
│
└── README.md               # Project documentation

````

---

## 🧩 Installation

Clone this repo and install dependencies:

```bash
git clone https://github.com/your-username/fake-news-detector.git
cd fake-news-detector
pip install -r requirements.txt
````

(Optional) Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

---

## 🧠 Train the Model

Train a Logistic Regression model and save artifacts:

```bash
python train_model.py
```

This creates:

* saved\_model/model.pkl
* saved\_model/vectorizer.pkl

---

## 🌐 Run the Web App

Start the Flask app locally:

```bash
python predict_web.py
```

Then go to:

```
http://127.0.0.1:5000
```

Enter a news sentence and click Predict.

---

## 🧪 Test via CLI

Use the terminal-based interface:

```bash
python predict_cli.py
```

Example:

```
🗞️  Enter news text: NASA discovers water on the moon.

🔎 Prediction: REAL
📊 Confidence: 92.13%
```

---

## 🔍 Sample Test Case (Web)

Input:

> The WHO issues new guidelines on COVID-19 vaccination for children under 12.

Output:

> 📰 Prediction: 🔴 FAKE NEWS
> 📊 Confidence: 94.25%


---

Made with ❤️ by Ishank

