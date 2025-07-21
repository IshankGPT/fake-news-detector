import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle
import os

# ✅ Load the cleaned data
df = pd.read_csv("data/fake_or_real_news_cleaned.csv")

# ✅ Replace any NaNs just in case
df['clean_text'] = df['clean_text'].fillna("")

# ✅ Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

# ✅ Fit and transform the clean_text column
X = vectorizer.fit_transform(df['clean_text'])

# ✅ Create train-test split
X_train, X_test, y_train, y_test = train_test_split(X, df['label'], test_size=0.2, random_state=42)

# ✅ Make sure 'saved_model/' folder exists
os.makedirs("saved_model", exist_ok=True)

# ✅ Save train-test split
with open("saved_model/train_test_split.pkl", "wb") as f:
    pickle.dump((X_train, X_test, y_train, y_test), f)

# ✅ Save the TF-IDF vectorizer
with open("saved_model/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ TF-IDF features created and saved in 'saved_model/' folder.")
