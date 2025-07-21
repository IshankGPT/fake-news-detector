import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

# Load the cleaned dataset
df = pd.read_csv("data/fake_or_real_news_cleaned.csv")

# Convert labels to binary: 0 = REAL, 1 = FAKE
df['label'] = df['label'].map({'REAL': 0, 'FAKE': 1})

# Split data
X = df['clean_text']
y = df['label']
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

# Train model
model = LogisticRegression(max_iter=1000)
print("⏳ Training the model...")
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("\n✅ Accuracy:", round(acc * 100, 2), "%")
print("\n🔎 Classification Report:\n", classification_report(y_test, y_pred))
print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model, vectorizer, and test data
os.makedirs('saved_model', exist_ok=True)
with open('saved_model/model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('saved_model/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
with open('saved_model/train_test_split.pkl', 'wb') as f:
    pickle.dump((X_train, X_test, y_train, y_test), f)

print("\n💾 Model, vectorizer, and test split saved in 'saved_model/'")
