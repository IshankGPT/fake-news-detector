import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Ensure required downloads
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

# Main execution
if __name__ == "__main__":
    try:
        df = pd.read_csv('data/fake_or_real_news.csv')
    except FileNotFoundError:
        print("❌ 'fake_or_real_news.csv' not found. Please run explore_dataset.py first.")
        exit()

    print("🧹 Cleaning texts… this may take 1–2 minutes.")
    df['clean_text'] = df['text'].apply(clean_text)
    df.to_csv('data/fake_or_real_news_cleaned.csv', index=False)
    print("✅ Cleaned dataset saved to data/fake_or_real_news_cleaned.csv")
