import pandas as pd

try:
    # Load CSV files
    fake_df = pd.read_csv('data/Fake.csv')
    real_df = pd.read_csv('data/True.csv')
except FileNotFoundError:
    print("❌ One or both source CSV files not found. Make sure 'Fake.csv' and 'True.csv' exist in the 'data/' folder.")
    exit()

# Preview the data
print("🔹 Fake News Sample:\n", fake_df.head(), "\n")
print("🔹 Real News Sample:\n", real_df.head(), "\n")

# Add labels
fake_df['label'] = 'FAKE'
real_df['label'] = 'REAL'

# Combine datasets
df = pd.concat([fake_df, real_df], ignore_index=True)

# Dataset info
print("✅ Combined Dataset Shape:", df.shape)
print("\n🔎 Missing Values:\n", df.isnull().sum())
print("\n📊 Label Distribution:\n", df['label'].value_counts())

# Save combined dataset
df.to_csv('data/fake_or_real_news.csv', index=False)
print("\n✅ Combined dataset saved as data/fake_or_real_news.csv")
