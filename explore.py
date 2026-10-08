import pandas as pd

# Load dataset
df = pd.read_csv('spam.csv', encoding='latin-1')

# First 5 rows paakka
print("=== First 5 rows ===")
print(df.head())

# Shape (rows, columns)
print("\n=== Shape ===")
print(df.shape)

# Column names
print("\n=== Columns ===")
print(df.columns.tolist())

# Useful columns mattum eduthukkalaam
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

print("\n=== After renaming ===")
print(df.head())

# Spam vs Ham count
print("\n=== Label Count ===")
print(df['label'].value_counts())