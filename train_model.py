import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load cleaned data
df = pd.read_csv('cleaned_spam.csv')

# NaN and empty remove
df = df.dropna()
df['clean_message'] = df['clean_message'].astype(str)
df = df[df['clean_message'].str.strip() != '']

print("=== Data Loaded ===")
print(f"Total messages: {len(df)}")

# 2. Features and Target
X = df['clean_message']
y = df['label'].map({'ham': 0, 'spam': 1})

print(f"Ham: {(y == 0).sum()}, Spam: {(y == 1).sum()}")

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {len(X_train)} messages")
print(f"Testing set : {len(X_test)} messages")

# 4. TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=3000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"\nVector shape: {X_train_vec.shape}")

# 5. Train Naive Bayes Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

print("\n✅ Model trained!")

# 6. Predict
y_pred = model.predict(X_test_vec)

# 7. Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"\n=== Accuracy: {accuracy * 100:.2f}% ===")

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))