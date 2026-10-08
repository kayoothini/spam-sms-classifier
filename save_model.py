import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# 1. Load cleaned data
df = pd.read_csv('cleaned_spam.csv')
df = df.dropna()
df['clean_message'] = df['clean_message'].astype(str)
df = df[df['clean_message'].str.strip() != '']

# 2. Features and Target
X = df['clean_message']
y = df['label'].map({'ham': 0, 'spam': 1})

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=3000)
X_train_vec = vectorizer.fit_transform(X_train)

# 5. Train Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 6. Save Model + Vectorizer as .pkl files
pickle.dump(model, open('spam_model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

print("✅ Model saved as spam_model.pkl")
print("✅ Vectorizer saved as vectorizer.pkl")

# 7. Test with custom messages
print("\n=== Testing Custom Messages ===\n")

test_messages = [
    "Free entry in 2 a wkly comp to win FA Cup final tkts",
    "Hey, are you coming to the party tonight?",
    "WINNER! You have won £1000 cash! Call now to claim",
    "Ok lar... Joking wif u oni...",
    "URGENT! Your mobile number has won a £2000 prize",
    "See you tomorrow at 5pm",
    "Congratulations! You've been selected for a free gift",
    "Don't forget to bring your books"
]

for msg in test_messages:
    # Clean pannaama direct-a test panrom (simple-a)
    vec = vectorizer.transform([msg.lower()])
    prediction = model.predict(vec)[0]
    result = "🚨 SPAM" if prediction == 1 else "✅ HAM"
    print(f"{result} → {msg}")