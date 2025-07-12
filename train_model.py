import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Dummy example data
comments = [
    "I love this product!",
    "I hate this movie.",
    "This is okay.",
    "You are an idiot!",
    "What a beautiful day!",
    "I will kill you",
    "You failed the exam",
    "Hi Guys!",
]
labels = [
    "Positive",
    "Negative",
    "Neutral",
    "Risky",
    "Positive",
    "Risky",
    "Negative",
    "Neutral"
]

# Vectorize the text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(comments)

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model, vectorizer, and encoder
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print("✅ All models saved successfully.")
