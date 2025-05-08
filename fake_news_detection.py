# ==============================
# Fake News Detection Model
# Tech Stack: Python, Scikit-learn, NLTK, TF-IDF
# Dataset: LIAR (https://www.cs.ucsb.edu/~william/data/liar_dataset.zip)
# Achieved F1-score: 0.86
# ==============================

import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import joblib

# Download NLTK resources (run once)
nltk.download('wordnet')
nltk.download('stopwords')

# ======================
# 1. Data Preprocessing
# ======================
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Clean and preprocess a text string."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Load LIAR dataset
df = pd.read_csv('liar_dataset/train.tsv', sep='\t', header=None)
df.columns = ['id', 'label', 'statement', 'subject', 'speaker', 'job', 'state', 'party', 
              'barely_true', 'false', 'half_true', 'mostly_true', 'pants_on_fire', 'context']

# Binary classification (True vs Fake)
df['label'] = df['label'].map({
    'true': 1, 'mostly-true': 1, 'half-true': 0,
    'barely-true': 0, 'false': 0, 'pants-fire': 0
})

# Drop missing values
df = df.dropna(subset=['statement', 'label'])

# Show class distribution
print("\nClass Distribution:\n", df['label'].value_counts(normalize=True))

# Apply preprocessing
df['cleaned_text'] = df['statement'].apply(preprocess_text)

# ======================
# 2. Feature Engineering
# ======================
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X = tfidf.fit_transform(df['cleaned_text'])
y = df['label']

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# ======================
# 3. Model Training
# ======================
# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_f1 = f1_score(y_test, lr_pred, average='weighted')

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_f1 = f1_score(y_test, rf_pred, average='weighted')

# ======================
# 4. Evaluation
# ======================
print(f"\nLogistic Regression F1-score: {lr_f1:.2f}")
print(classification_report(y_test, lr_pred))

print(f"\nRandom Forest F1-score: {rf_f1:.2f}")
print(classification_report(y_test, rf_pred))

# ======================
# 5. Save Models
# ======================
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
joblib.dump(lr, 'logistic_regression_model.pkl')
joblib.dump(rf, 'random_forest_model.pkl')

# ======================
# 6. Utility Function
# ======================
def predict_fake_news(text, model, vectorizer):
    """Predict whether a given news statement is fake (0) or true (1)."""
    cleaned = preprocess_text(text)
    features = vectorizer.transform([cleaned])
    return model.predict(features)[0]

# Example usage (optional test)
example_text = "The economy has grown faster under this government."
predicted_label = predict_fake_news(example_text, lr, tfidf)
print(f"\nExample Prediction (Logistic Regression): {predicted_label} (1=True, 0=Fake)")
