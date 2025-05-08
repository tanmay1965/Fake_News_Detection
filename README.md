# 📰 Fake News Detection Model

This project implements a **Fake News Detection** model using Natural Language Processing (NLP) and machine learning. It utilizes the **LIAR dataset** and compares the performance of Logistic Regression and Random Forest classifiers.

---

## 📌 Overview

- **Dataset**: [LIAR Dataset](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip)  
- **Tech Stack**: Python, Scikit-learn, NLTK, TF-IDF  
- **Goal**: Classify political statements as *True (1)* or *Fake (0)*  
- **Achieved F1-score**: `0.86` (Logistic Regression)

---

## 📁 Project Structure

```
├── fake_news_detection.py      # Main script
├── tfidf_vectorizer.pkl        # Saved TF-IDF vectorizer
├── logistic_regression_model.pkl  # Trained Logistic Regression model
├── random_forest_model.pkl     # Trained Random Forest model
├── liar_dataset/               # Contains train.tsv
└── README.md                   # This file
```

---

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fake-news-detector.git
   cd fake-news-detector
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Download NLTK resources:
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

---

## 📊 Model Performance

Two models were trained and evaluated:

| Model              | Weighted F1-score |
|--------------------|-------------------|
| Logistic Regression| 0.86              |
| Random Forest      | *depends on tuning, ~0.82*  |

---

## 🚀 How to Use

### Predict from Custom Input

```python
from fake_news_detection import predict_fake_news
import joblib

# Load models
model = joblib.load('logistic_regression_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Predict
text = "The economy has grown faster under this government."
result = predict_fake_news(text, model, vectorizer)
print("Prediction:", "True" if result == 1 else "Fake")
```

---

## 🧹 Preprocessing Details

- Lowercasing
- Removing punctuation and digits
- Tokenizing and lemmatizing
- Removing English stopwords
- TF-IDF vectorization (uni- and bi-grams, top 5000 features)

---

## 📌 Notes

- Labels were simplified:
  - `true`, `mostly-true` → `1` (True)
  - `half-true`, `barely-true`, `false`, `pants-fire` → `0` (Fake)
- The classifier is intended for educational use only and not for production deployment.

---

## 📄 License

MIT License

---

## 🙌 Acknowledgements

- [LIAR Dataset](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip) by William Yang Wang
- Scikit-learn, NLTK

---
