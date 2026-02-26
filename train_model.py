import pandas as pd
import re
import nltk
import joblib

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

nltk.download('stopwords')

# ------------------------
# LOAD SMALLER SAMPLE
# ------------------------
data = pd.read_csv("AI_Human.csv", nrows=20000)

print("Loaded rows:", len(data))

# Rename column
data.rename(columns={'generated': 'label'}, inplace=True)

# ------------------------
# CLEAN TEXT (FASTER VERSION)
# ------------------------
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

data['text'] = data['text'].apply(preprocess)

X = data['text']
y = data['label']

# ------------------------
# SPLIT
# ------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------
# TF-IDF (Reduced features)
# ------------------------
vectorizer = TfidfVectorizer(max_features=3000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ------------------------
# MODEL (Probability support)
# ------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# ------------------------
# ACCURACY
# ------------------------
y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))

# ------------------------
# SAVE
# ------------------------
joblib.dump(model, "ppt_ai_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model saved successfully!")