from flask import Flask, request, render_template
from pptx import Presentation
import joblib
import re
import nltk
from nltk.corpus import stopwords
import os

nltk.download('stopwords')

app = Flask(__name__)

# -------------------------
# LOAD MODEL + VECTORIZER
# -------------------------
model = joblib.load("ppt_ai_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

stop_words = set(stopwords.words('english'))

# -------------------------
# PREPROCESS FUNCTION
# -------------------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# -------------------------
# EXTRACT TEXT FROM PPT
# -------------------------
def extract_text_from_ppt(file_path):
    presentation = Presentation(file_path)
    text = ""
    for slide in presentation.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text += shape.text + " "
    return text

# -------------------------
# HOME ROUTE
# -------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -------------------------
# PREDICTION ROUTE
# -------------------------
@app.route("/predict_ppt", methods=["POST"])
def predict_ppt():
    file = request.files["file"]
    file_path = "temp.pptx"
    file.save(file_path)

    text = extract_text_from_ppt(file_path)
    processed_text = preprocess(text)

    # ✅ VECTORIZE BEFORE PREDICTION
    vector_text = vectorizer.transform([processed_text])

    prediction = model.predict(vector_text)[0]
    probability = model.predict_proba(vector_text)[0][1]

    os.remove(file_path)

    result = "AI Generated PPT" if prediction == 1 else "Human Written PPT"
    probability = round(float(probability) * 100, 2)

    return render_template("result.html", result=result, probability=probability)

# -------------------------
if __name__ == "__main__":
    app.run(debug=True)