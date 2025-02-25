from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
from pydantic import BaseModel
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os




# Load model and vectorizer
model = joblib.load("../ml_model/model.pkl")
vectorizer = joblib.load("../ml_model/vectorizer.pkl")

app = FastAPI()

# Allow all origins for development (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class NewsInput(BaseModel):
    text: str

@app.post("/predict/")
def predict_category(news: NewsInput):
    text_vectorized = vectorizer.transform([news.text])
    prediction = model.predict(text_vectorized)[0]
    return {"category": prediction}


@app.get("/evaluate/")
def evaluate_model():
    # Get the absolute path of the dataset file
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../ml_model/dataset/News_Category_Dataset_v3.json")
    # Load the test data
    df = pd.read_json(dataset_path, lines=True)
    X = df['headline']
    y = df['category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test_tfidf = vectorizer.transform(X_test)

    # Predict and calculate accuracy
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    return {"accuracy": accuracy}