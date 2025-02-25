from fastapi import FastAPI, HTTPException  # Import HTTPException here
from fastapi.middleware.cors import CORSMiddleware
import joblib
from pydantic import BaseModel
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os

app = FastAPI()

# Allow all origins for development (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and vectorizer
model = joblib.load("../ml_model/model.pkl")
vectorizer = joblib.load("../ml_model/vectorizer.pkl")

class NewsInput(BaseModel):
    text: str

@app.post("/predict/")
def predict_category(news: NewsInput):
    try:
        # Vectorize the input text
        text_vectorized = vectorizer.transform([news.text])
        prediction = model.predict(text_vectorized)[0]
        return {"category": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/evaluate/")
def evaluate_model():
    try:
        # Get the absolute path of the dataset file
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../ml_model/dataset/News_Category_Dataset_v3.json")

        # Check if the dataset exists
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found at {dataset_path}")

        # Load the dataset
        df = pd.read_json(dataset_path, lines=True)

        # Validate dataset columns
        required_columns = ["headline", "category"]
        if not all(column in df.columns for column in required_columns):
            raise ValueError(f"Dataset must contain the following columns: {required_columns}")

        # Prepare the dataset
        X = df["headline"]
        y = df["category"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_test_tfidf = vectorizer.transform(X_test)

        # Predict and calculate accuracy
        y_pred = model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        return {"accuracy": accuracy}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "Welcome to the News Classification API"}