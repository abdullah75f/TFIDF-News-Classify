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


 #Evaluation endpoint
@app.get("/evaluate/")
def evaluate_model():
    try:
        # Check if model and vectorizer exist
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            raise FileNotFoundError("Model or vectorizer file is missing.")

        # Load the model and vectorizer
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)

        # Define the path to the dataset
        dataset_path = 'TFIDFNewsClassify/ml_model/dataset/News_Category_Dataset_v3.json'
        
        # Load the dataset
        df = pd.read_json(dataset_path, lines=True)
        X = df['headline']
        y = df['category']

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Transform the test set using the trained vectorizer
        X_test_tfidf = vectorizer.transform(X_test)

        # Predict and calculate accuracy
        y_pred = model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)

        return {"accuracy": accuracy}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {e}")

# Root route
@app.get("/")
def read_root():
    return {"message": "Welcome to the News Classification API"}
