from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
from pydantic import BaseModel
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
import kaggle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Helper function to download dataset from Kaggle
def download_dataset():
    try:
        logger.info("Dataset not found locally. Downloading from Kaggle...")
        kaggle.api.dataset_download_files('rmisra/news-category-dataset', path='../ml_model/dataset', unzip=True)
        logger.info("Dataset downloaded successfully")
    except Exception as e:
        logger.error(f"Error downloading dataset from Kaggle: {e}")
        return False
    return True

# Helper function to calculate model accuracy
def calculate_accuracy():
    try:
        # Define the dataset path
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../ml_model/dataset/News_Category_Dataset_v3.json")
        logger.info(f"Dataset path: {dataset_path}")
        
        # Check if the dataset exists, download if not
        if not os.path.exists(dataset_path):
            if not download_dataset():
                return None

        # Load dataset
        logger.info("Loading dataset for accuracy calculation...")
        df = pd.read_json(dataset_path, lines=True)
        X = df['headline']
        y = df['category']
        
        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Transform the text data using the vectorizer
        X_test_tfidf = vectorizer.transform(X_test)
        
        # Predict using the model
        y_pred = model.predict(X_test_tfidf)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model accuracy: {accuracy:.2f}")
        
        return accuracy
    except Exception as e:
        logger.error(f"Error calculating accuracy: {e}")
        return None

@app.get("/evaluate/")
def evaluate_model():
    accuracy = calculate_accuracy()
    if accuracy is not None:
        return {"accuracy": accuracy}
    else:
        return {"error": "Unable to fetch accuracy"}
