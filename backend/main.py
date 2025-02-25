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

@app.get("/evaluate/")
def evaluate_model():
    try:
        # Define the dataset path
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../ml_model/dataset/News_Category_Dataset_v3.json")
        logger.info(f"Dataset path: {dataset_path}")
        
        # Check if the dataset already exists
        if not os.path.exists(dataset_path):
            try:
                # Download the dataset from Kaggle
                logger.info("Dataset not found, downloading from Kaggle...")
                kaggle.api.dataset_download_files('rmisra/news-category-dataset', path='../ml_model/dataset', unzip=True)
                logger.info("Dataset downloaded successfully")
                
                # Log the contents of the directory
                dataset_dir = os.path.dirname(dataset_path)
                logger.info(f"Contents of dataset directory: {os.listdir(dataset_dir)}")
                
                # Check if the dataset file exists after download
                if not os.path.exists(dataset_path):
                    logger.error(f"Dataset file not found after download: {dataset_path}")
                    return {"error": "Dataset file not found after download"}
                
            except Exception as e:
                logger.error(f"Error downloading dataset from Kaggle: {e}")
                return {"error": "Unable to download dataset from Kaggle"}
        
        # Load the test data
        df = pd.read_json(dataset_path, lines=True)
        logger.info("Dataset loaded successfully")
        
        X = df['headline']
        y = df['category']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_test_tfidf = vectorizer.transform(X_test)

        # Predict and calculate accuracy
        y_pred = model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model accuracy: {accuracy}")
        
        # Return the accuracy in the response
        return {"accuracy": accuracy}
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        return {"error": "Unable to fetch accuracy"}