from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
from pydantic import BaseModel
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
import kaggle
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

accuracy_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../ml_model/model_accuracy.txt")
logger.info(f"Checking accuracy file at: {accuracy_file}")
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
        # Define the accuracy file path
        accuracy_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../ml_model/accuracy.txt")
        
        # If the accuracy file exists, read and return the stored accuracy
        if os.path.exists(accuracy_file):
            with open(accuracy_file, "r") as file:
                accuracy = file.read().strip()  # Read and remove extra spaces/newlines
            return {"accuracy": float(accuracy)}  # Convert to float before returning
        
        # Define the dataset path
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../ml_model/dataset/News_Category_Dataset_v3.json")
        logger.info(f"Dataset path: {dataset_path}")
        
        # Load the test data
        logger.info("Loading dataset...")
        try:
            df = pd.read_json(dataset_path, lines=True)
            logger.info("Dataset loaded successfully")
        except ValueError as e:
            logger.error(f"Error loading dataset (line-delimited): {e}")
            return {"error": "Unable to load dataset"}
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return {"error": "Unable to load dataset"}
        
        X = df['headline']
        y = df['category']
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        except Exception as e:
            logger.error(f"Error splitting data: {e}")
            return {"error": "Unable to split data"}
        
        try:
            X_test_tfidf = vectorizer.transform(X_test)
        except Exception as e:
            logger.error(f"Error transforming data: {e}")
            return {"error": "Unable to transform data"}
        
        # Predict and calculate accuracy
        try:
            y_pred = model.predict(X_test_tfidf)
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"Model accuracy: {accuracy}")
        except Exception as e:
            logger.error(f"Error predicting or calculating accuracy: {e}")
            return {"error": "Unable to predict or calculate accuracy"}
        
        # Save accuracy to file
        try:
            with open(accuracy_file, "w") as file:
                file.write(str(accuracy))
            logger.info("Accuracy saved to file")
        except Exception as e:
            logger.error(f"Error saving accuracy to file: {e}")
            return {"error": "Unable to save accuracy"}
        
        # Return the accuracy in the response
        return {"accuracy": accuracy}
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        return {"error": "Unable to fetch accuracy"}

@app.get("/accuracy_txt/")
def get_stored_accuracy():
    """Returns the stored model accuracy from model_accuracy.txt"""
    try:
        logger.info(f"Looking for file at: {os.path.abspath(accuracy_file)}")  # Log absolute path
        if os.path.exists(accuracy_file):
            with open(accuracy_file, "r") as file:
                content = file.read().strip()
                # Extract the numeric value from the string
                accuracy = float(content.split(":")[-1].strip())
            return {"accuracy": accuracy}
        else:
            logger.error(f"Model accuracy file not found at {accuracy_file}")
            return {"error": f"Model accuracy file not found at {accuracy_file}"}
    except Exception as e:
        logger.error(f"Error reading model accuracy file: {e}")
        return {"error": f"Unable to read model accuracy file: {str(e)}"}
