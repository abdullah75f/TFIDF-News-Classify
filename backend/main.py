from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
from pydantic import BaseModel
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
import kaggle

app = FastAPI()

# Allow all origins for development (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure the .kaggle directory exists
kaggle_dir = os.path.expanduser("~/.kaggle")
if not os.path.exists(kaggle_dir):
    os.makedirs(kaggle_dir)

# Ensure the kaggle.json file exists in the .kaggle directory
kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")
if not os.path.exists(kaggle_json_path):
    # Add your Kaggle API credentials here
    with open(kaggle_json_path, "w") as f:
        f.write('{"username":"abdullah75f","key":"c2ee65eeab3c12aed3f8100586f95782"}')
    os.chmod(kaggle_json_path, 0o600)  # Set proper permissions for the kaggle.json file

# Authenticate with the Kaggle API
kaggle.api.authenticate()

# Define the path to the dataset folder
dataset_folder = 'dataset/'
dataset_file = os.path.join(dataset_folder, 'News_Category_Dataset_v3.json')

# Function to download the dataset using Kaggle API
def download_dataset():
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    # Use Kaggle API to download the dataset
    print("Downloading dataset from Kaggle...")
    kaggle.api.dataset_download_files('rmisra/news-category-dataset', path=dataset_folder, unzip=True)

    # Check if the dataset was downloaded
    if not os.path.exists(dataset_file):
        raise Exception("Dataset download failed or the file is missing.")
    print("Dataset downloaded successfully.")

# Load model and vectorizer
model = joblib.load("../ml_model/model.pkl")
vectorizer = joblib.load("../ml_model/vectorizer.pkl")

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
        # Download the dataset if it doesn't exist
        if not os.path.exists(dataset_file):
            download_dataset()

        # Load the test data
        df = pd.read_json(dataset_file, lines=True)
        X = df['headline']
        y = df['category']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_test_tfidf = vectorizer.transform(X_test)

        # Predict and calculate accuracy
        y_pred = model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        return {"accuracy": accuracy}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during evaluation: {e}")