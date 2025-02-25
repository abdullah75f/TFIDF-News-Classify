from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
from pydantic import BaseModel
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import os
import kaggle

# Paths
MODEL_DIR = "../ml_model/"
DATASET_FOLDER = os.path.join(MODEL_DIR, "dataset")
DATASET_FILE = os.path.join(DATASET_FOLDER, "News_Category_Dataset_v3.json")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")

# 🟢 Ensure Kaggle credentials are set
KAGGLE_USERNAME = "abdullah75f"
KAGGLE_KEY = "c2ee65eeab3c12aed3f8100586f95782"

kaggle_dir = os.path.expanduser("~/.kaggle")
kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")

if not os.path.exists(kaggle_json_path):
    os.makedirs(kaggle_dir, exist_ok=True)
    with open(kaggle_json_path, "w") as f:
        f.write(f'{{"username":"{KAGGLE_USERNAME}","key":"{KAGGLE_KEY}"}}')
    os.chmod(kaggle_json_path, 0o600)  # Secure permissions

# Authenticate with Kaggle API
kaggle.api.authenticate()

# 🟢 Function to download dataset from Kaggle
def download_dataset():
    try:
        if not os.path.exists(DATASET_FOLDER):
            os.makedirs(DATASET_FOLDER)
        print("Downloading dataset from Kaggle...")
        kaggle.api.dataset_download_files("rmisra/news-category-dataset", path=DATASET_FOLDER, unzip=True)

        # Check if dataset exists
        if not os.path.exists(DATASET_FILE):
            raise Exception("Dataset download failed or file is missing.")
        print("Dataset downloaded successfully.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download dataset: {e}")

# 🟢 Ensure dataset exists before evaluation
if not os.path.exists(DATASET_FILE):
    print("Dataset not found. Downloading...")
    download_dataset()

# Load Model and Vectorizer
try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("Model and vectorizer loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    raise HTTPException(status_code=500, detail="Model could not be loaded.")

# FastAPI App
app = FastAPI()

# CORS (Allow all origins for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input Model for Prediction
class NewsInput(BaseModel):
    text: str

# 🟢 Prediction Route
@app.post("/predict/")
def predict_category(news: NewsInput):
    try:
        text_vectorized = vectorizer.transform([news.text])
        prediction = model.predict(text_vectorized)[0]
        return {"category": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# 🟢 Evaluation Route
@app.get("/evaluate/")
def evaluate_model():
    try:
        # Load dataset
        df = pd.read_json(DATASET_FILE, lines=True)
        X = df['headline']
        y = df['category']

        # Split dataset
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Vectorize test data
        X_test_tfidf = vectorizer.transform(X_test)

        # Predict categories
        y_pred = model.predict(X_test_tfidf)

        # Compute Evaluation Metrics
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred).tolist()  # Convert to list for JSON response

        return {
            "accuracy": accuracy,
            "classification_report": class_report,
            "confusion_matrix": conf_matrix
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model evaluation failed: {e}")

# 🟢 Root Endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the News Classification API"}
