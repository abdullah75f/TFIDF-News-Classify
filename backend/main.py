from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
from pydantic import BaseModel
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
import kaggle





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

# Kaggle API setup
def setup_kaggle():
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

# Function to download the dataset
def download_dataset():
    dataset_folder = "dataset"
    dataset_file = os.path.join(dataset_folder, "News_Category_Dataset_v3.json")

    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    # Use Kaggle API to download the dataset
    print("Downloading dataset from Kaggle...")
    kaggle.api.dataset_download_files("rmisra/news-category-dataset", path=dataset_folder, unzip=True)

    # Check if the dataset was downloaded
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Dataset file not found at {dataset_file}")
    print("Dataset downloaded successfully.")

# Initialize Kaggle API
setup_kaggle()


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