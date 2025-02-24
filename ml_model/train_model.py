import os
import kaggle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS: Allow only frontend URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://tfidf-news-classify.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure the .kaggle directory exists
kaggle_dir = os.path.expanduser("~/.kaggle")
os.makedirs(kaggle_dir, exist_ok=True)

# Ensure the kaggle.json file exists
kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")
if not os.path.exists(kaggle_json_path):
    with open(kaggle_json_path, "w") as f:
        f.write('{"username":"abdullah75f","key":"c2ee65eeab3c12aed3f8100586f95782"}')
    os.chmod(kaggle_json_path, 0o600)  # Secure permissions

# Authenticate with the Kaggle API
kaggle.api.authenticate()

# Paths
dataset_folder = 'dataset/'
dataset_file = os.path.join(dataset_folder, 'News_Category_Dataset_v3.json')
model_path = "model.pkl"
vectorizer_path = "vectorizer.pkl"

# Function to download the dataset
def download_dataset():
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    kaggle.api.dataset_download_files('rmisra/news-category-dataset', path=dataset_folder, unzip=True)
    if not os.path.exists(dataset_file):
        raise Exception("Dataset download failed.")

# Function to train and save the model
def train_and_save_model():
    df = pd.read_json(dataset_file, lines=True)
    X_train, X_test, y_train, y_test = train_test_split(df['headline'], df['category'], test_size=0.2, random_state=42)

    # Use parallel processing for vectorization
    vectorizer = TfidfVectorizer(stop_words='english', n_jobs=-1)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    # Save trained model
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

# Background model training task
def background_training():
    try:
        download_dataset()
        train_and_save_model()
    except Exception as e:
        print(f"Error during training: {e}")

# Preload model & vectorizer if available
if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
else:
    model, vectorizer = None, None  # Train in the background
    background_training()

# API to predict category
@app.post("/predict")
def predict(headline: str):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model is still training. Try again later.")

    headline_vectorized = vectorizer.transform([headline])
    prediction = model.predict(headline_vectorized)
    return {"predicted_category": prediction[0]}

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to TFIDFNewsClassify!"}

# Run background training if model is missing
@app.on_event("startup")
async def startup_event():
    global model, vectorizer
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        background_training()
    else:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
