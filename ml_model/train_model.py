import os
import json
import re
import joblib
import pandas as pd
import kaggle
import nltk
from nltk.corpus import stopwords
from fastapi import FastAPI, HTTPException
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC  # Improved model
from sklearn.naive_bayes import MultinomialNB  # Original model
from sklearn.metrics import accuracy_score

app = FastAPI()

# Ensure the .kaggle directory exists
kaggle_dir = os.path.expanduser("~/.kaggle")
if not os.path.exists(kaggle_dir):
    os.makedirs(kaggle_dir)

# Ensure the kaggle.json file exists in the .kaggle directory
kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")
if not os.path.exists(kaggle_json_path):
    with open(kaggle_json_path, "w") as f:
        f.write('{"username":"abdullah75f","key":"c2ee65eeab3c12aed3f8100586f95782"}')
    os.chmod(kaggle_json_path, 0o600)

# Authenticate with the Kaggle API
kaggle.api.authenticate()

# Dataset paths
dataset_folder = "dataset"
dataset_file = os.path.join(dataset_folder, "News_Category_Dataset_v3.json")

# Download dataset if not exists
def download_dataset():
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    if not os.path.exists(dataset_file):
        print("Downloading dataset from Kaggle...")
        kaggle.api.dataset_download_files('rmisra/news-category-dataset', path=dataset_folder, unzip=True)
        if not os.path.exists(dataset_file):
            raise Exception("Dataset download failed.")
        print("Dataset downloaded successfully.")

# Preprocess text
nltk.download('stopwords')
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    stop_words = set(stopwords.words('english'))
    text = " ".join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Train and save model
def train_and_save_model():
    print("Loading dataset...")
    df = pd.read_json(dataset_file, lines=True)
    df['headline'] = df['headline'].apply(clean_text)  # Apply text cleaning

    # Prepare data
    X = df['headline']
    y = df['category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # TF-IDF transformation
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=50000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train the model (Switchable: Naïve Bayes → SVM)
    model = LinearSVC()  # Change to MultinomialNB() if you want Naïve Bayes
    model.fit(X_train_tfidf, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy:.2f}')

    # Save model, vectorizer, and metadata
    joblib.dump(model, 'model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')

    metadata = {"model": "LinearSVC", "accuracy": accuracy}
    with open("metadata.json", "w") as f:
        json.dump(metadata, f)

    print("Training complete and files saved.")

# Load model and vectorizer once to speed up API requests
if not os.path.exists('model.pkl') or not os.path.exists('vectorizer.pkl'):
    print("Model or vectorizer not found. Training the model...")
    try:
        download_dataset()
        train_and_save_model()
    except Exception as e:
        print(f"Error during model training: {e}")
        raise HTTPException(status_code=500, detail="Model training failed.")
else:
    print("Model and vectorizer found. Loading...")
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')

# API endpoint for predictions
@app.post("/predict")
def predict(headline: str):
    try:
        headline_clean = clean_text(headline)
        headline_vectorized = vectorizer.transform([headline_clean])
        prediction = model.predict(headline_vectorized)
        return {"predicted_category": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# API to return model accuracy
@app.get("/accuracy")
def get_model_accuracy():
    try:
        with open("metadata.json", "r") as f:
            metadata = json.load(f)
        return metadata
    except Exception:
        raise HTTPException(status_code=500, detail="Error fetching model accuracy")

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Improved News Classification API"}
