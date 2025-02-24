import os
import kaggle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib
from fastapi import FastAPI, HTTPException

app = FastAPI()

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

# Function to download the dataset
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

# Function to train and save the model
def train_and_save_model():
    # Load the dataset
    print("Loading dataset...")
    df = pd.read_json(dataset_file, lines=True)

    # Show the first few rows of the dataset
    print(df.head())

    # Prepare the dataset for training
    X = df['headline']
    y = df['category']

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert text to TF-IDF features
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train the Naive Bayes Classifier
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy:.2f}')

    # Save the trained model and vectorizer
    print("Saving the model and vectorizer...")
    joblib.dump(model, 'model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')

    print("Training complete and files saved.")

# Check if the model and vectorizer exist
if not os.path.exists('model.pkl') or not os.path.exists('vectorizer.pkl'):
    print("Model or vectorizer not found. Training the model...")
    try:
        # Download the dataset
        download_dataset()

        # Train and save the model
        train_and_save_model()
    except Exception as e:
        print(f"Error during model training: {e}")
        raise HTTPException(status_code=500, detail="Model training failed.")
else:
    print("Model and vectorizer found. Skipping training.")

# API route for predictions
@app.post("/predict")
def predict(headline: str):
    try:
        # Load the model and vectorizer
        model = joblib.load('model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')

        # Vectorize the input headline
        headline_vectorized = vectorizer.transform([headline])

        # Predict the category
        prediction = model.predict(headline_vectorized)
        return {"predicted_category": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# Root route
@app.get("/")
def read_root():