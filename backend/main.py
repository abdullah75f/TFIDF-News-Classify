import os
import json
import kaggle
import pandas as pd
from fastapi import FastAPI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

app = FastAPI()

# Define dataset path
DATASET_PATH = "News_Category_Dataset_v3.json"
KAGGLE_DATASET = "rmisra/news-category-dataset"

# Ensure dataset is downloaded
def download_dataset():
    if not os.path.exists(DATASET_PATH):
        print("Dataset not found. Downloading from Kaggle...")
        kaggle.api.dataset_download_files(KAGGLE_DATASET, path=".", unzip=True)
        print("Dataset downloaded successfully.")

# Load dataset
def load_dataset():
    download_dataset()
    with open(DATASET_PATH, "r") as file:
        data = [json.loads(line) for line in file]
    return pd.DataFrame(data)

# Train the model
def train_model():
    df = load_dataset()
    
    # Using 'headline' as text and 'category' as target
    X = df["headline"]
    y = df["category"]
    
    vectorizer = TfidfVectorizer(stop_words="english")
    X_tfidf = vectorizer.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
    
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return model, vectorizer, accuracy, report

# Train the model at startup
model, vectorizer, accuracy, report = train_model()

@app.get("/")
def home():
    return {"message": "News Classification API is running!"}

@app.get("/evaluate")
def evaluate():
    return {"accuracy": accuracy, "classification_report": report}

@app.post("/predict/")
def predict(text: str):
    transformed_text = vectorizer.transform([text])
    prediction = model.predict(transformed_text)[0]
    return {"prediction": prediction}
