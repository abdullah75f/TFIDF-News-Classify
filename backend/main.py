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
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Ensure the .kaggle directory exists for authentication
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
    logger.info("Downloading dataset from Kaggle...")
    kaggle.api.dataset_download_files('rmisra/news-category-dataset', path=dataset_folder, unzip=True)

    # Check if the dataset was downloaded
    if not os.path.exists(dataset_file):
        raise HTTPException(status_code=500, detail="Dataset download failed or the file is missing.")
    logger.info("Dataset downloaded successfully.")

# Function to load the dataset
def load_dataset():
    try:
        logger.info("Loading dataset...")
        df = pd.read_json(dataset_file, lines=True)
        logger.info(f"Dataset loaded successfully with {len(df)} rows")
        return df
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"Error reading JSON file: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error while loading dataset: {e}")

# Function to train and save the model
def train_and_save_model(df):
    try:
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
        logger.info(f'Model Accuracy: {accuracy:.2f}')

        # Save the trained model and vectorizer
        logger.info("Saving the model and vectorizer...")
        joblib.dump(model, 'model.pkl')
        joblib.dump(vectorizer, 'vectorizer.pkl')

        logger.info("Training complete and files saved.")
        return accuracy

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during model training: {e}")

# Endpoint to evaluate the model
@app.get("/evaluate/")
def evaluate_model():
    try:
        # Check if the model and vectorizer exist
        if not os.path.exists('model.pkl') or not os.path.exists('vectorizer.pkl'):
            logger.info("Model or vectorizer not found. Training the model...")
            try:
                # Download the dataset
                download_dataset()

                # Load and train the model
                df = load_dataset()
                accuracy = train_and_save_model(df)

            except HTTPException as e:
                raise e
            except Exception as e:
                logger.error(f"Error during model evaluation: {e}")
                raise HTTPException(status_code=500, detail="Model evaluation failed.")
        else:
            logger.info("Model and vectorizer found. Skipping training.")
            # Load the dataset and use the pre-trained model
            df = load_dataset()
            X = df['headline']
            y = df['category']
            vectorizer = joblib.load('vectorizer.pkl')
            model = joblib.load('model.pkl')

            # Transform the data and predict
            X_test_tfidf = vectorizer.transform(X)
            y_pred = model.predict(X_test_tfidf)
            accuracy = accuracy_score(y, y_pred)
            logger.info(f'Model Accuracy: {accuracy:.2f}')

        # Return the accuracy
        return {"accuracy": accuracy}

    except Exception as e:
        logger.error(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail="Unable to fetch accuracy")

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

# @app.get("/evaluate/")
# def evaluate_model():
#     try:
#         # Define the dataset path
#         dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../ml_model/dataset/News_Category_Dataset_v3.json")
#         logger.info(f"Dataset path: {dataset_path}")
        
#         # Check if the dataset already exists
#         if not os.path.exists(dataset_path):
#             try:
#                 # Download the dataset from Kaggle
#                 logger.info("Dataset not found, downloading from Kaggle...")
#                 kaggle.api.dataset_download_files('rmisra/news-category-dataset', path='../ml_model/dataset', unzip=True)
#                 logger.info("Dataset downloaded successfully")
                
#                 # Log the contents of the directory
#                 dataset_dir = os.path.dirname(dataset_path)
#                 logger.info(f"Contents of dataset directory: {os.listdir(dataset_dir)}")
                
#                 # Check if the dataset file exists after download
#                 if not os.path.exists(dataset_path):
#                     logger.error(f"Dataset file not found after download: {dataset_path}")
#                     return {"error": "Dataset file not found after download"}
                
#             except Exception as e:
#                 logger.error(f"Error downloading dataset from Kaggle: {e}")
#                 return {"error": "Unable to download dataset from Kaggle"}

                
        
#         # Load the test data
#         logger.info("Loading dataset...")
#         try:
#             df = pd.read_json(dataset_path, lines=True)
#             logger.info("Dataset loaded successfully")
#         except ValueError as e:
#             logger.error(f"Error loading dataset (line-delimited): {e}")
#             return {"error": "Unable to load dataset"}
#         except Exception as e:
#             logger.error(f"Error loading dataset: {e}")
#             return {"error": "Unable to load dataset"}
        
#         X = df['headline']
#         y = df['category']
#         logger.info(f"Data split: {len(X)} headlines, {len(y)} categories")
        
#         try:
#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#             logger.info(f"Data split into train and test sets: {len(X_train)} train, {len(X_test)} test")
#         except Exception as e:
#             logger.error(f"Error splitting data: {e}")
#             return {"error": "Unable to split data"}
        
#         try:
#             X_test_tfidf = vectorizer.transform(X_test)
#             logger.info("Data transformed using vectorizer")
#         except Exception as e:
#             logger.error(f"Error transforming data: {e}")
#             return {"error": "Unable to transform data"}
        
#         # Predict and calculate accuracy
#         try:
#             y_pred = model.predict(X_test_tfidf)
#             accuracy = accuracy_score(y_test, y_pred)
#             logger.info(f"Model accuracy: {accuracy}")
#         except Exception as e:
#             logger.error(f"Error predicting or calculating accuracy: {e}")
#             return {"error": "Unable to predict or calculate accuracy"}
        
#         # Return the accuracy in the response
#         return {"accuracy": accuracy}
#     except Exception as e:
#         logger.error(f"Error occurred: {e}")
#         return {"error": "Unable to fetch accuracy"}