import os
import kaggle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib

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

# Define the path to the dataset folder
dataset_folder = 'dataset/'

# Download dataset using Kaggle API (if not already downloaded)
if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)

# Use Kaggle API to download the dataset
print("Downloading dataset from Kaggle...")
kaggle.api.dataset_download_files('rmisra/news-category-dataset', path=dataset_folder, unzip=True)

# Check if the dataset was downloaded
if not os.listdir(dataset_folder):
    print("Dataset download failed or the folder is empty.")
else:
    print("Dataset downloaded successfully.")
    print("Contents of the dataset folder:")
    print(os.listdir(dataset_folder))

# Load the dataset
print("Loading dataset...")
df = pd.read_json(os.path.join(dataset_folder, 'News_Category_Dataset_v3.json'), lines=True)

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

print("Training complete and files saved .")