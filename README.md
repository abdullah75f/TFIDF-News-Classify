# **TFIDFNewsClassify: News Article Classification Using TF-IDF and Naive Bayes**

This project classifies news articles into categories like **Sports**, **Business**, **Tech**, etc., using a **TF-IDF** (Term Frequency-Inverse Document Frequency) vectorizer and a **Naive Bayes** classifier. The model is trained on the **News Category Dataset** from Kaggle.

## **Project Structure**

TFIDFNewsClassify/ ├── backend/ # Backend folder (FastAPI) │ └── main.py # FastAPI backend for serving model predictions ├── ml_model/ # Machine learning model folder │ ├── train_model.py # Script for training and saving the ML model │ ├── predict.py # Script for making predictions │ ├── venv/ # Virtual environment │ ├── model.pkl # Trained Naive Bayes model │ ├── vectorizer.pkl # Saved TF-IDF vectorizer │ └── requirements.txt # Python dependencies for ML model ├── .gitignore # Ignored files for Git version control ├── .kaggle/ # Kaggle API credentials folder │ └── kaggle.json # Kaggle API credentials file ├── README.md # Project documentation

bash
Copy
Edit

## **Installation**

### **1. Clone the repository**

```bash
git clone https://github.com/your-username/TFIDFNewsClassify.git
cd TFIDFNewsClassify
2. Set up the Virtual Environment
First, ensure that you have Python 3.9+ installed. If not, please install it first.

Then, create a virtual environment:

bash
Copy
Edit
python3 -m venv venv
Activate the virtual environment:

For macOS/Linux:
bash
Copy
Edit
source venv/bin/activate
For Windows:
bash
Copy
Edit
.\venv\Scripts\activate
3. Install Dependencies
Install the necessary dependencies by running:

bash
Copy
Edit
pip install -r ml_model/requirements.txt
This will install all the required packages, including kaggle, pandas, scikit-learn, joblib, and others.

4. Set Up Kaggle API Credentials
To download the dataset from Kaggle, you need to set up the Kaggle API credentials:

If you don't have a Kaggle account, create one at https://www.kaggle.com.
Go to the Account section of your profile and click Create New API Token. This will download the kaggle.json file, which contains your API credentials.
Place the kaggle.json file in the .kaggle folder inside your home directory:

bash
Copy
Edit
~/.kaggle/kaggle.json
On macOS/Linux, you can do this by running:

bash
Copy
Edit
mkdir -p ~/.kaggle
mv path/to/kaggle.json ~/.kaggle/
5. Run the Training Script
Once everything is set up, you can start training your model:

bash
Copy
Edit
python ml_model/train_model.py
This will:

Download the Kaggle dataset using the Kaggle API.
Process the dataset and train the Naive Bayes classifier using a TF-IDF vectorizer.
Save the trained model and vectorizer as model.pkl and vectorizer.pkl in the ml_model folder.
6. Making Predictions
After training the model, you can use the predict.py script to classify new news articles.

bash
Copy
Edit
python ml_model/predict.py
This script will load the trained model and vectorizer, accept an input headline, and predict the category of the article.

Model Overview
The model uses the following steps:

Dataset Loading: The News Category Dataset is loaded using pandas. This dataset contains headlines and their corresponding categories (e.g., Sports, Tech, Business, etc.).
Data Preprocessing: The text data (headlines) is transformed into numerical features using TF-IDF vectorization.
Model Training: A Naive Bayes classifier is trained on the preprocessed text data.
Model Evaluation: The model is evaluated on a test set using accuracy as the performance metric.
Model Saving: The trained model and vectorizer are saved to disk using joblib.
Model Performance
The model's performance is evaluated based on its accuracy on the test set. You can experiment with different machine learning models or hyperparameters to improve performance.

Backend (Optional)
You can use the trained model in the FastAPI backend. Here's an overview of how it works:

The FastAPI backend (backend/main.py) serves an API endpoint where you can send POST requests with the headline text, and it will return the predicted category.

Install FastAPI and Uvicorn (if not already done):

bash
Copy
Edit
pip install fastapi uvicorn
Run the Backend:
bash
Copy
Edit
uvicorn backend.main:app --reload
This will start a local server, and you can test the API by sending POST requests to the /predict endpoint.

Dependencies
ML Model (in ml_model/requirements.txt)
nginx
Copy
Edit
kaggle
pandas
scikit-learn
joblib
Backend (in backend/requirements.txt)
nginx
Copy
Edit
fastapi
uvicorn
License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgements
Kaggle Dataset: The News Category Dataset was sourced from Kaggle. You can find more details here.
Libraries Used:
scikit-learn for machine learning and text vectorization.
pandas for data handling.
joblib for saving the model.