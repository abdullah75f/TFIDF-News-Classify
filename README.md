# TFIDFNewsClassify 🗞️🤖

A **machine learning project** to classify news articles into categories like **Sports**, **Business**, **Tech**, and more using **TF-IDF** for feature extraction and **Naive Bayes** for classification. Perfect for understanding text classification and building a robust NLP pipeline!

---

## 🌟 Features

- **TF-IDF Vectorization**: Converts text data into meaningful numerical features.
- **Naive Bayes Classifier**: Lightweight and efficient for text classification tasks.
- **Easy-to-Use API**: Optional FastAPI backend for serving predictions.
- **Kaggle Integration**: Automatically downloads the dataset using the Kaggle API.

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/TFIDFNewsClassify.git
cd TFIDFNewsClassify
2. Set Up a Virtual Environment
bash
Copy
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
.\venv\Scripts\activate   # Windows
3. Install Dependencies
bash
Copy
pip install -r ml_model/requirements.txt
4. Set Up Kaggle API
Get your kaggle.json from Kaggle API.

Place it in ~/.kaggle/ on your machine.

🛠️ Training the Model
Train the model and download the dataset with a single command:

bash
Copy
python ml_model/train_model.py
What Happens?
Downloads the News Category Dataset from Kaggle.

Preprocesses the data and applies TF-IDF vectorization.

Trains a Naive Bayes classifier.

Saves the trained model (model.pkl) and vectorizer (vectorizer.pkl) for future use.

🎯 Making Predictions
Predict the category of a news headline:

bash
Copy
python ml_model/predict.py
Example Input:
Copy
"Apple unveils new iPhone with revolutionary features"
Example Output:
Copy
Predicted Category: Tech
🚀 Optional: Backend API
Serve your model as an API using FastAPI:

Install FastAPI and Uvicorn:

bash
Copy
pip install fastapi uvicorn
Run the server:

bash
Copy
uvicorn backend.main:app --reload
Access the API at http://127.0.0.1:8000 and use the /predict endpoint to get predictions.

📦 Dependencies
ML Model (ml_model/requirements.txt)
Copy
kaggle
pandas
scikit-learn
joblib
Backend (backend/requirements.txt)
Copy
fastapi
uvicorn
📝 License
This project is licensed under the MIT License. See the LICENSE file for details.

👏 Acknowledgements
Dataset: News Category Dataset from Kaggle.

Libraries: Scikit-learn, Pandas, Joblib, FastAPI, and Uvicorn.

💡 Contributing
Contributions are welcome! If you have ideas for improvements or find any issues, feel free to:

Open an issue.

Submit a pull request.

📬 Contact
Have questions or suggestions? Reach out to me at abdullah75farid@gmail.com

Happy coding! 🚀
Let’s build something amazing together!

