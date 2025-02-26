TFIDFNewsClassify 🗞️🤖
A machine learning project that classifies news headlines into categories like Sports, Business, and Tech using TF-IDF and LinearSVC.

🚀 Live API
🔹 Frontend: TFIDFNewsClassify UI
🔹 Backend API: TFIDFNewsClassify API

📌 How to Use the API
1️⃣ Make a Prediction
Send a POST request:

plaintext
Copy
Edit
POST https://tfidf-news-classify-6.onrender.com/predict
Request Body (JSON)
json
Copy
Edit
{"headline": "Apple unveils new iPhone with revolutionary features"}
Response
json
Copy
Edit
{"predicted_category": "Tech"}
2️⃣ Check Model Accuracy
plaintext
Copy
Edit
GET https://tfidf-news-classify-6.onrender.com/accuracy
📦 Tech Stack
Machine Learning: Scikit-learn, TF-IDF, LinearSVC
Backend: FastAPI
Deployment: Render
🔗 Dataset
News Category Dataset from Kaggle
📬 Contact
Have questions or suggestions? Reach out at abdullah75farid@gmail.com
