# **TFIDFNewsClassify 🗞️🤖**  

A **machine learning project** that classifies news headlines into categories like Sports, Business, and Tech using TF-IDF for feature extraction and LinearSVC for classification. The dataset is preprocessed by cleaning the text and removing stopwords. TF-IDF Vectorizer transforms the cleaned text into numerical features, and the LinearSVC model is trained to predict the category. The trained model is saved for future use, and the prediction is served through an API.

  

## 🚀 **Live API**  
🔹 **Frontend:** [TFIDFNewsClassify UI](https://tfidf-news-classify.onrender.com)  
🔹 **Backend:** [TFIDFNewsClassify API](https://tfidf-news-classify-6.onrender.com)  

## 📌 **How to Use the API**  

### **1️⃣ Make a Prediction**  
Send a **POST** request:  
```http
POST https://tfidf-news-classify-6.onrender.com/predict
Request Body (JSON)
{
  "headline": "Apple unveils new iPhone with revolutionary features"
}
Response
{
  "predicted_category": "Tech"
}
```
### **2️⃣ Check Model Accuracy**  
Send a **GET** request:
```http
GET https://tfidf-news-classify-6.onrender.com/accuracy
```
## **📦 Tech Stack**
**Machine Learning:** Scikit-learn, TF-IDF, LinearSVC  
**Backend:** FastAPI  
**Frontend:** Reactjs  
**Deployment**: Render  
## **🔗 Dataset**
News Category Dataset from Kaggle
## **📬 Contact**
Have questions or suggestions? Reach out at abdullah75farid@gmail.com 🚀
