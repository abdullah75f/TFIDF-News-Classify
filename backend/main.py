from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
from pydantic import BaseModel

# Load model and vectorizer once at startup
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = FastAPI()

# Allow only your frontend origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://tfidf-news-classify.onrender.com"],  # Your frontend URL
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

@app.get("/")
def root():
    return {"message": "TFIDF News Classifier API is running"}
