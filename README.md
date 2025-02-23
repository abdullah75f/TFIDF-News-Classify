# TFIDFNewsClassify

Classify news articles into categories like **Sports**, **Business**, **Tech**, etc., using **TF-IDF** and **Naive Bayes**. This project leverages the Kaggle dataset for training and predicting news categories.

---

## Setup & Installation

1. **Clone the Repo**:
    ```bash
    git clone https://github.com/your-username/TFIDFNewsClassify.git
    cd TFIDFNewsClassify
    ```

2. **Create Virtual Env**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # macOS/Linux
    .\venv\Scripts\activate   # Windows
    ```

3. **Install Dependencies**:
    ```bash
    pip install -r ml_model/requirements.txt
    ```

4. **Kaggle API Setup**:
    - Get `kaggle.json` from [Kaggle API](https://www.kaggle.com/docs/api).
    - Place it in `~/.kaggle/` on your machine.

---

## Training the Model

Run the script to train the model:
```bash
python ml_model/train_model.py

This will:

Download the News Category Dataset from Kaggle.
Train the model using TF-IDF and Naive Bayes.
Save model.pkl and vectorizer.pkl.
Making Predictions
To predict the category of a new headline:
python ml_model/predict.py

Backend API (Optional)
To serve the model via an API:

Install FastAPI:

bash
Copy
Edit
pip install fastapi uvicorn
Run the Server:

bash
Copy
Edit
uvicorn backend.main:app --reload
Dependencies
ML Model Dependencies (ml_model/requirements.txt):
text
Copy
Edit
kaggle
pandas
scikit-learn
joblib
Backend Dependencies (backend/requirements.txt):
text
Copy
Edit
fastapi
uvicorn
License
MIT License. See the LICENSE file.

Acknowledgements
Kaggle Dataset: News Category Dataset - Link
Libraries: scikit-learn, pandas, joblib, FastAPI