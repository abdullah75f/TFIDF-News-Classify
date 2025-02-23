import React, { useState } from "react";
import axios from "axios";

function App() {
  const [text, setText] = useState("");
  const [category, setCategory] = useState(null);

  const handleSubmit = async () => {
    try {
      // Send the input text to the FastAPI backend for prediction
      const response = await axios.post("https://tfidf-news-classify-6.onrender.com/predict/", {
        text: text,
      });
      setCategory(response.data.category); // Set the predicted category
    } catch (error) {
      console.error("Error:", error);
    }
  };

  return (
    <div style={{ fontFamily: "'Roboto', sans-serif", textAlign: "center", marginTop: "50px", backgroundColor: "#f4f7fc", padding: "20px" }}>
      <h1 style={{ color: "#3f72af", fontSize: "36px", fontWeight: "700" }}>News Classifier</h1>
      <input
        type="text"
        placeholder="Enter news headline..."
        value={text}
        onChange={(e) => setText(e.target.value)}
        style={{
          padding: "12px",
          width: "350px",
          borderRadius: "8px",
          border: "1px solid #ccc",
          fontSize: "16px",
          marginBottom: "20px",
          boxShadow: "0 2px 5px rgba(0,0,0,0.1)",
        }}
      />
      <br />
      <button
        onClick={handleSubmit}
        style={{
          marginLeft: "10px",
          padding: "12px 20px",
          backgroundColor: "#3f72af",
          color: "#fff",
          border: "none",
          borderRadius: "8px",
          cursor: "pointer",
          fontSize: "16px",
          boxShadow: "0 4px 8px rgba(0,0,0,0.1)",
        }}
      >
        Predict
      </button>
      {category && (
        <h2 style={{ color: "#3f72af", marginTop: "20px", fontSize: "24px" }}>
          Predicted Category: {category}
        </h2>
      )}
    </div>
  );
}

export default App;
