import React, { useState } from "react";
import axios from "axios";

function App() {
  const [text, setText] = useState("");
  const [category, setCategory] = useState(null);

  const handleSubmit = async () => {
    try {
      // Send the input text to the FastAPI backend for prediction
      const response = await axios.post("http://127.0.0.1:8000/predict/", {
        text: text,
      });
      setCategory(response.data.category); // Set the predicted category
    } catch (error) {
      console.error("Error:", error);
    }
  };

  return (
    <div style={{ textAlign: "center", marginTop: "50px" }}>
      <h1>News Classifier</h1>
      <input
        type="text"
        placeholder="Enter news headline..."
        value={text}
        onChange={(e) => setText(e.target.value)}
        style={{ padding: "10px", width: "300px" }}
      />
      <button onClick={handleSubmit} style={{ marginLeft: "10px", padding: "10px" }}>
        Predict
      </button>
      {category && <h2>Predicted Category: {category}</h2>}
    </div>
  );
}

export default App;
