import React, { useState } from "react";
import axios from "axios";
import { FaSpinner, FaAngleDown, FaAngleUp } from "react-icons/fa"; // Arrow icons for toggling

function App() {
  const [text, setText] = useState("");
  const [category, setCategory] = useState(null);
  const [loading, setLoading] = useState(false);
  const [darkMode, setDarkMode] = useState(false);
  const [selectedHeadline, setSelectedHeadline] = useState("");
  const [accuracy, setAccuracy] = useState(null);
  const [tfidfValues, setTfidfValues] = useState(null);
  const [showTfidf, setShowTfidf] = useState(false); // Track TF-IDF visibility


  const exampleHeadlines = [
    {
      headline: "Apple announces new iPhone with AI features",
      prediction: "Tech",
    },
    {
      headline: "Stock market hits record high amid economic growth",
      prediction: "POLITICS",
    },
    {
      headline: "Champions League final: Exciting match ends in draw",
      prediction: "Sports",
    },
  ];

  // const handleSubmit = async () => {
  //   if (!text.trim()) return;
  //   setLoading(true);
  //   try {
  //     const response = await axios.post(
  //       "https://tfidf-news-classify-6.onrender.com/predict/",
  //       {
  //         text: text,
  //       }
  //     );
  //     // const response = await axios.post("http://localhost:8000/predict/", {
  //     //   text: text,
  //     // });
  //     setCategory(response.data.category);
  //   } catch (error) {
  //     console.error("Error:", error);
  //     setCategory("Error: Unable to classify");
  //   }
  //   setLoading(false);
  // };

  const handleSubmit = async () => {
    if (!text.trim()) return;
    setLoading(true);
    try {
      const response = await axios.post(
        "https://tfidf-news-classify-6.onrender.com/predict/",
        { text }
      );
      setCategory(response.data.category);
      setTfidfValues(response.data.tfidf_values); // Store TF-IDF values
    } catch (error) {
      console.error("Error:", error);
      setCategory("Error: Unable to classify");
    }
    setLoading(false);
  };

  const handleSelectChange = (e) => {
    setSelectedHeadline(e.target.value);
    setText(e.target.value); // Update the text area when a headline is selected
  };

  const handleEvaluate = async () => {
    try {
      const response = await axios.get(
        "https://tfidf-news-classify-6.onrender.com/accuracy_txt/"
      );
      if (response.data.accuracy !== undefined) {
        setAccuracy(response.data.accuracy);
      } else {
        setAccuracy("Error: Unable to fetch accuracy");
      }
    } catch (error) {
      console.error("Error:", error);
      setAccuracy("Error: Unable to fetch accuracy");
    }
  };

  return (
    <div
      style={{
        fontFamily: "'Poppins', sans-serif",
        display: "flex",
        flexDirection: "column",
        minHeight: "100vh", // Ensure the full height is used
        background: darkMode ? "#121212" : "#F0F4F8",
        color: darkMode ? "white" : "black",
        transition: "all 0.3s ease-in-out",
      }}
    >
      {/* NAVBAR */}
      <div
        style={{
          backgroundColor: darkMode ? "#333" : "#3f72af",
          padding: "15px 30px",
          color: "white",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          fontSize: "24px",
        }}
      >
        <div>📰 News Classifier</div>
        <button
          onClick={() => setDarkMode(!darkMode)}
          style={{
            background: "none",
            border: "1px solid white",
            padding: "8px 12px",
            color: "white",
            borderRadius: "6px",
            cursor: "pointer",
          }}
        >
          {darkMode ? "☀ Light Mode" : "🌙 Dark Mode"}
        </button>
      </div>

      {/* MAIN CONTENT */}
      <div
        style={{
          display: "flex",
          flex: 1, // Ensure the content takes up remaining space
          justifyContent: "space-evenly",
          alignItems: "center",
          padding: "40px 10px",
          flexWrap: "wrap",
        }}
      >
        {/* LEFT SIDE - INPUT SECTION */}
        <div
          style={{
            backgroundColor: darkMode ? "#444" : "#fff",
            padding: "30px",
            borderRadius: "12px",
            boxShadow: "0 10px 30px rgba(0,0,0,0.1)",
            width: "45%",
            maxWidth: "500px",
            transition: "all 0.3s ease",
          }}
        >
          <h2
            style={{
              color: "#3f72af",
              textAlign: "center",
              marginBottom: "20px",
            }}
          >
            Enter or Select News Headline
          </h2>

          {/* Select Dropdown for Predefined Headlines */}
          <select
            value={selectedHeadline}
            onChange={handleSelectChange}
            style={{
              width: "100%",
              padding: "12px",
              fontSize: "16px",
              borderRadius: "8px",
              border: darkMode ? "1px solid #444" : "1px solid #ccc",
              backgroundColor: darkMode ? "#333" : "#f4f4f4",
              color: darkMode ? "white" : "black",
              marginBottom: "15px",
            }}
          >
            <option value="">Select a sample headline...</option>
            {exampleHeadlines.map((example, index) => (
              <option key={index} value={example.headline}>
                {example.headline}
              </option>
            ))}
          </select>

          {/* Or, Text Input for Custom Headline */}
          <textarea
            placeholder="Type your news headline here..."
            value={text}
            onChange={(e) => setText(e.target.value)}
            style={{
              width: "100%",
              height: "150px",
              padding: "12px",
              fontSize: "16px",
              borderRadius: "8px",
              border: darkMode ? "1px solid #444" : "1px solid #ccc",
              backgroundColor: darkMode ? "#333" : "#f4f4f4",
              color: darkMode ? "white" : "black",
              resize: "none",
              boxSizing: "border-box",
            }}
          />
          <button
            onClick={handleSubmit}
            style={{
              padding: "12px 20px",
              backgroundColor: "#3f72af",
              color: "white",
              border: "none",
              borderRadius: "8px",
              cursor: "pointer",
              width: "100%",
              marginTop: "20px",
              fontWeight: "600",
            }}
          >
            {loading ? <FaSpinner className="fa-spin" /> : "Predict"}
          </button>
          <button
            onClick={handleEvaluate}
            style={{
              padding: "12px 20px",
              backgroundColor: "#3f72af",
              color: "white",
              border: "none",
              borderRadius: "8px",
              cursor: "pointer",
              width: "100%",
              marginTop: "20px",
              fontWeight: "600",
            }}
          >
            Evaluate Model
          </button>
        </div>

        {/* RIGHT SIDE - EXAMPLE HEADLINES AND PREDICTIONS */}
        <div
          style={{
            backgroundColor: darkMode ? "#444" : "#fff",
            padding: "10px",
            borderRadius: "12px",
            boxShadow: "0 10px 30px rgba(0,0,0,0.1)",
            width: "45%",
            maxWidth: "500px",
            transition: "all 0.3s ease",
          }}
        >
          <h2
            style={{
              color: "#3f72af",
              textAlign: "center",
              marginBottom: "20px",
            }}
          >
            Example Headlines
          </h2>
          <div>
            {exampleHeadlines.map((example, index) => (
              <div
                key={index}
                style={{
                  backgroundColor: darkMode ? "#333" : "#f4f7fc",
                  marginBottom: "15px",
                  padding: "12px",
                  borderRadius: "8px",
                  border: darkMode ? "1px solid #444" : "1px solid #ccc",
                }}
              >
                <h4 style={{ color: "#3f72af", fontWeight: "600" }}>
                  {example.headline}
                </h4>
                <p style={{ color: darkMode ? "#ccc" : "#333" }}>
                  Predicted Category: <strong>{example.prediction}</strong>
                </p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* PREDICTION RESULT */}
      <div
        style={{
          backgroundColor: darkMode ? "#333" : "#f9f9f9",
          textAlign: "center",
          fontSize: "18px",
          fontWeight: "600",
          marginBottom: "80px",
        }}
      >
        {category ? (
          <div>
            <h3 style={{ color: "#3f72af" }}>Prediction Result: {category}, to look tf-idf values press 👇</h3>
          </div>
        ) : (
          <p>
            {loading
              ? "Classifying..."
              : "Enter a headline to get predictions!"}
          </p>
        )}
        {accuracy !== null && (
          <div>
            <h3 style={{ color: "#3f72af" }}>Model Accuracy: {accuracy}</h3>
          </div>
        )}
        {tfidfValues && (
          <div style={{ textAlign: "center", marginTop: "20px" }}>
            <button
              onClick={() => setShowTfidf(!showTfidf)}
              style={{
                padding: "10px",
                backgroundColor: "#3f72af",
                color: "white",
                border: "none",
                borderRadius: "8px",
                cursor: "pointer",
                marginBottom: "10px",
              }}
            >
              {showTfidf ? <FaAngleUp /> : <FaAngleDown />} Show TF-IDF Values
            </button>
            {showTfidf && (
              <div>
                <h3 style={{ color: "#3f72af" }}>TF-IDF Values:</h3>
                <div
                  style={{
                    display: "flex",
                    flexWrap: "wrap",
                    justifyContent: "center",
                    gap: "15px",
                  }}
                >
                  {Object.entries(tfidfValues).map(([word, score]) => (
                    <div
                      key={word}
                      style={{
                        backgroundColor: darkMode ? "#333" : "#f4f7fc",
                        padding: "8px 12px",
                        borderRadius: "8px",
                        border: darkMode ? "1px solid #444" : "1px solid #ccc",
                        display: "inline-block",
                      }}
                    >
                      <strong>{word}:</strong> {score.toFixed(4)}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* FIXED FOOTER */}
      <div
        style={{
          position: "fixed",
          bottom: "0",
          width: "100%",
          backgroundColor: darkMode ? "#222" : "#3f72af",
          color: "white",
          padding: "15px",
          textAlign: "center",
          fontSize: "16px",
        }}
      >
        © 2025 News Classifier | Made with ❤️ for News Prediction | Designed by
        Abdullah F. Al-Shehabi
      </div>
    </div>
  );
}

export default App;
