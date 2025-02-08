import React, { useEffect, useState } from "react";
import "./App.css";

function App() {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");

  // Function to fetch data
  const fetchData = async () => {
    try {
      const response = await fetch("http://127.0.0.1:5000/getdata");
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const jsonData = await response.json();
      setData(jsonData);
      setError(""); // Clear error when successful
    } catch (error) {
      setError("Cam server Not Found, Trying to connect...");
      console.error("Cam server Not Found", error);
    }
  };

  // Use useEffect to fetch data initially and every 5 seconds
  useEffect(() => {
    fetchData(); // Initial fetch
    const interval = setInterval(fetchData, 5000); // Fetch data every 5 seconds

    return () => clearInterval(interval); // Cleanup on unmount
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <h1>ChairFriend</h1>
      </header>

      <div id="data-container">
        {error ? (
          <p style={{ color: "red" }}>{error}</p>
        ) : data ? (
          <pre>{JSON.stringify(data, null, 2)}</pre>
        ) : (
          <p>Loading...</p>
        )}
      </div>
    </div>
  );
}

export default App;
