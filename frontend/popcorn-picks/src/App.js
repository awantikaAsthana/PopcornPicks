import './App.css';
import React, {useState} from 'react';
import axios from 'axios';


function App() {

var [Movie, setMovie] = useState(null);
var [Error, setError] = useState(null);


async function fetchRecommendations() {
  try {
    var  title  = document.getElementById('title').value;
var  count = document.getElementById('count').value;
var   rating = document.getElementById('rating').value;
    const body = {
      movies: title,
      count: count,
      min_rating: rating
    };

    const res = await axios.post("http://127.0.0.1:5000/recommend", body, {
      headers: { "Content-Type": "application/json" }
    });

    console.log("API response:", res.data);
    setMovie(res.data.recommendations);
    return res.data;

  } catch (error) {
    if (error.response) {
      // Flask returned an error JSON
      console.error("API error:", error.response.data);
      setError(error.response.data.message || "An error occurred while fetching recommendations.");
    } else {
      console.error("Request error:", error.message);
      setError("An error occurred while fetching recommendations.");
    }
    return null; // fallback
  }
}

  return (
    <div className="App">
      <header className="App-header">
       Find your next favorite movie!
      <div>
        <input type="text" placeholder="Search for a movie..." id='title' />
        <input type="int" placeholder="enter count" defaultValue={5} id='count'/>
     
<select name="" id="rating" defaultValue={5}>
  <option value="1">1+</option>
  <option value="2">2+</option>
  <option value="3">3+</option>
  <option value="4">4+</option>
</select>
        <button onClick={fetchRecommendations}>Search</button>

      </div>


<table>
  <thead>
    <td>Name</td>
    <td>Year</td>
    <td>Genre</td>
    <td>Rating</td>
    <td>Similarity</td>
  </thead>
  <tbody>
  {Movie?
  Movie.map((movie, index) => (
    <tr key={index}>
      <td>{movie.title}</td>
      <td>{movie.year}</td>
      <td>{movie.genres}</td>
      <td>{movie.rating}</td>
      <td>{movie.similarity}</td>
    </tr>
  ))
:Error? <div style={{color:'red'}}> {Error} </div>:
    <div> Magic will happen here! </div>}
  </tbody>
</table>

      </header>
    </div>
  );
}

export default App;
