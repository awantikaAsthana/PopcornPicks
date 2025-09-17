
import React, {useState} from 'react'
import axios from 'axios'
import MovieCard from './MovieCard';





export default function SearchBar() {
    const [Data,changeData] = useState(null);
    const handleChange = () => {
        var Data = {
            title: document.getElementById("Title").value,
            rating: document.getElementById("Rating").value,
            count: document.getElementById("Count").value
        }
        console.log(Data);
        axios.post('http://127.0.0.1:5000/recommend', Data,  {headers: {
    'Content-Type': 'application/json'
  }})
  .then(response => {
    console.log('User created successfully:', response.data);
    changeData(response.data.recommendations);
  })
  .catch(error => {
    console.error('Error creating user:', error);
  });
    }
  return (
    
    <div>
        <div>
       <input type="text" placeholder='Enter Title' id='Title' required />
       <input type="range" min={0.0} max={5.0} id='Rating' />
       <input type="text" id='Count'  />
       <button onClick={handleChange}>Submit</button>
       </div>

       <div>
        { Data? 
            Data.map((x)=>{
                return(
                    
                        <MovieCard prop ={x}/>
                    
                )
            }): <></>
        }
       </div>
    </div>
  )
}
