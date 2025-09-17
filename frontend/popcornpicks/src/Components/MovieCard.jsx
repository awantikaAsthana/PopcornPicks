import React from 'react'


export default function MovieCard({prop}) {
    console.log(prop)
  return (
    <div>
{
    prop.title
}{
    prop.year
}{
    prop.genres
}{
    prop.imdb_search
}{
    prop.rating
}{
    prop.score
}

    </div>
  )
}
