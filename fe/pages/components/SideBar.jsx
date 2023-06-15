import axios from 'axios'
import React, { Component } from 'react'
import data from '../../data.json'
import { v4 as uuidv4 } from 'uuid';


export default class SideBar extends Component {
  render() {
    return (
      <div className="side-bar">
        <h2>CWE Helper</h2>
        {data.map(d => (
          <span 
            className='cwe' 
            key={uuidv4()}
            // onClick={window.open(d.url)}
          >
          {d.name_cwe}
          </span>
        ))}
      </div>
    )
  }

  componentDidMount() {
    axios.get('http://localhost:5000/CWE')
      .then(res => {
        console.log(res);
      }) 
      .catch(err => console.error(err))
  }
}
