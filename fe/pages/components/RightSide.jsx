import React, { useEffect } from 'react';
import SyntaxHighlighter from 'react-syntax-highlighter';
import { atomOneDark, atomOneLight } from 'react-syntax-highlighter/dist/cjs/styles/hljs';

const RightSide = (props) => {


  const handleLine = (lineNumber) => {
    console.log(props.data.data)
    console.log(props.data.datalength)
    for(let i = 0; i < props.data.data.length; i++) {
      const d = props.data.data;
      if (d[i]['line'] === lineNumber) {
        return {
          style: { backgroundColor: `rgba(214, 0, 28, ${d[i]['confident']*0.5})`, opacity: `${1.0* d[i]['confident'] }` ,display: 'block', cursor: 'pointer'},
          onClick() { window.open(d[i]['url']) }
        }
      }
    }
    return {
      style: {display: 'block'},
    }
  }

  return (
    <div className='wrapper'>
      <h2>Output Code</h2>
      <SyntaxHighlighter 
        language="python" 
        style={{...atomOneLight
          , height:'450'}}
        showLineNumbers
        wrapLines
        lineProps={lineNumber => (handleLine(lineNumber))}
      >
        {props.outputCode}
      </SyntaxHighlighter>

    </div>
  )
}

export default RightSide