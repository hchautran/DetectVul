import React, { useState } from 'react';
import CodeMirror from '@uiw/react-codemirror';
import { langs } from '@uiw/codemirror-extensions-langs';
import  { atomone } from '@uiw/codemirror-theme-atomone';

const LeftSide = (props) => {
  const onChange =  React.useCallback((value, viewUpdate) => props.setInputCode(value));
  return (
    <div className="wrapper">
      <h2>Input Code</h2>
      <CodeMirror 
        value={props.inputCode}
        maxHeight='600px'
        minHeight='600px'
        width='800px'
        placeholder={"Type your code here..."}
        onChange={onChange}
        theme={atomone}
        extensions={[langs.python()]}
      />
    </div>
  )
}

export default LeftSide