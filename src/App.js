import React, { Component } from 'react';
import Stage from "./Stage/Stage"
import './Css/bootstrap.css';
import './Css/App.css';


class App extends Component {
  render() {
    return (
      <div className="App container-fluid">
        <Stage />
      </div>
    );
  }
}

export default App;
