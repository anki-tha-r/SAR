import logo from './logo.svg';
import './App.css';
import { BrowserRouter as Router,Routes, Route, Switch } from 'react-router-dom';
import Dashboard from './Dashboard';
import RoadDetection from './RoadDetection';
import LandWaterDetection from './LandWaterDetection';
import Colorization from './Colorization';  

function App() {
  return (
    <Router>
     <Routes>
        <Route path="/" element={<Dashboard/>} />
        <Route path="/road" element={<RoadDetection/>} />
        <Route path="/landwater" element={<LandWaterDetection/>} />
        <Route path="/colorization" element={<Colorization/>} />
    </Routes>
    </Router>
  );
}

export default App;
