// ...existing code...
import React, { useState } from "react";
import './Dashboard.css';
import { useNavigate } from 'react-router-dom';

const Dashboard = ({ onSelect }) => {
    const [selected, setSelected] = useState(null);
    const navigate = useNavigate();

    const options = [
        { id: 'road', title: 'Road Detection', desc: 'Detect roads from SAR imagery' },
        { id: 'landwater', title: 'Land & Water Detection', desc: 'Segment land and water' },
        { id: 'building', title: 'Building Detection', desc: 'Detect buildings in SAR imagery' },
    ];

    const handleSelect = (id) => {
        setSelected(id);
        console.log('Dashboard selection:', id);
        if (onSelect) onSelect(id);

        if(id==='road'){
            // navigate('/road');
            console.log('Navigating to /road');
            window.location.replace('http://localhost:8001/road_output');    
        }
        else if(id==='landwater'){
            // navigate('/landwater');
            console.log('Navigating to /landwater');
            window.location.replace('http://localhost:8001/land_output');
        }
        else{
            // navigate('/colorization');
            console.log('Navigating to /building');
            window.location.replace('http://localhost:8001/building_output');
         }

    };

    return (
        <div className="ss-dashboard">
            <nav className="ss-navbar">
                <div className="ss-brand">
                    <div className="ss-logo" aria-hidden="true">
                        <svg width="36" height="36" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <circle cx="12" cy="12" r="10" stroke="#2dd4bf" strokeWidth="1.5" fill="none"/>
                            <path d="M4 16c2-3 6-5 8-5s6 2 8 5" stroke="#6366f1" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" fill="none"/>
                        </svg>
                    </div>
                    <div className="ss-title">
                       <a href="/"><div className="ss-project">Spectra Space</div></a> 
                        <div className="ss-sub">Colorized SAR</div>
                    </div>
                </div>
            </nav>

            <main className="ss-main">
                <h1 className="ss-heading">Choose a task</h1>

                <div className="ss-options">
                    {options.map(opt => (
                        <button
                            key={opt.id}
                            className={`ss-card ${selected === opt.id ? 'selected' : ''}`}
                            onClick={() => handleSelect(opt.id)}
                            aria-pressed={selected === opt.id}
                        >
                            <div className="ss-card-icon" aria-hidden="true">
                                {opt.id === 'road' && '🛣️'}
                                {opt.id === 'landwater' && '🌊'}
                                {opt.id === 'building' && '🏢'}
                            </div>
                            <div className="ss-card-body">
                                <h3>{opt.title}</h3>
                                <p>{opt.desc}</p>
                            </div>
                        </button>
                    ))}
                </div>

                {/* {selected && (
                    <div className="ss-selection">
                        Selected: <strong>{options.find(o => o.id === selected).title}</strong>
                    </div>
                )} */}
            </main>
        </div>
    );
}

export default Dashboard;
// ...existing code...