import React, { useEffect } from 'react';
import { Routes, Route, Link, useNavigate } from 'react-router-dom';
import ChatPage from './components/ChatPage';
import DocumentsPage from './components/DocumentsPage';
import HomePage from './components/HomePage';

const App = () => {
  const navigate = useNavigate();

  // Handle initial path from server-side routing
  useEffect(() => {
    // Check if we have an original path stored by our script in index.html
    if (window.originalPath && window.originalPath !== '/') {
      const path = window.originalPath;
      // Clear the stored path
      window.originalPath = null;
      // Navigate to the original path
      navigate(path);
    }
  }, [navigate]);

  return (
    <div className="app-container">
      <nav className="navbar navbar-expand-lg navbar-dark bg-primary">
        <div className="container">
          <Link className="navbar-brand" to="/">DISS</Link>
          <button className="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
            <span className="navbar-toggler-icon"></span>
          </button>
          <div className="collapse navbar-collapse" id="navbarNav">
            <ul className="navbar-nav">
              <li className="nav-item">
                <Link className="nav-link" to="/">Home</Link>
              </li>
              <li className="nav-item">
                <Link className="nav-link" to="/chat">Chat</Link>
              </li>
              <li className="nav-item">
                <Link className="nav-link" to="/documents">Documents</Link>
              </li>
            </ul>
          </div>
        </div>
      </nav>

      <div className="container mt-4">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/chat" element={<ChatPage />} />
          <Route path="/documents" element={<DocumentsPage />} />
        </Routes>
      </div>
    </div>
  );
};

export default App;
