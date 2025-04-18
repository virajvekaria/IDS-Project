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
      <nav className="navbar navbar-expand-lg navbar-dark">
        <div className="container">
          <Link className="navbar-brand" to="/">
            <i className="bi bi-search-heart me-2"></i>
            DISS
          </Link>
          <button className="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
            <span className="navbar-toggler-icon"></span>
          </button>
          <div className="collapse navbar-collapse" id="navbarNav">
            <ul className="navbar-nav ms-auto">
              <li className="nav-item mx-1">
                <Link className="nav-link" to="/">
                  <i className="bi bi-house me-1"></i> Home
                </Link>
              </li>
              <li className="nav-item mx-1">
                <Link className="nav-link" to="/chat">
                  <i className="bi bi-chat-dots me-1"></i> Chat
                </Link>
              </li>
              <li className="nav-item mx-1">
                <Link className="nav-link" to="/documents">
                  <i className="bi bi-file-earmark-text me-1"></i> Documents
                </Link>
              </li>
            </ul>
          </div>
        </div>
      </nav>

      <main>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/chat" element={<ChatPage />} />
          <Route path="/documents" element={<DocumentsPage />} />
        </Routes>
      </main>
    </div>
  );
};

export default App;
