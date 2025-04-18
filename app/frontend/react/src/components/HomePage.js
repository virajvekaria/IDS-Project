import React from 'react';
import { Link } from 'react-router-dom';

const HomePage = () => {
  return (
    <div>
      <div className="hero-section">
        <div className="container py-5">
          <h1 className="hero-title">Document Intelligence Search System</h1>
          <p className="hero-subtitle">
            A powerful AI-powered solution for searching and querying your documents
          </p>
          <div className="d-flex gap-3 justify-content-center">
            <Link to="/documents" className="btn btn-light btn-lg">
              <i className="bi bi-file-earmark-text me-2"></i>
              Manage Documents
            </Link>
            <Link to="/chat" className="btn btn-light btn-lg">
              <i className="bi bi-chat-dots-fill me-2"></i>
              Start Chatting
            </Link>
          </div>
        </div>
      </div>

      <div className="container mt-5">
        <div className="row mb-5">
          <div className="col-12 text-center">
            <h2 className="mb-4">How It Works</h2>
            <p className="lead text-muted">Unlock the power of your documents with AI-powered search and chat</p>
          </div>
        </div>

        <div className="row g-4">
          <div className="col-md-4">
            <div className="card feature-card h-100">
              <div className="card-body text-center">
                <div className="feature-icon">
                  <i className="bi bi-cloud-upload"></i>
                </div>
                <h3 className="h5 mb-3">Upload Documents</h3>
                <p className="text-muted">Upload your PDF documents to the system for processing and indexing.</p>
                <Link to="/documents" className="btn btn-sm btn-outline-primary mt-auto">
                  Upload Now
                </Link>
              </div>
            </div>
          </div>

          <div className="col-md-4">
            <div className="card feature-card h-100">
              <div className="card-body text-center">
                <div className="feature-icon">
                  <i className="bi bi-search"></i>
                </div>
                <h3 className="h5 mb-3">Intelligent Processing</h3>
                <p className="text-muted">Documents are automatically processed, chunked, and indexed for semantic search.</p>
              </div>
            </div>
          </div>

          <div className="col-md-4">
            <div className="card feature-card h-100">
              <div className="card-body text-center">
                <div className="feature-icon">
                  <i className="bi bi-chat-square-text"></i>
                </div>
                <h3 className="h5 mb-3">Chat with Your Documents</h3>
                <p className="text-muted">Ask questions and get answers directly from your document content.</p>
                <Link to="/chat" className="btn btn-sm btn-outline-primary mt-auto">
                  Start Chatting
                </Link>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HomePage;
