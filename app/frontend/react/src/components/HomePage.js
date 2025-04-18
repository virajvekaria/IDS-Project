import React from 'react';
import { Link } from 'react-router-dom';

const HomePage = () => {
  return (
    <div className="jumbotron">
      <h1 className="display-4">Document Intelligence Search System</h1>
      <p className="lead">
        Welcome to DISS, a powerful system for searching and querying your documents using AI.
      </p>
      <hr className="my-4" />
      <p>
        Upload your documents, process them, and start chatting with your data.
      </p>
      <div className="d-flex gap-2">
        <Link to="/documents" className="btn btn-primary">
          Manage Documents
        </Link>
        <Link to="/chat" className="btn btn-success">
          Start Chatting
        </Link>
      </div>
    </div>
  );
};

export default HomePage;
