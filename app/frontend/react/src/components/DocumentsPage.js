import React, { useState, useEffect } from 'react';
import { getDocuments } from '../services/api';

const DocumentsPage = () => {
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchDocuments = async () => {
      try {
        setLoading(true);
        const data = await getDocuments();
        setDocuments(data);
        setError(null);
      } catch (err) {
        setError('Failed to load documents. Please try again later.');
        console.error('Error fetching documents:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchDocuments();
  }, []);

  const getStatusClass = (doc) => {
    if (!doc.processed) return 'status-pending';
    if (!doc.indexed) return 'status-processing';
    return 'status-completed';
  };

  const getStatusText = (doc) => {
    if (!doc.processed) return 'Pending';
    if (!doc.indexed) return 'Processing';
    return 'Completed';
  };

  return (
    <div className="row">
      <div className="col-12">
        <div className="card">
          <div className="card-header d-flex justify-content-between align-items-center">
            <h5 className="card-title mb-0">Documents</h5>
            <button className="btn btn-primary btn-sm">Upload Document</button>
          </div>
          <div className="card-body">
            {loading ? (
              <div className="text-center py-4">
                <div className="spinner-border" role="status">
                  <span className="visually-hidden">Loading...</span>
                </div>
                <p className="mt-2">Loading documents...</p>
              </div>
            ) : error ? (
              <div className="alert alert-danger">{error}</div>
            ) : documents.length === 0 ? (
              <div className="text-center py-4">
                <p className="text-muted">No documents found. Upload a document to get started.</p>
              </div>
            ) : (
              <div className="table-responsive">
                <table className="table table-hover">
                  <thead>
                    <tr>
                      <th>Filename</th>
                      <th>Type</th>
                      <th>Size</th>
                      <th>Pages</th>
                      <th>Status</th>
                      <th>Uploaded</th>
                    </tr>
                  </thead>
                  <tbody>
                    {documents.map((doc) => (
                      <tr key={doc.id}>
                        <td>{doc.filename}</td>
                        <td>{doc.file_type}</td>
                        <td>{Math.round(doc.file_size / 1024)} KB</td>
                        <td>{doc.page_count}</td>
                        <td>
                          <span className={`document-status ${getStatusClass(doc)}`}>
                            {getStatusText(doc)}
                          </span>
                        </td>
                        <td>{new Date(doc.created_at).toLocaleString()}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default DocumentsPage;
