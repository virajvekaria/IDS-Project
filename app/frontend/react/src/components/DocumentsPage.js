import React, { useState, useEffect } from 'react';
import { getDocuments, uploadDocument, deleteDocument } from '../services/api';

const DocumentsPage = () => {
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [uploadMessage, setUploadMessage] = useState(null);
  const [showUploadForm, setShowUploadForm] = useState(false);
  const [uploadFile, setUploadFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [documentToDelete, setDocumentToDelete] = useState(null);
  const [deleting, setDeleting] = useState(false);

  const fileInputRef = React.useRef(null);

  const fetchDocuments = async (isInitialLoad = false) => {
    if (isInitialLoad) {
      setLoading(true);
    }

    try {
      console.log('Fetching documents...');

      // Use the new all-documents endpoint
      const response = await fetch('/react-api/all-documents');

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log('All documents fetched:', data);
      setDocuments(data);
      setError(null);
    } catch (err) {
      console.error('Error fetching documents:', err);
      if (isInitialLoad) {
        setError('Failed to load documents. Please try again later.');
        setDocuments([]);
      }
    } finally {
      if (isInitialLoad) {
        setLoading(false);
      }
    }
  };

  // Initial fetch only
  useEffect(() => {
    // Initial load with loading indicator
    fetchDocuments(true);
  }, []); // Empty dependency array to run only once

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

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setUploadFile(file);
    }
  };

  const handleFileUpload = async () => {
    if (!uploadFile) {
      setUploadMessage('Please select a file to upload');
      return;
    }

    try {
      setUploading(true);
      setUploadMessage('Uploading document...');

      await uploadDocument(uploadFile);

      // Refresh the documents list
      await fetchDocuments(false);

      // Reset the form
      setUploadFile(null);
      setShowUploadForm(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }

      setUploadMessage('Document uploaded successfully!');
    } catch (err) {
      console.error('Error uploading document:', err);
      setUploadMessage(`Failed to upload document: ${err.message}`);
    } finally {
      setUploading(false);
    }
  };

  const toggleUploadForm = () => {
    setShowUploadForm(!showUploadForm);
    if (!showUploadForm) {
      setUploadFile(null);
      setUploadMessage(null);
    }
  };

  const handleDeleteConfirm = (document) => {
    setDocumentToDelete(document);
  };

  const handleDeleteCancel = () => {
    setDocumentToDelete(null);
  };

  const handleDeleteDocument = async () => {
    if (!documentToDelete) return;

    try {
      setDeleting(true);
      await deleteDocument(documentToDelete.id);

      // Update the documents list
      await fetchDocuments(false);

      // Close the modal first
      setDocumentToDelete(null);
      setDeleting(false);

      // Then show the success message
      setUploadMessage(`Document "${documentToDelete.filename}" deleted successfully.`);
    } catch (err) {
      console.error('Error deleting document:', err);

      // Close the modal first
      setDocumentToDelete(null);
      setDeleting(false);

      // Then show the error message
      setUploadMessage(`Failed to delete document: ${err.message}`);
    }
  };

  return (
    <div className="row">
      <div className="col-12">
        <div className="card">
          <div className="card-header d-flex justify-content-between align-items-center">
            <div className="d-flex align-items-center">
              <h5 className="card-title mb-0 me-3">Documents</h5>
              <button
                className="btn btn-outline-secondary btn-sm me-2"
                onClick={() => fetchDocuments(true)}
                disabled={loading}
              >
                {loading ? (
                  <>
                    <span className="spinner-border spinner-border-sm me-1" role="status" aria-hidden="true"></span>
                    Refreshing...
                  </>
                ) : 'Refresh'}
              </button>
            </div>
            <button
              className="btn btn-primary btn-sm"
              onClick={toggleUploadForm}
            >
              {showUploadForm ? 'Cancel Upload' : 'Upload Document'}
            </button>
          </div>
          {uploadMessage && (
            <div className="alert alert-info alert-dismissible fade show m-2" role="alert">
              {uploadMessage}
              <button
                type="button"
                className="btn-close"
                onClick={() => setUploadMessage(null)}
                aria-label="Close"
              ></button>
            </div>
          )}
          {showUploadForm && (
            <div className="card-body border-bottom">
              <div className="mb-3">
                <label htmlFor="documentFile" className="form-label">Select PDF Document</label>
                <input
                  type="file"
                  className="form-control"
                  id="documentFile"
                  accept=".pdf"
                  onChange={handleFileChange}
                  ref={fileInputRef}
                  disabled={uploading}
                />
                <div className="form-text">Only PDF files are supported. Maximum file size: 10MB.</div>
              </div>
              <button
                className="btn btn-success"
                onClick={handleFileUpload}
                disabled={!uploadFile || uploading}
              >
                {uploading ? (
                  <>
                    <span className="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
                    Uploading...
                  </>
                ) : 'Upload'}
              </button>
            </div>
          )}
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
                <p className="text-muted small">Documents are stored in the data/documents folder and processed automatically.</p>
                <p className="text-muted small">You can also place PDF files in the PDFs folder and they will be processed on startup.</p>
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
                      <th>Actions</th>
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
                        <td>
                          <button
                            className="btn btn-danger btn-sm"
                            onClick={() => handleDeleteConfirm(doc)}
                          >
                            Delete
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Delete Confirmation Modal */}
      {documentToDelete && (
        <>
          <div className="modal fade show" style={{ display: 'block' }} tabIndex="-1">
            <div className="modal-dialog">
              <div className="modal-content">
                <div className="modal-header">
                  <h5 className="modal-title">Confirm Delete</h5>
                  <button type="button" className="btn-close" onClick={handleDeleteCancel}></button>
                </div>
                <div className="modal-body">
                  <p>Are you sure you want to delete the document "{documentToDelete.filename}"?</p>
                  <p className="text-danger">This action cannot be undone.</p>
                </div>
                <div className="modal-footer">
                  <button type="button" className="btn btn-secondary" onClick={handleDeleteCancel}>Cancel</button>
                  <button
                    type="button"
                    className="btn btn-danger"
                    onClick={handleDeleteDocument}
                    disabled={deleting}
                  >
                    {deleting ? (
                      <>
                        <span className="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
                        Deleting...
                      </>
                    ) : 'Delete'}
                  </button>
                </div>
              </div>
            </div>
          </div>
          <div className="modal-backdrop fade show" onClick={handleDeleteCancel}></div>
        </>
      )}
    </div>
  );
};

export default DocumentsPage;
