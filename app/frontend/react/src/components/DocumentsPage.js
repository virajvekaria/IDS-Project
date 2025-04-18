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
    <div className="documents-container">
      <div className="d-flex justify-content-between align-items-center mb-4">
        <div className="d-flex align-items-center">
          <h2 className="mb-0 me-3">Document Library</h2>
          <button
            className="btn btn-outline-primary btn-sm me-2"
            onClick={() => fetchDocuments(true)}
            disabled={loading}
          >
            {loading ? (
              <>
                <span className="spinner-border spinner-border-sm me-1" role="status" aria-hidden="true"></span>
                Refreshing...
              </>
            ) : (
              <>
                <i className="bi bi-arrow-clockwise me-1"></i> Refresh
              </>
            )}
          </button>
        </div>
        <button
          className="btn btn-primary"
          onClick={toggleUploadForm}
        >
          {showUploadForm ? (
            <>
              <i className="bi bi-x-circle me-1"></i> Cancel Upload
            </>
          ) : (
            <>
              <i className="bi bi-cloud-upload me-1"></i> Upload Document
            </>
          )}
        </button>
      </div>

      {uploadMessage && (
        <div className={`alert ${uploadMessage.includes('successfully') ? 'alert-success' : 'alert-info'} alert-dismissible fade show mb-4`} role="alert">
          <i className={`bi ${uploadMessage.includes('successfully') ? 'bi-check-circle' : 'bi-info-circle'} me-2`}></i>
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
        <div className="upload-form mb-4">
          <div className="mb-3">
            <h5><i className="bi bi-cloud-arrow-up me-2"></i>Upload New Document</h5>
            <p className="text-muted">Select a PDF file to upload and process.</p>
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
              ) : (
                <>
                  <i className="bi bi-cloud-arrow-up-fill me-1"></i> Upload Document
                </>
              )}
            </button>
          </div>
        </div>
      )}

      {loading ? (
        <div className="text-center py-5">
          <div className="spinner-border text-primary" role="status">
            <span className="visually-hidden">Loading...</span>
          </div>
          <p className="mt-3 text-muted">Loading documents...</p>
        </div>
      ) : error ? (
        <div className="alert alert-danger">
          <i className="bi bi-exclamation-triangle-fill me-2"></i>
          {error}
        </div>
      ) : documents.length === 0 ? (
        <div className="text-center py-5">
          <i className="bi bi-file-earmark-text" style={{ fontSize: '3rem', color: 'var(--gray-300)' }}></i>
          <h4 className="mt-3">No documents found</h4>
          <p className="text-muted">Upload a document to get started.</p>
          <p className="text-muted small">Documents are stored in the data/documents folder and processed automatically.</p>
          <p className="text-muted small">You can also place PDF files in the PDFs folder and they will be processed on startup.</p>
        </div>
      ) : (
        <div className="card">
          <div className="card-header">
            <h5 className="mb-0">Your Documents</h5>
          </div>
          <div className="table-responsive">
            <table className="table table-hover mb-0">
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
                    <td>
                      <div className="d-flex align-items-center">
                        <i className="bi bi-file-earmark-pdf text-danger me-2"></i>
                        {doc.filename}
                      </div>
                    </td>
                    <td><span className="badge bg-secondary">{doc.file_type.toUpperCase()}</span></td>
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
                        className="btn btn-outline-danger btn-sm"
                        onClick={() => handleDeleteConfirm(doc)}
                      >
                        <i className="bi bi-trash me-1"></i>
                        Delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Delete Confirmation Modal */}
      {documentToDelete && (
        <>
          <div className="modal fade show" style={{ display: 'block' }} tabIndex="-1">
            <div className="modal-dialog modal-dialog-centered">
              <div className="modal-content">
                <div className="modal-header">
                  <h5 className="modal-title">
                    <i className="bi bi-exclamation-triangle-fill text-danger me-2"></i>
                    Confirm Delete
                  </h5>
                  <button type="button" className="btn-close" onClick={handleDeleteCancel}></button>
                </div>
                <div className="modal-body">
                  <p>Are you sure you want to delete the document <strong>"{documentToDelete.filename}"</strong>?</p>
                  <div className="alert alert-warning">
                    <i className="bi bi-info-circle me-2"></i>
                    This action will remove the document and all associated data. This cannot be undone.
                  </div>
                </div>
                <div className="modal-footer">
                  <button type="button" className="btn btn-outline-secondary" onClick={handleDeleteCancel}>
                    <i className="bi bi-x me-1"></i> Cancel
                  </button>
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
                    ) : (
                      <>
                        <i className="bi bi-trash me-1"></i> Delete Document
                      </>
                    )}
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
