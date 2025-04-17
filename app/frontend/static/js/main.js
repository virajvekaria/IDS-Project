// Main JavaScript for DISS

document.addEventListener('DOMContentLoaded', function() {
    // Load documents on page load
    loadDocuments();

    // Set up event listeners
    document.getElementById('upload-form').addEventListener('submit', uploadDocument);
    document.getElementById('process-button').addEventListener('click', processDocument);
});

// Load documents from API
async function loadDocuments() {
    try {
        const response = await fetch('/documents');
        const documents = await response.json();
        
        const tableBody = document.getElementById('documents-table');
        tableBody.innerHTML = '';
        
        if (documents.length === 0) {
            tableBody.innerHTML = '<tr><td colspan="7" class="text-center">No documents found</td></tr>';
            return;
        }
        
        documents.forEach(doc => {
            const row = document.createElement('tr');
            
            // Format file size
            const fileSize = formatFileSize(doc.file_size);
            
            // Determine status
            let statusText = 'Not processed';
            let statusClass = 'status-pending';
            
            if (doc.processed && doc.indexed) {
                statusText = 'Indexed';
                statusClass = 'status-completed';
            } else if (doc.processed) {
                statusText = 'Processed';
                statusClass = 'status-processing';
            }
            
            row.innerHTML = `
                <td>${doc.id}</td>
                <td>${doc.filename}</td>
                <td>${doc.file_type}</td>
                <td>${fileSize}</td>
                <td>${doc.page_count}</td>
                <td><span class="document-status ${statusClass}">${statusText}</span></td>
                <td>
                    <button class="btn btn-sm btn-primary process-btn" data-id="${doc.id}">Process</button>
                    <button class="btn btn-sm btn-danger delete-btn" data-id="${doc.id}">Delete</button>
                </td>
            `;
            
            tableBody.appendChild(row);
        });
        
        // Add event listeners to buttons
        document.querySelectorAll('.process-btn').forEach(btn => {
            btn.addEventListener('click', function() {
                const docId = this.getAttribute('data-id');
                showProcessModal(docId);
            });
        });
        
        document.querySelectorAll('.delete-btn').forEach(btn => {
            btn.addEventListener('click', function() {
                const docId = this.getAttribute('data-id');
                deleteDocument(docId);
            });
        });
        
    } catch (error) {
        console.error('Error loading documents:', error);
        alert('Failed to load documents. Please try again.');
    }
}

// Upload document
async function uploadDocument(event) {
    event.preventDefault();
    
    const formData = new FormData(document.getElementById('upload-form'));
    const statusDiv = document.getElementById('upload-status');
    
    statusDiv.classList.remove('d-none', 'alert-success', 'alert-danger');
    statusDiv.classList.add('alert-info');
    statusDiv.innerHTML = '<div class="spinner-border" role="status"></div> Uploading document...';
    
    try {
        const response = await fetch('/documents', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Upload failed');
        }
        
        const result = await response.json();
        
        statusDiv.classList.remove('alert-info');
        statusDiv.classList.add('alert-success');
        statusDiv.textContent = `Document "${result.filename}" uploaded successfully!`;
        
        // Reset form
        document.getElementById('upload-form').reset();
        
        // Reload documents
        loadDocuments();
        
    } catch (error) {
        console.error('Error uploading document:', error);
        
        statusDiv.classList.remove('alert-info');
        statusDiv.classList.add('alert-danger');
        statusDiv.textContent = `Upload failed: ${error.message}`;
    }
}

// Show process modal
function showProcessModal(docId) {
    document.getElementById('document-id').value = docId;
    
    // Show modal
    const modal = new bootstrap.Modal(document.getElementById('processModal'));
    modal.show();
}

// Process document
async function processDocument() {
    const docId = document.getElementById('document-id').value;
    const chunkSize = document.getElementById('chunk-size').value;
    const chunkOverlap = document.getElementById('chunk-overlap').value;
    const indexName = document.getElementById('index-name').value;
    
    // Hide modal
    const modalElement = document.getElementById('processModal');
    const modal = bootstrap.Modal.getInstance(modalElement);
    modal.hide();
    
    // Show processing status
    const statusDiv = document.getElementById('upload-status');
    statusDiv.classList.remove('d-none', 'alert-success', 'alert-danger');
    statusDiv.classList.add('alert-info');
    statusDiv.innerHTML = '<div class="spinner-border" role="status"></div> Processing document...';
    
    try {
        const response = await fetch('/documents/process', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                document_id: parseInt(docId),
                chunk_size: parseInt(chunkSize),
                chunk_overlap: parseInt(chunkOverlap),
                index_name: indexName
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Processing failed');
        }
        
        const result = await response.json();
        
        statusDiv.classList.remove('alert-info');
        statusDiv.classList.add('alert-success');
        statusDiv.textContent = `Document processing started. Check status for updates.`;
        
        // Reload documents after a delay
        setTimeout(loadDocuments, 2000);
        
    } catch (error) {
        console.error('Error processing document:', error);
        
        statusDiv.classList.remove('alert-info');
        statusDiv.classList.add('alert-danger');
        statusDiv.textContent = `Processing failed: ${error.message}`;
    }
}

// Delete document
async function deleteDocument(docId) {
    if (!confirm('Are you sure you want to delete this document?')) {
        return;
    }
    
    try {
        const response = await fetch(`/documents/${docId}`, {
            method: 'DELETE'
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Delete failed');
        }
        
        // Reload documents
        loadDocuments();
        
    } catch (error) {
        console.error('Error deleting document:', error);
        alert(`Failed to delete document: ${error.message}`);
    }
}

// Format file size
function formatFileSize(bytes) {
    if (bytes < 1024) {
        return bytes + ' B';
    } else if (bytes < 1024 * 1024) {
        return (bytes / 1024).toFixed(2) + ' KB';
    } else if (bytes < 1024 * 1024 * 1024) {
        return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
    } else {
        return (bytes / (1024 * 1024 * 1024)).toFixed(2) + ' GB';
    }
}
