import axios from 'axios';

// Create axios instance with base URL
const api = axios.create({
  baseURL: 'http://localhost:8000',
  headers: {
    'Content-Type': 'application/json'
  }
});

// Conversations API
export const getConversations = async () => {
  const response = await api.get('/react-api/conversations');
  return response.data;
};

export const createConversation = async () => {
  const response = await api.post('/conversations', {});
  return response.data;
};

export const getMessages = async (conversationId) => {
  const response = await api.get(`/react-api/conversations/${conversationId}/messages`);
  return response.data;
};

export const updateMessage = async (conversationId, messageId, content) => {
  const response = await api.put(`/react-api/conversations/${conversationId}/messages/${messageId}`, {
    content
  });
  return response.data;
};

export const deleteConversation = async (conversationId) => {
  const response = await api.delete(`/conversations/${conversationId}`);
  return response.data;
};

// Chat API
export const sendMessage = async (conversationId, message) => {
  const response = await api.post('/react-api/chat', {
    conversation_id: conversationId,
    message
  });
  return response.data;
};

// Streaming chat API
export const sendStreamingMessage = async (conversationId, message, onChunk, onComplete, onError) => {
  try {
    const response = await fetch('/react-api/chat?stream=true', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream'
      },
      body: JSON.stringify({
        conversation_id: conversationId,
        message
      })
    });

    if (!response.ok) {
      try {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to send message');
      } catch (jsonError) {
        // If the response is not JSON, use the status text
        throw new Error(`Failed to send message: ${response.status} ${response.statusText}`);
      }
    }

    // Check if we got a streaming response
    const contentType = response.headers.get('content-type');
    if (!contentType || !contentType.includes('text/event-stream')) {
      // Fall back to regular response
      const result = await response.json();
      onComplete(result);
      return;
    }

    // Process the streaming response
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let references = [];
    let messageContent = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      buffer += chunk;

      // Process complete SSE messages
      const messages = buffer.split('\n\n');
      buffer = messages.pop() || '';

      for (const message of messages) {
        if (message.startsWith('data: ')) {
          try {
            const data = JSON.parse(message.substring(6));

            if (data.references) {
              // This is the initial message with references
              references = data.references;
            } else if (data.content) {
              // This is a content chunk
              messageContent += data.content;
              onChunk(data.content, messageContent);
            } else if (data.done) {
              // This is the end of the stream
              onComplete({
                message: messageContent,
                references
              });
            }
          } catch (e) {
            console.error('Error parsing streaming data:', e, message);
          }
        }
      }
    }
  } catch (error) {
    onError(error);
  }
};

// Documents API
export const getDocuments = async () => {
  const response = await api.get('/react-api/documents');
  return response.data;
};

export const uploadDocument = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch('/react-api/documents/upload', {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    try {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to upload document');
    } catch (jsonError) {
      // If the response is not JSON, use the status text
      throw new Error(`Upload failed: ${response.status} ${response.statusText}`);
    }
  }

  return response.json();
};

export const deleteDocument = async (documentId) => {
  const response = await api.delete(`/react-api/documents/${documentId}`);
  return response.data;
};

export default {
  getConversations,
  createConversation,
  getMessages,
  updateMessage,
  deleteConversation,
  sendMessage,
  sendStreamingMessage,
  getDocuments,
  uploadDocument,
  deleteDocument
};
