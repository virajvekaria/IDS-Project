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
  const response = await api.get('/conversations');
  return response.data;
};

export const createConversation = async () => {
  const response = await api.post('/conversations', {});
  return response.data;
};

export const getMessages = async (conversationId) => {
  const response = await api.get(`/conversations/${conversationId}/messages`);
  return response.data;
};

export const deleteConversation = async (conversationId) => {
  const response = await api.delete(`/conversations/${conversationId}`);
  return response.data;
};

// Chat API
export const sendMessage = async (conversationId, message) => {
  const response = await api.post('/search/chat', {
    conversation_id: conversationId,
    message
  });
  return response.data;
};

// Streaming chat API
export const sendStreamingMessage = async (conversationId, message, onChunk, onComplete, onError) => {
  try {
    const response = await fetch('/search/chat?stream=true', {
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
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to send message');
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
  const response = await api.get('/documents');
  return response.data;
};

export default {
  getConversations,
  createConversation,
  getMessages,
  deleteConversation,
  sendMessage,
  sendStreamingMessage,
  getDocuments
};
