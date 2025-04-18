import React, { useState, useEffect, useCallback } from 'react';
import { useSearchParams } from 'react-router-dom';
import ConversationList from './ConversationList';
import MessageList from './MessageList';
import MessageInput from './MessageInput';
import {
  getConversations,
  createConversation,
  getMessages,
  deleteConversation,
  sendMessage,
  sendStreamingMessage
} from '../services/api';

const ChatPage = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const [conversations, setConversations] = useState([]);
  const [messages, setMessages] = useState([]);
  const [activeConversationId, setActiveConversationId] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingConversations, setLoadingConversations] = useState(false);
  const [error, setError] = useState(null);
  const [streamingEnabled, setStreamingEnabled] = useState(true);

  // Load conversations
  const loadConversations = useCallback(async () => {
    try {
      setLoadingConversations(true);
      const data = await getConversations();
      setConversations(data);
      return data;
    } catch (err) {
      console.error('Error loading conversations:', err);
      setError('Failed to load conversations');
      return [];
    } finally {
      setLoadingConversations(false);
    }
  }, []);

  // Load messages for a conversation
  const loadMessages = useCallback(async (conversationId) => {
    try {
      setLoading(true);
      const data = await getMessages(conversationId);
      setMessages(data);
    } catch (err) {
      console.error('Error loading messages:', err);
      setError('Failed to load messages');
    } finally {
      setLoading(false);
    }
  }, []);

  // Create a new conversation
  const handleNewConversation = async () => {
    try {
      setLoading(true);
      const newConversation = await createConversation();

      // Update conversations list
      await loadConversations();

      // Set as active conversation
      setActiveConversationId(newConversation.id);
      setSearchParams({ conversation_id: newConversation.id });

      // Clear messages
      setMessages([]);
    } catch (err) {
      console.error('Error creating conversation:', err);
      setError('Failed to create conversation');
    } finally {
      setLoading(false);
    }
  };

  // Select a conversation
  const handleSelectConversation = (conversationId) => {
    setActiveConversationId(conversationId);
    setSearchParams({ conversation_id: conversationId });
    loadMessages(conversationId);
  };

  // Delete a conversation
  const handleDeleteConversation = async (conversationId) => {
    try {
      setLoading(true);

      // Confirm deletion
      if (!window.confirm('Are you sure you want to delete this conversation?')) {
        setLoading(false);
        return;
      }

      // Delete the conversation
      await deleteConversation(conversationId);

      // Update conversations list
      const updatedConversations = await loadConversations();

      // If the deleted conversation was active, select another one
      if (conversationId === activeConversationId) {
        if (updatedConversations.length > 0) {
          // Select the first conversation
          const firstId = updatedConversations[0].id;
          setActiveConversationId(firstId);
          setSearchParams({ conversation_id: firstId });
          loadMessages(firstId);
        } else {
          // No conversations left, clear messages and active conversation
          setActiveConversationId(null);
          setSearchParams({});
          setMessages([]);
        }
      }
    } catch (err) {
      console.error('Error deleting conversation:', err);
      setError('Failed to delete conversation');
    } finally {
      setLoading(false);
    }
  };

  // Send a message
  const handleSendMessage = async (message) => {
    if (!activeConversationId) {
      setError('No active conversation');
      return;
    }

    // Add user message to the UI immediately
    const userMessage = { role: 'user', content: message };
    setMessages(prev => [...prev, userMessage]);

    // Set loading state
    setLoading(true);
    setError(null);

    try {
      if (streamingEnabled) {
        // Use streaming API
        let streamedContent = '';
        let streamedMessage = { role: 'assistant', content: '', references: [] };

        // Add an empty assistant message that will be updated
        setMessages(prev => [...prev, streamedMessage]);

        await sendStreamingMessage(
          activeConversationId,
          message,
          // On chunk received
          (chunk, fullContent) => {
            streamedContent = fullContent;
            setMessages(prev => {
              const newMessages = [...prev];
              newMessages[newMessages.length - 1] = {
                ...newMessages[newMessages.length - 1],
                content: streamedContent
              };
              return newMessages;
            });
          },
          // On complete
          (result) => {
            setMessages(prev => {
              const newMessages = [...prev];
              newMessages[newMessages.length - 1] = {
                role: 'assistant',
                content: result.message,
                references: result.references
              };
              return newMessages;
            });
            setLoading(false);
          },
          // On error
          (error) => {
            console.error('Error in streaming:', error);
            setError('Failed to get response');
            setLoading(false);
          }
        );
      } else {
        // Use regular API
        const response = await sendMessage(activeConversationId, message);

        // Add assistant response to messages
        const assistantMessage = {
          role: 'assistant',
          content: response.message,
          references: response.references
        };

        setMessages(prev => [...prev, assistantMessage]);
        setLoading(false);
      }
    } catch (err) {
      console.error('Error sending message:', err);
      setError('Failed to send message');
      setLoading(false);
    }
  };

  // Initialize from URL params
  useEffect(() => {
    const initializeChat = async () => {
      const conversationId = searchParams.get('conversation_id');

      // Load all conversations
      const conversationsData = await loadConversations();

      if (conversationId && conversationsData.some(c => c.id === parseInt(conversationId))) {
        // If URL has a valid conversation ID, load that conversation
        const id = parseInt(conversationId);
        setActiveConversationId(id);
        loadMessages(id);
      } else if (conversationsData.length > 0) {
        // Otherwise, load the first conversation
        const firstId = conversationsData[0].id;
        setActiveConversationId(firstId);
        setSearchParams({ conversation_id: firstId });
        loadMessages(firstId);
      } else {
        // If no conversations exist, create a new one
        handleNewConversation();
      }
    };

    initializeChat();
  }, [searchParams, loadConversations, loadMessages, setSearchParams]);

  return (
    <div className="row">
      <div className="col-md-3">
        <ConversationList
          conversations={conversations}
          activeConversationId={activeConversationId}
          onSelectConversation={handleSelectConversation}
          onNewConversation={handleNewConversation}
          onDeleteConversation={handleDeleteConversation}
          loading={loadingConversations}
        />
      </div>
      <div className="col-md-9">
        <div className="card">
          <div className="card-header d-flex justify-content-between align-items-center">
            <h5 className="card-title mb-0">Chat</h5>
            <div className="form-check form-switch">
              <input
                className="form-check-input"
                type="checkbox"
                id="streamingToggle"
                checked={streamingEnabled}
                onChange={() => setStreamingEnabled(!streamingEnabled)}
              />
              <label className="form-check-label" htmlFor="streamingToggle">
                Streaming
              </label>
            </div>
          </div>
          <div className="card-body p-0 d-flex flex-column" style={{ height: '600px' }}>
            {error && (
              <div className="alert alert-danger m-2">
                {error}
              </div>
            )}
            <MessageList messages={messages} loading={loading} />
            <MessageInput onSendMessage={handleSendMessage} disabled={loading || !activeConversationId} />
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatPage;
