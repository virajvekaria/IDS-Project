import React, { useEffect, useRef } from 'react';
import Message from './Message';

const MessageList = ({ messages, loading }) => {
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  return (
    <div className="chat-messages">
      {messages.length === 0 ? (
        <div className="text-center text-muted my-4">
          No messages yet. Start a conversation!
        </div>
      ) : (
        <>
          {messages.map((msg, index) => (
            <Message
              key={index}
              role={msg.role}
              content={msg.content}
              references={msg.references}
            />
          ))}
          {loading && (
            <div className="message message-assistant">
              <div className="spinner" /> Thinking...
            </div>
          )}
          <div ref={messagesEndRef} />
        </>
      )}
    </div>
  );
};

export default MessageList;
