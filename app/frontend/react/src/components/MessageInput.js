import React, { useState } from 'react';

const MessageInput = ({ onSendMessage, disabled }) => {
  const [message, setMessage] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (message.trim() === '') return;
    
    onSendMessage(message);
    setMessage('');
  };

  return (
    <div className="chat-input">
      <form onSubmit={handleSubmit}>
        <div className="input-group">
          <input
            type="text"
            className="form-control"
            placeholder="Type your message..."
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            disabled={disabled}
            required
          />
          <button 
            className="btn btn-primary" 
            type="submit"
            disabled={disabled || message.trim() === ''}
          >
            Send
          </button>
        </div>
      </form>
    </div>
  );
};

export default MessageInput;
