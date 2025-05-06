import React, { useState } from 'react';
import './Input.css';

const Input = ({ onSendMessage, disabled }) => {
  const [message, setMessage] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (message.trim() && !disabled) {
      onSendMessage(message);
      setMessage('');
    }
  };

  return (
    <form className="input-form" onSubmit={handleSubmit}>
      <input
        type="text"
        placeholder="Ask a question about neonatal healthcare..."
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        disabled={disabled}
        className="message-input"
      />
      <button
        type="submit"
        disabled={!message.trim() || disabled}
        className="send-button"
      >
        Send
      </button>
    </form>
  );
};

export default Input;
