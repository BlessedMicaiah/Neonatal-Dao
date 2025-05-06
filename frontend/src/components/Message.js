import React from 'react';
import './Message.css';

const Message = ({ message }) => {
  const { text, sender, timestamp, isError } = message;
  const formattedTime = new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  
  return (
    <div className={`message ${sender}-message ${isError ? 'error-message' : ''}`}>
      <div className="message-content">
        <span className="message-text">{text}</span>
        <span className="message-time">{formattedTime}</span>
      </div>
    </div>
  );
};

export default Message;
