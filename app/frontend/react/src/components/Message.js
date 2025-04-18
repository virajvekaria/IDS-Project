import React from 'react';

const Message = ({ role, content, references = [] }) => {
  // Format content with line breaks
  const formattedContent = content.replace(/\n/g, '<br>');

  return (
    <div className={`message message-${role}`}>
      <div dangerouslySetInnerHTML={{ __html: formattedContent }} />
      
      {references && references.length > 0 && (
        <div className="message-references">
          <span>Sources: </span>
          {references.map((ref, index) => (
            <span 
              key={index} 
              className="reference" 
              title={ref.text ? ref.text.substring(0, 100) + '...' : ''}
            >
              {index > 0 && ', '}
              Page {ref.page_number}
            </span>
          ))}
        </div>
      )}
    </div>
  );
};

export default Message;
