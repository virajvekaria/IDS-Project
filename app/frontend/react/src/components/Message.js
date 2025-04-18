import React, { useState } from 'react';

const Message = ({ role, content, references = [] }) => {
  const [expandedReference, setExpandedReference] = useState(null);

  // Format content with line breaks
  const formattedContent = content.replace(/\n/g, '<br>');

  // Always show references for assistant messages with references
  const shouldShowReferences = role === 'assistant' && references && references.length > 0;

  // Toggle reference expansion
  const toggleReference = (index) => {
    if (expandedReference === index) {
      setExpandedReference(null);
    } else {
      setExpandedReference(index);
    }
  };

  return (
    <div className={`message message-${role}`}>
      <div dangerouslySetInnerHTML={{ __html: formattedContent }} />

      {shouldShowReferences && (
        <div className="message-references">
          <div className="sources-header">
            <strong>Sources Used:</strong>
          </div>
          <div className="reference-list">
            {references.map((ref, index) => (
              <div key={index} className="reference-item">
                <span
                  className={`reference ${expandedReference === index ? 'expanded' : ''}`}
                  onClick={() => toggleReference(index)}
                >
                  {index > 0 && ' '}
                  Page {ref.page_number} {expandedReference === index ? '▼' : '▶'}
                </span>

                {expandedReference === index && ref.text && (
                  <div className="reference-content">
                    <div className="reference-text">{ref.text}</div>
                    <div className="reference-source">From: {ref.source_doc || 'Unknown source'}</div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default Message;
