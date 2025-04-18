import React from 'react';

const ConversationList = ({
  conversations,
  activeConversationId,
  onSelectConversation,
  onNewConversation,
  onDeleteConversation,
  loading
}) => {
  return (
    <div className="card h-100">
      <div className="card-header d-flex justify-content-between align-items-center">
        <h5 className="card-title mb-0">Conversations</h5>
        <button
          className="btn btn-primary btn-sm"
          onClick={onNewConversation}
          disabled={loading}
        >
          New
        </button>
      </div>
      <div className="card-body p-0">
        {loading ? (
          <div className="text-center py-4">
            <div className="spinner-border" role="status">
              <span className="visually-hidden">Loading...</span>
            </div>
          </div>
        ) : conversations.length === 0 ? (
          <div className="text-center py-4">
            <p className="text-muted">No conversations yet.</p>
            <button
              className="btn btn-outline-primary btn-sm"
              onClick={onNewConversation}
            >
              Start a new conversation
            </button>
          </div>
        ) : (
          <ul className="list-group list-group-flush conversation-list">
            {conversations.map((conv) => (
              <li
                key={conv.id}
                className={`conversation-item ${activeConversationId === conv.id ? 'active' : ''}`}
              >
                <div
                  className="d-flex justify-content-between align-items-center"
                  onClick={() => onSelectConversation(conv.id)}
                  style={{ cursor: 'pointer', flex: 1 }}
                >
                  <span>{conv.title || `Conversation ${conv.id}`}</span>
                  <div className="d-flex align-items-center">
                    <small className="text-muted me-2">
                      {new Date(conv.created_at).toLocaleDateString()}
                    </small>
                    <button
                      className="btn btn-sm btn-outline-danger"
                      onClick={(e) => {
                        e.stopPropagation();
                        onDeleteConversation(conv.id);
                      }}
                      title="Delete conversation"
                    >
                      Delete
                    </button>
                  </div>
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
};

export default ConversationList;
