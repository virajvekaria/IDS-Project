// Chat JavaScript for DISS

document.addEventListener('DOMContentLoaded', function() {
    // Get conversation ID from URL if present
    const urlParams = new URLSearchParams(window.location.search);
    const conversationId = urlParams.get('conversation_id');
    
    // Initialize chat
    initChat(conversationId);
    
    // Set up event listeners
    document.getElementById('chat-form').addEventListener('submit', sendMessage);
    document.getElementById('new-conversation-btn').addEventListener('click', createNewConversation);
});

// Global variables
let currentConversationId = null;

// Initialize chat
async function initChat(conversationId) {
    // Load conversations
    await loadConversations();
    
    if (conversationId) {
        // Load specific conversation
        currentConversationId = parseInt(conversationId);
        await loadMessages(currentConversationId);
        
        // Highlight active conversation
        const conversationItems = document.querySelectorAll('.conversation-item');
        conversationItems.forEach(item => {
            if (parseInt(item.getAttribute('data-id')) === currentConversationId) {
                item.classList.add('active');
            }
        });
    } else if (document.querySelector('.conversation-item')) {
        // Load first conversation if available
        const firstConversation = document.querySelector('.conversation-item');
        currentConversationId = parseInt(firstConversation.getAttribute('data-id'));
        firstConversation.classList.add('active');
        await loadMessages(currentConversationId);
    } else {
        // Create new conversation if none exist
        await createNewConversation();
    }
}

// Load conversations
async function loadConversations() {
    try {
        const response = await fetch('/conversations');
        const conversations = await response.json();
        
        const conversationsList = document.getElementById('conversations-list');
        conversationsList.innerHTML = '';
        
        if (conversations.length === 0) {
            return;
        }
        
        conversations.forEach(conv => {
            const item = document.createElement('li');
            item.className = 'list-group-item conversation-item';
            item.setAttribute('data-id', conv.id);
            
            const title = conv.title || `Conversation ${conv.id}`;
            const date = new Date(conv.created_at).toLocaleDateString();
            
            item.innerHTML = `
                <div class="d-flex justify-content-between align-items-center">
                    <span>${title}</span>
                    <small class="text-muted">${date}</small>
                </div>
            `;
            
            item.addEventListener('click', function() {
                // Remove active class from all conversations
                document.querySelectorAll('.conversation-item').forEach(item => {
                    item.classList.remove('active');
                });
                
                // Add active class to clicked conversation
                this.classList.add('active');
                
                // Load messages for this conversation
                const convId = parseInt(this.getAttribute('data-id'));
                currentConversationId = convId;
                loadMessages(convId);
                
                // Update URL
                const url = new URL(window.location);
                url.searchParams.set('conversation_id', convId);
                window.history.pushState({}, '', url);
            });
            
            conversationsList.appendChild(item);
        });
        
    } catch (error) {
        console.error('Error loading conversations:', error);
    }
}

// Load messages for a conversation
async function loadMessages(conversationId) {
    try {
        const response = await fetch(`/conversations/${conversationId}/messages`);
        const messages = await response.json();
        
        const chatMessages = document.getElementById('chat-messages');
        chatMessages.innerHTML = '';
        
        if (messages.length === 0) {
            chatMessages.innerHTML = '<div class="text-center text-muted my-4">No messages yet. Start a conversation!</div>';
            return;
        }
        
        messages.forEach(msg => {
            addMessageToChat(msg.role, msg.content);
        });
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
    } catch (error) {
        console.error('Error loading messages:', error);
    }
}

// Create new conversation
async function createNewConversation() {
    try {
        const response = await fetch('/conversations', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({})
        });
        
        const conversation = await response.json();
        
        // Update current conversation ID
        currentConversationId = conversation.id;
        
        // Reload conversations
        await loadConversations();
        
        // Highlight active conversation
        const conversationItems = document.querySelectorAll('.conversation-item');
        conversationItems.forEach(item => {
            if (parseInt(item.getAttribute('data-id')) === currentConversationId) {
                item.classList.add('active');
            }
        });
        
        // Clear chat messages
        document.getElementById('chat-messages').innerHTML = '<div class="text-center text-muted my-4">New conversation started. Type a message to begin!</div>';
        
        // Update URL
        const url = new URL(window.location);
        url.searchParams.set('conversation_id', currentConversationId);
        window.history.pushState({}, '', url);
        
    } catch (error) {
        console.error('Error creating conversation:', error);
    }
}

// Send message
async function sendMessage(event) {
    event.preventDefault();
    
    const messageInput = document.getElementById('message-input');
    const message = messageInput.value.trim();
    
    if (!message) {
        return;
    }
    
    if (!currentConversationId) {
        alert('No active conversation. Please create a new one.');
        return;
    }
    
    // Add user message to chat
    addMessageToChat('user', message);
    
    // Clear input
    messageInput.value = '';
    
    // Add loading message
    const chatMessages = document.getElementById('chat-messages');
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'message message-assistant';
    loadingDiv.innerHTML = '<div class="spinner-border" role="status"></div> Thinking...';
    chatMessages.appendChild(loadingDiv);
    
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    try {
        const response = await fetch('/search/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                conversation_id: currentConversationId,
                message: message
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to send message');
        }
        
        const result = await response.json();
        
        // Remove loading message
        chatMessages.removeChild(loadingDiv);
        
        // Add assistant message to chat
        addMessageToChat('assistant', result.message, result.references);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
    } catch (error) {
        console.error('Error sending message:', error);
        
        // Remove loading message
        chatMessages.removeChild(loadingDiv);
        
        // Add error message
        addMessageToChat('assistant', `Error: ${error.message}`);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
}

// Add message to chat
function addMessageToChat(role, content, references = []) {
    const chatMessages = document.getElementById('chat-messages');
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message message-${role}`;
    
    // Format content with line breaks
    const formattedContent = content.replace(/\n/g, '<br>');
    messageDiv.innerHTML = formattedContent;
    
    // Add references if available
    if (references && references.length > 0) {
        const referencesDiv = document.createElement('div');
        referencesDiv.className = 'message-references';
        
        const referencesList = references.map((ref, index) => {
            return `<span class="reference" title="${ref.text.substring(0, 100)}...">Page ${ref.page_number}</span>`;
        }).join(', ');
        
        referencesDiv.innerHTML = `Sources: ${referencesList}`;
        messageDiv.appendChild(referencesDiv);
    }
    
    chatMessages.appendChild(messageDiv);
    
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
}
