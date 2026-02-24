/* script.js - Handles Chat Logic */

class SessionManager {
    constructor() {
        this.sessions = [];
        this.currentSessionId = null;
        // Load sessions immediately
        try {
            const saved = localStorage.getItem('ailee_sessions');
            if (saved) {
                this.sessions = JSON.parse(saved);
            }
        } catch (e) {
            console.error("Failed to load sessions:", e);
            this.sessions = [];
        }

        if (this.sessions.length === 0) {
            this.createNewSession();
        } else {
            const lastActive = localStorage.getItem('ailee_last_session_id');
            const session = this.sessions.find(s => s.id === lastActive);
            this.currentSessionId = session ? session.id : this.sessions[0].id;
        }
    }

    saveSessions() {
        localStorage.setItem('ailee_sessions', JSON.stringify(this.sessions));
        if (this.currentSessionId) {
            localStorage.setItem('ailee_last_session_id', this.currentSessionId);
        }
    }

    createNewSession() {
        const id = 'sess_' + Date.now();
        const newSession = {
            id: id,
            title: 'New Session',
            messages: [],
            timestamp: Date.now()
        };
        this.sessions.unshift(newSession);
        this.currentSessionId = id;
        this.saveSessions();
        return newSession;
    }

    getCurrentSession() {
        return this.sessions.find(s => s.id === this.currentSessionId);
    }

    updateSessionTitle(sessionId, newTitle) {
        const session = this.sessions.find(s => s.id === sessionId);
        if (session) {
            session.title = newTitle.substring(0, 30) + (newTitle.length > 30 ? '...' : '');
            this.saveSessions();
        }
    }

    addMessage(role, content, metadata = null) {
        const session = this.getCurrentSession();
        if (!session) return;

        const message = {
            role: role,
            content: content,
            metadata: metadata,
            timestamp: Date.now()
        };
        session.messages.push(message);

        // Update title on EVERY user message (shows the last user message as title)
        if (role === 'user') {
             this.updateSessionTitle(session.id, content);
        }

        this.saveSessions();
        return message;
    }

    switchSession(id) {
        if (this.sessions.some(s => s.id === id)) {
            this.currentSessionId = id;
            this.saveSessions();
            return true;
        }
        return false;
    }
}

// Global instances
const sessionManager = new SessionManager();

// DOM Elements (will be populated on load)
let sidebar, sessionList, messagesContainer, chatInput, sendBtn, toggleSidebarBtn, newSessionBtn;

document.addEventListener('DOMContentLoaded', () => {
    sidebar = document.querySelector('.sidebar');
    sessionList = document.querySelector('.session-list'); // Corrected selector
    messagesContainer = document.querySelector('.messages-container');
    chatInput = document.querySelector('.chat-input');
    sendBtn = document.querySelector('.send-btn');
    toggleSidebarBtn = document.querySelector('.sidebar-toggle');
    newSessionBtn = document.querySelector('.new-session-btn');

    // Initial Render
    renderSidebar();
    renderChat();

    // Event Listeners
    sendBtn.addEventListener('click', sendMessage);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });

    newSessionBtn.addEventListener('click', () => {
        sessionManager.createNewSession();
        renderSidebar();
        renderChat();
        if (window.innerWidth < 768 && sidebar && !sidebar.classList.contains('collapsed')) {
             sidebar.classList.add('collapsed');
        }
    });

    toggleSidebarBtn.addEventListener('click', () => {
        if (sidebar) sidebar.classList.toggle('collapsed');
    });
});

function renderSidebar() {
    if (!sessionList) return;
    sessionList.innerHTML = '';
    const currentId = sessionManager.currentSessionId;

    sessionManager.sessions.forEach(session => {
        const item = document.createElement('div');
        item.className = `session-item ${session.id === currentId ? 'active' : ''}`;
        item.textContent = session.title || 'New Session';
        item.onclick = () => {
            sessionManager.switchSession(session.id);
            renderSidebar();
            renderChat();
        };
        sessionList.appendChild(item);
    });
}

function renderChat() {
    if (!messagesContainer) return;
    messagesContainer.innerHTML = '';
    const session = sessionManager.getCurrentSession();
    if (!session) return;

    session.messages.forEach(msg => {
        appendMessageToUI(msg);
    });
    scrollToBottom();
}

function appendMessageToUI(msg) {
    if (!messagesContainer) return;

    const div = document.createElement('div');
    div.className = `message ${msg.role}`;

    let contentHtml = '';

    if (msg.role === 'user') {
        contentHtml = `<div class="message-content">${escapeHtml(msg.content)}</div>`;
    } else {
        const meta = msg.metadata || {};
        const status = meta.safety_status || 'UNKNOWN';
        let badgeClass = 'status-borderline';
        if (status === 'ACCEPTED') badgeClass = 'status-accepted';
        if (status === 'OUTRIGHT_REJECTED') badgeClass = 'status-rejected';

        // Escape content THEN replace newlines
        const safeText = escapeHtml(msg.content || '');
        const text = safeText.replace(/\n/g, '<br>');

        contentHtml = `
            <div class="message-content">${text}</div>
            <div class="message-meta">
                <span class="status-badge ${badgeClass}">${escapeHtml(status)}</span>
                <span>Score: ${meta.trust_score !== undefined ? Number(meta.trust_score).toFixed(1) : 'N/A'}</span>
            </div>
            <button class="details-toggle" onclick="var d = this.nextElementSibling; d.style.display = d.style.display === 'block' ? 'none' : 'block'">Show Details</button>
            <div class="json-details" style="display:none;">${escapeHtml(JSON.stringify(meta, null, 2))}</div>
        `;
    }

    div.innerHTML = contentHtml;
    messagesContainer.appendChild(div);
}

async function sendMessage() {
    if (!chatInput) return;
    const text = chatInput.value.trim();
    if (!text) return;

    chatInput.value = '';
    chatInput.disabled = true;
    sendBtn.disabled = true;

    // Add User Message
    const userMsg = sessionManager.addMessage('user', text);
    appendMessageToUI(userMsg);
    scrollToBottom();
    renderSidebar();

    // Add Loading Indicator
    const loadingId = 'loading-' + Date.now();
    const loadingDiv = document.createElement('div');
    loadingDiv.id = loadingId;
    loadingDiv.className = 'message ai';
    loadingDiv.innerHTML = '<em>Thinking... (Searching & Validating)</em>';
    messagesContainer.appendChild(loadingDiv);
    scrollToBottom();

    try {
        const response = await fetch(`/trust?query=${encodeURIComponent(text)}&format=json`);

        const loadingEl = document.getElementById(loadingId);
        if (loadingEl) loadingEl.remove();

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const data = await response.json();

        // data: { query, trusted_answer, safety_status, trust_score, ... }

        // Use trusted_answer for display content
        const content = data.trusted_answer || "No answer generated.";
        const aiMsg = sessionManager.addMessage('ai', content, data);
        appendMessageToUI(aiMsg);
        scrollToBottom();

    } catch (error) {
        const loadingEl = document.getElementById(loadingId);
        if (loadingEl) loadingEl.remove();

        // Save error message to session so it persists
        const errorMsg = sessionManager.addMessage('ai', "Error: " + error.message, { safety_status: 'ERROR' });
        appendMessageToUI(errorMsg);
    } finally {
        chatInput.disabled = false;
        sendBtn.disabled = false;
        chatInput.focus();
    }
}

function scrollToBottom() {
    if (messagesContainer) {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
}

function escapeHtml(text) {
    if (!text) return '';
    return String(text)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}
