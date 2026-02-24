/* script.js - Handles Chat Logic */

class SessionManager {
    constructor() {
        this.sessions = [];
        this.currentSessionId = null;
        this.loadSessions();
    }

    loadSessions() {
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
            session.title = newTitle.substring(0, 40) + (newTitle.length > 40 ? '...' : '');
            this.saveSessions();
        }
    }

    renameSession(sessionId, newName) {
        const session = this.sessions.find(s => s.id === sessionId);
        if (session && newName.trim()) {
            session.title = newName.trim();
            this.saveSessions();
            return true;
        }
        return false;
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

        // Auto-update title based on last user message
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

    exportSession(format = 'json') {
        const session = this.getCurrentSession();
        if (!session) return;

        let content = '';
        let mimeType = 'application/json';
        let filename = `session-${session.id}.${format}`;

        if (format === 'json') {
            content = JSON.stringify(session, null, 2);
        } else {
            // Text format
            mimeType = 'text/plain';
            content = `Session: ${session.title}\nDate: ${new Date(session.timestamp).toLocaleString()}\n\n`;
            session.messages.forEach(msg => {
                const role = msg.role.toUpperCase();
                content += `[${role}]: ${msg.content}\n`;
                if (msg.role === 'ai' && msg.metadata) {
                    content += `(Trust Score: ${msg.metadata.trust_score}, Status: ${msg.metadata.safety_status})\n`;
                }
                content += '-'.repeat(40) + '\n';
            });
        }

        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
}

class ThemeManager {
    constructor() {
        this.toggleBtn = document.getElementById('theme-toggle');
        this.init();
    }

    init() {
        const savedTheme = localStorage.getItem('ailee_theme') || 'light';
        document.documentElement.setAttribute('data-theme', savedTheme);
        this.updateIcon(savedTheme);

        this.toggleBtn.addEventListener('click', () => {
            const current = document.documentElement.getAttribute('data-theme');
            const newTheme = current === 'dark' ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', newTheme);
            localStorage.setItem('ailee_theme', newTheme);
            this.updateIcon(newTheme);
        });
    }

    updateIcon(theme) {
        // SVG logic inside button handled by pure HTML/CSS state if needed,
        // or we assume generic icon works for toggle.
    }
}

class SidebarManager {
    constructor() {
        this.sidebar = document.querySelector('.sidebar');
        this.handle = document.querySelector('.resize-handle');
        this.toggleBtn = document.getElementById('sidebar-toggle');
        this.isResizing = false;

        this.init();
    }

    init() {
        // Load saved width
        const savedWidth = localStorage.getItem('ailee_sidebar_width');
        if (savedWidth && window.innerWidth > 768) {
            this.sidebar.style.width = `${savedWidth}px`;
        }

        // Mobile Auto-Collapse on Init
        if (window.innerWidth <= 768) {
             this.sidebar.classList.add('collapsed');
        }

        // Toggle logic
        this.toggleBtn.addEventListener('click', () => {
            this.sidebar.classList.toggle('collapsed');
            if (window.innerWidth <= 768) {
                this.sidebar.classList.toggle('mobile-open');
            }
        });

        // Resize logic
        this.handle.addEventListener('mousedown', (e) => {
            this.isResizing = true;
            document.body.style.cursor = 'col-resize';
            this.handle.classList.add('resizing');
        });

        document.addEventListener('mousemove', (e) => {
            if (!this.isResizing) return;

            // Constrain width
            let newWidth = e.clientX;
            if (newWidth < 200) newWidth = 200;
            if (newWidth > 500) newWidth = 500;

            this.sidebar.style.width = `${newWidth}px`;
        });

        document.addEventListener('mouseup', () => {
            if (this.isResizing) {
                this.isResizing = false;
                document.body.style.cursor = '';
                this.handle.classList.remove('resizing');
                localStorage.setItem('ailee_sidebar_width', parseInt(this.sidebar.style.width));
            }
        });
    }
}

// Global instances
const sessionManager = new SessionManager();
let uiComponents = {};

document.addEventListener('DOMContentLoaded', () => {
    // Initialize UI Components
    uiComponents.theme = new ThemeManager();
    uiComponents.sidebar = new SidebarManager();

    // DOM Elements
    const sessionList = document.getElementById('session-list');
    const messagesContainer = document.getElementById('messages-container');
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    const newSessionBtn = document.getElementById('new-session-btn');
    const exportBtn = document.getElementById('export-btn');
    const exportMenu = document.getElementById('export-menu');

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
        if (window.innerWidth <= 768) {
             document.querySelector('.sidebar').classList.add('collapsed');
        }
    });

    // Export Logic
    exportBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        exportMenu.style.display = exportMenu.style.display === 'block' ? 'none' : 'block';
    });

    document.addEventListener('click', () => {
        if (exportMenu) exportMenu.style.display = 'none';
    });

    document.querySelectorAll('.export-option').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const format = e.target.getAttribute('data-format');
            sessionManager.exportSession(format);
        });
    });
});

function renderSidebar() {
    const sessionList = document.getElementById('session-list');
    sessionList.innerHTML = '';
    const currentId = sessionManager.currentSessionId;

    sessionManager.sessions.forEach(session => {
        const item = document.createElement('div');
        item.className = `session-item ${session.id === currentId ? 'active' : ''}`;

        // Title container
        const titleSpan = document.createElement('span');
        titleSpan.className = 'session-title';
        titleSpan.textContent = session.title || 'New Session';

        // Rename on double click
        titleSpan.addEventListener('dblclick', (e) => {
            e.stopPropagation(); // Prevent switch
            const input = document.createElement('input');
            input.type = 'text';
            input.className = 'rename-input';
            input.value = session.title;

            input.addEventListener('blur', () => {
                sessionManager.renameSession(session.id, input.value);
                renderSidebar();
            });

            input.addEventListener('keypress', (ev) => {
                if (ev.key === 'Enter') input.blur();
            });

            titleSpan.replaceWith(input);
            input.focus();
        });

        item.appendChild(titleSpan);

        item.onclick = (e) => {
            // Don't switch if clicking input
            if (e.target.tagName === 'INPUT') return;
            sessionManager.switchSession(session.id);
            renderSidebar();
            renderChat();

            if (window.innerWidth <= 768) {
                 document.querySelector('.sidebar').classList.add('collapsed');
            }
        };

        sessionList.appendChild(item);
    });
}

function renderChat() {
    const messagesContainer = document.getElementById('messages-container');
    messagesContainer.innerHTML = '';
    const session = sessionManager.getCurrentSession();
    if (!session) return;

    session.messages.forEach(msg => {
        appendMessageToUI(msg);
    });
    scrollToBottom();
}

function appendMessageToUI(msg, animate = false) {
    const messagesContainer = document.getElementById('messages-container');
    const div = document.createElement('div');
    div.className = `message ${msg.role}`;

    let contentHtml = '';

    // Markdown parsing WITH SANITIZATION
    const rawContent = msg.content;
    let parsedContent;

    if (typeof marked !== 'undefined') {
        const rawMarkdown = marked.parse(rawContent);
        parsedContent = (typeof DOMPurify !== 'undefined')
            ? DOMPurify.sanitize(rawMarkdown)
            : rawMarkdown; // Fallback if purify missing (shouldn't happen with CDN)
    } else {
        parsedContent = escapeHtml(rawContent).replace(/\n/g, '<br>');
    }

    if (msg.role === 'user') {
        contentHtml = `<div class="message-content">${parsedContent}</div>`;
        div.innerHTML = contentHtml;
    } else {
        const meta = msg.metadata || {};
        const status = meta.safety_status || 'UNKNOWN';
        let badgeClass = 'status-borderline';
        if (status === 'ACCEPTED') badgeClass = 'status-accepted';
        if (status === 'OUTRIGHT_REJECTED') badgeClass = 'status-rejected';

        contentHtml = `
            <div class="message-content">${parsedContent}</div>
            <div class="message-meta">
                <span class="status-badge ${badgeClass}">${escapeHtml(status)}</span>
                <span>Score: ${meta.trust_score !== undefined ? Number(meta.trust_score).toFixed(1) : 'N/A'}</span>
            </div>
            <button class="details-toggle" onclick="var d = this.nextElementSibling; d.style.display = d.style.display === 'block' ? 'none' : 'block'">Details</button>
            <div class="json-details" style="display:none;">${escapeHtml(JSON.stringify(meta, null, 2))}</div>
        `;
        div.innerHTML = contentHtml;
    }

    messagesContainer.appendChild(div);
    return div;
}

async function sendMessage() {
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    const messagesContainer = document.getElementById('messages-container');

    const text = chatInput.value.trim();
    if (!text) return;

    chatInput.value = '';
    chatInput.disabled = true;
    sendBtn.disabled = true;

    // Add User Message
    const userMsg = sessionManager.addMessage('user', text);
    appendMessageToUI(userMsg, true);
    scrollToBottom();
    renderSidebar();

    // Add Typing Indicator
    const loadingId = 'loading-' + Date.now();
    const loadingDiv = document.createElement('div');
    loadingDiv.id = loadingId;
    loadingDiv.className = 'message ai typing-indicator-container';
    loadingDiv.innerHTML = `
        <span class="typing-indicator">
            AILEE is thinking
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
        </span>`;
    messagesContainer.appendChild(loadingDiv);
    scrollToBottom();

    try {
        const response = await fetch(`/trust?query=${encodeURIComponent(text)}&format=json`);

        // Remove loading
        const loadingEl = document.getElementById(loadingId);
        if (loadingEl) loadingEl.remove();

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const data = await response.json();
        const content = data.trusted_answer || "No answer generated.";

        // Add AI Message to data
        const aiMsg = sessionManager.addMessage('ai', content, data);

        // Streaming Effect Simulation
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message ai';
        messagesContainer.appendChild(msgDiv);

        const meta = data || {};
        const status = meta.safety_status || 'UNKNOWN';
        let badgeClass = 'status-borderline';
        if (status === 'ACCEPTED') badgeClass = 'status-accepted';
        if (status === 'OUTRIGHT_REJECTED') badgeClass = 'status-rejected';

        let i = 0;
        const speed = 10;

        // Create container structure
        msgDiv.innerHTML = `
            <div class="message-content"></div>
            <div class="message-meta" style="opacity:0; transition:opacity 0.5s">
                <span class="status-badge ${badgeClass}">${escapeHtml(status)}</span>
                <span>Score: ${meta.trust_score !== undefined ? Number(meta.trust_score).toFixed(1) : 'N/A'}</span>
            </div>
            <button class="details-toggle" style="opacity:0; transition:opacity 0.5s" onclick="var d = this.nextElementSibling; d.style.display = d.style.display === 'block' ? 'none' : 'block'">Details</button>
            <div class="json-details" style="display:none;">${escapeHtml(JSON.stringify(meta, null, 2))}</div>
        `;

        const contentDiv = msgDiv.querySelector('.message-content');

        function typeWriter() {
            if (i < content.length) {
                const chunk = content.substring(i, i + 3);
                i += 3;
                contentDiv.textContent += chunk;
                scrollToBottom();
                setTimeout(typeWriter, speed);
            } else {
                // Done typing: Parse Markdown + SANITIZE
                if (typeof marked !== 'undefined') {
                    const rawMarkdown = marked.parse(content);
                    const safeMarkdown = (typeof DOMPurify !== 'undefined')
                        ? DOMPurify.sanitize(rawMarkdown)
                        : rawMarkdown;
                    contentDiv.innerHTML = safeMarkdown;
                }
                msgDiv.querySelector('.message-meta').style.opacity = 1;
                msgDiv.querySelector('.details-toggle').style.opacity = 1;
                scrollToBottom();
            }
        }

        typeWriter();

    } catch (error) {
        const loadingEl = document.getElementById(loadingId);
        if (loadingEl) loadingEl.remove();

        const errorMsg = sessionManager.addMessage('ai', "Error: " + error.message, { safety_status: 'ERROR' });
        appendMessageToUI(errorMsg);
    } finally {
        chatInput.disabled = false;
        sendBtn.disabled = false;
        chatInput.focus();
    }
}

function scrollToBottom() {
    const messagesContainer = document.getElementById('messages-container');
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
